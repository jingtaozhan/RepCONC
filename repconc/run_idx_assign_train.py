# coding=utf-8
import logging
import os
from tqdm import tqdm
import numpy as np
import torch
import faiss
from packaging import version
from torch import nn
from typing import Any, Dict, Union
from torch.cuda.amp import autocast
from dataclasses import dataclass, field
import transformers
from transformers import (
    RobertaConfig, RobertaTokenizer,
    HfArgumentParser, Trainer,
    set_seed, TrainingArguments
)
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback, TrainerCallback
from transformers.optimization import Adafactor, AdamW, get_scheduler

from repconc.model import RepCONC
from repconc.dataset import (TextTokenIdsCache, load_rel, 
    TrainInbatchWithHardDataset,
    triple_get_collate_function,
)
logger = logging.Logger(__name__)


class RepCONCTrainerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True


class RepCONCTensorBoardCallback(TensorBoardCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        pass

def is_main_process(local_rank):
    return local_rank in [-1, 0]


@dataclass
class DataTrainingArguments:
    preprocess_dir: str = field()
    label_path: str = field()
    hardneg_path: str = field() 
    max_query_length: int = field()
    max_doc_length: int = field() 


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    MCQ_M: int = field()
    MCQ_K: int = field()
    opq_path: str = field()
    init_model_path: str = field()
    gradient_checkpointing: bool = field(default=False)    


@dataclass
class RepCONCTrainingArguments(TrainingArguments):
    mse_weight: float = field(default=1.0)
    sk_epsilon: float=field(default=0.1)
    sk_iters: int=field(default=30)
    multibatch_per_forward: int = field(default=1, metadata={"help": "At each training step, compute representations with mutiple batches to save peak cuda memory."})
    continue_train: bool = field(default=False, metadata={"help": "Resume training."})
    no_constraint: bool = field(default=False)

    per_device_train_batch_size: int = field(default=128)
    learning_rate: float = field(default=5e-6, metadata={"help": "The initial learning rate for Adam."})
    centroid_lr: float = field(default=5e-6)
    weight_decay: float = field(default=0.001, metadata={"help": "Weight decay if we apply some."})

    logging_steps: int = field(default=25, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=99999999999, metadata={"help": "Save checkpoint every X updates steps."})
    seed: int = field(default=521, metadata={"help": "random seed for initialization"})


class RepCONCTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        We override this function to backpropogate gradient to centroids
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "centroids" not in n and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if "centroids" not in n and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {   "params": [p for n, p in self.model.named_parameters() if "centroids" in n ],
                    "weight_decay": 0.0, 'lr': self.args.centroid_lr
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_dpp:
                raise NotImplementedError()
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        """
        We override this function to log both ranking loss and the mse loss.
        If the mse loss increases, training becomes unstable. 
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # my log begin
            logs["rank_loss"] = round(self._my_log_rank_loss.item() / (self.state.global_step - self._globalstep_last_logged), 4)
            self._my_log_rank_loss -= self._my_log_rank_loss
            logs["mse_loss"] = round(self._my_log_mse_loss.item() / (self.state.global_step - self._globalstep_last_logged), 4)
            self._my_log_mse_loss -= self._my_log_mse_loss
            # end

            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )

            lr = logs["learning_rate"]
            logs["learning_rate"] = float(f"{lr:.3e}")
            
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        We override this function to log both ranking loss and the mse loss.
        If the mse loss increases, training becomes unstable. 
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss, (_, rank_loss, bm25_loss) = self.compute_loss(model, inputs, return_outputs=True)
        else:
            loss, (_, rank_loss, bm25_loss) = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            rank_loss = rank_loss.mean()
            bm25_loss = bm25_loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            rank_loss = rank_loss / self.args.gradient_accumulation_steps
            bm25_loss = bm25_loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            raise NotImplementedError()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        # log rank_loss and bm25_loss 
        if not hasattr(self, "_my_log_rank_loss"):
            self._my_log_rank_loss = torch.tensor(0.0).to(self.args.device)
        if not hasattr(self, "_my_log_mse_loss"):
            self._my_log_mse_loss = torch.tensor(0.0).to(self.args.device)
        self._my_log_rank_loss += rank_loss.detach()
        self._my_log_mse_loss += bm25_loss.detach()

        return loss.detach()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, RepCONCTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert os.path.exists(model_args.opq_path) and model_args.MCQ_K == 256

    resume_model_path = None
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
    ):
        if training_args.continue_train:
            ckpts = os.listdir(training_args.output_dir)
            ckpts = list(filter(lambda x: x.startswith("checkpoint"), ckpts))
            ckpts = sorted(ckpts, key=lambda x:int(x.split("-")[1]))
            resume_model_path = os.path.join(training_args.output_dir, ckpts[-1])
        elif not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
    logger.info(f"Resume model path: {resume_model_path}")

    assert is_main_process(training_args.local_rank) and training_args.n_gpu==1, "The script only supports single-gpu training"
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = RobertaConfig.from_pretrained(
        model_args.init_model_path,
        gradient_checkpointing=model_args.gradient_checkpointing,
        MCQ_M=model_args.MCQ_M, MCQ_K=model_args.MCQ_K,
        return_dict=False
    )
    config.MCQ_M, config.MCQ_K = model_args.MCQ_M, model_args.MCQ_K
    config.small_batch_num = training_args.multibatch_per_forward
    config.sk_epsilon, config.sk_iters = training_args.sk_epsilon, training_args.sk_iters 
    config.mse_weight = training_args.mse_weight
    if training_args.no_constraint:
        config.no_constraint = True
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.init_model_path,
    )

    rel_dict = load_rel(data_args.label_path)
    train_dataset = TrainInbatchWithHardDataset(
        rel_file=data_args.label_path,
        rank_file=data_args.hardneg_path,
        queryids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="train-query"),
        docids_cache=TextTokenIdsCache(data_dir=data_args.preprocess_dir, prefix="passages"),
        max_query_length=data_args.max_query_length,
        max_doc_length=data_args.max_doc_length,
        hard_num=1
    )
    data_collator = triple_get_collate_function(
        data_args.max_query_length, data_args.max_doc_length,
        rel_dict=rel_dict, padding=True)
    model = RepCONC.from_pretrained(
        model_args.init_model_path,
        config=config
    )
    print("init_model_with_opq")
    init_model_with_opq(model, model_args.opq_path)

    # Initialize our Trainer
    trainer = RepCONCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.remove_callback(TensorBoardCallback)
    trainer.add_callback(RepCONCTensorBoardCallback(
        tb_writer=SummaryWriter(training_args.logging_dir)))
    trainer.add_callback(RepCONCTrainerCallback())

    # Training
    trainer.train(resume_model_path)
    trainer.save_model()  # Saves the tokenizer too for easy upload


def init_model_with_opq(model:RepCONC, index_path:str):
    opqindex = faiss.read_index(index_path)
    assert isinstance(opqindex, faiss.IndexPreTransform)
    vt = faiss.downcast_VectorTransform(opqindex.chain.at(0))            
    assert isinstance(vt, faiss.LinearTransform)
    rotation = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
    model.rotation.copy_(torch.from_numpy(rotation))

    index = faiss.downcast_index(opqindex.index)
    pq = index.pq
    centroids = faiss.vector_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
    assert pq.nbits == 8
    model.centroids.data.copy_(torch.from_numpy(centroids))
    

if __name__ == "__main__":
    main()
