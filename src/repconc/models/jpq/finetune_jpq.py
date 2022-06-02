import os
import faiss
import logging
import torch
import random
import numpy as np
import torch.distributed as dist
import faiss.contrib.torch_utils
from tqdm import tqdm
from torch import nn, Tensor
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Union, List, Dict, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import TrainingArguments, Trainer
from transformers.integrations import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.file_utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    raise NotImplementedError()
    import smdistributed.modelparallel.torch as smp

from repconc.models.repconc import RepCONC
from repconc.models.repconc.finetune_repconc import RepCONC_Norm_Centroid_Callback
from repconc.models.repconc.evaluate_repconc import from_pq_to_ivfpq, load_index_to_gpu
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    qrel_path: str = field()
    query_path: str = field()
    valid_qrel_path: str = field()
    valid_query_path: str = field()
    max_query_len: int = field()


@dataclass
class JPQFinetuneArguments(TrainingArguments):
    dynamic_topk_negative: int = field(default=200)
    centroid_learning_rate: float = field(default=1e-3)
    temperature: float = field(default=1.0)
    seed: int = field(default=2023)

    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    


@dataclass
class FinetuneQueryCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_query_len: int ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenizing batch of text is much faster
        query_input = self.tokenizer(
            [x['query'] for x in features],
            padding=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_query_len
        )
        # we have to prevent inbatch false negatives when gathering tensors in the trainer
        # because each distributed process has its own collators
        qids = torch.tensor([x['qid'] for x in features], dtype=torch.long)

        batch_data = {
                "query_input_ids": query_input["input_ids"],
                "query_attention_mask": query_input["attention_mask"],
                "qids": qids,
            }
        return batch_data


class QueryDataset(Dataset):
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            qrel_path: str, 
            query_path: str, 
            max_query_len: int, 
            index_doc_ids: np.ndarray,
            rel_threshold=1,
            verbose=True):
        '''
        negative: choices from `inbatch', `random', or a path to a json file that contains \
            the qid:neg_pid_lst  
        '''
        super().__init__()
        self.tokenizer = tokenizer

        docid2offset = dict(((str(docid), idx) for idx, docid in enumerate(index_doc_ids)))
        self.queries, qid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(query_path), disable=not verbose, mininterval=10)):
            qid, query = line.split("\t")
            qid2offset[qid] = idx
            self.queries.append(query.strip())

        self.qrels = defaultdict(list)
        for line in tqdm(open(qrel_path), disable=not verbose, mininterval=10):
            qid, _, docid, rel = line.split()
            if int(rel) >= rel_threshold:
                qoffset = qid2offset[qid]
                docoffset = docid2offset[docid]
                self.qrels[qoffset].append(docoffset)

        self.qids = sorted(self.qrels.keys())
        self.max_query_len = max_query_len
        self.qrels = dict(self.qrels)

    def get_qrels(self):
        return self.qrels

    def __len__(self):
        return len(self.qids)
    
    def __getitem__(self, index):
        '''
        We do not tokenize text here and instead tokenize batch of text in the collator because
            a. Tokenizing batch of text is much faster then tokenizing one by one
            b. Usually, the corpus is too large and we cannot afford to use multiple num workers
        '''
        qid = self.qids[index]
        query = self.queries[qid]
        data = {
            "query": query,
            "qid": qid
        }
        return data


class JPQ(nn.Module):
    def __init__(self, 
            repconc: RepCONC,
            pq_index: faiss.IndexIVFPQ, 
            qrels: Dict[int, List[int]],
            neg_top_k: int,
            temperature: float,
            gpu_id: int,
        ):
        super().__init__()
        self.repconc = repconc
        self.qrels = qrels
        self.neg_top_k = neg_top_k
        self.temperature = temperature
        self.pq_index = pq_index
        self.cpu_ivf_index = from_pq_to_ivfpq(pq_index)
        self.gpu_id = gpu_id
        self.synchronize_model_index()
        codes = torch.from_numpy(
            faiss.vector_to_array(self.pq_index.codes).astype(np.int64).reshape(-1, repconc.config.MCQ_M))
        self.register_buffer("codes", codes)
        
    def forward(self, 
            query_input_ids: torch.Tensor, 
            query_attention_mask: torch.Tensor, 
            qids: torch.Tensor):
        # query_bs, dim
        query_embeds = self.repconc(
            query_input_ids, 
            query_attention_mask, 
            return_code = False,
            return_quantized_embedding = False).continuous_embeds
        # query_bs,
        # query_bs, k
        neg_pids = self.gpu_ivf_index.search(query_embeds.detach(), self.neg_top_k)[1]
        assert query_embeds.device == neg_pids.device, f"{query_embeds.device} == {neg_pids.device}"
        # query_bs, k, dim
        neg_doc_embeds = self.repconc.decode(self.codes[neg_pids.reshape(-1)]).reshape(*neg_pids.shape, -1)
        neg_masks = self._compute_negative_mask(qids, neg_pids)
        # query_bs, k
        query_negdoc_scores = (query_embeds.unsqueeze(1) * neg_doc_embeds).sum(-1) / self.temperature
        
        # query_bs,
        pos_pids = torch.LongTensor([random.choice(self.qrels[qid.item()]) for qid in qids]).to(neg_pids.device)
        # query_bs, dim
        rel_doc_embeds = self.repconc.decode(self.codes[pos_pids])
        # query_bs, 1
        query_reldoc_scores = (query_embeds * rel_doc_embeds).sum(-1, keepdim=True) / self.temperature

        loss = self.compute_loss(query_reldoc_scores, query_negdoc_scores, neg_masks)
        return {"loss": loss}

    @torch.no_grad()
    def _compute_negative_mask(self, qids, docids):
        '''
        qids: N
        docids: N x K
        return: N x K
        '''
        negative_mask = torch.zeros(docids.shape, dtype=torch.bool, device=docids.device)
        for i, qid in enumerate(qids):
            for d in self.qrels[qid.item()]:
                negative_mask[i] = torch.logical_or(negative_mask[i], docids[i]==d)
        negative_mask = negative_mask.type(torch.float32)
        return negative_mask

    @torch.no_grad()
    def synchronize_model_index(self):
        self.gpu_ivf_index = None
        centroids = self.repconc.centroids.data.detach().cpu().numpy().ravel()
        faiss.copy_array_to_vector(centroids, self.pq_index.pq.centroids)
        faiss.copy_array_to_vector(centroids, self.cpu_ivf_index.pq.centroids)
        self.gpu_ivf_index = load_index_to_gpu(self.cpu_ivf_index, self.gpu_id)
    
    @torch.no_grad()
    def normalize_centrodis(self):
        self.repconc.normalize_centrodis()

    def state_dict(self):
        return self.repconc.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        '''
        for resume training
        '''
        missing = self.repconc.load_state_dict(state_dict, strict)
        self.synchronize_model_index()
        return missing

    def compute_loss(self, query_reldoc_scores, query_negdoc_scores, neg_masks):
        '''
        query_reldoc_scores: N, 1
        query_negdoc_scores: N, K
        neg_masks: N, K
        '''
        query_doc_scores = torch.hstack((query_reldoc_scores, query_negdoc_scores))
        # compute contrastive loss
        labels = torch.zeros(query_doc_scores.size(0)).long().to(query_doc_scores.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(query_doc_scores, labels)
    
        return loss


class JPQ_SyncIndex_Callback(TrainerCallback):
    def on_step_end(self, *args, **kwargs):
        """
        Event called at the end of a training step. Normalize centroids if necessary.
        """
        model: JPQ = kwargs.pop('model')
        model.synchronize_model_index()


class JPQFinetuner(Trainer):
    def __init__(self, *args, **kwargs):
        super(JPQFinetuner, self).__init__(*args, **kwargs)
        self.args: JPQFinetuneArguments
        self.model: JPQ
        # it is important to first normalize the centroids
        # and then sync index?
        if self.model.repconc.config.similarity_metric == "METRIC_CENTROID_COS":
            self.add_callback(RepCONC_Norm_Centroid_Callback)
        self.add_callback(JPQ_SyncIndex_Callback)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.model.repconc.config.save_pretrained(output_dir)
        self.model.repconc.dense_encoder.save_pretrained(os.path.join(output_dir, 'dense_encoder'))
    
    def evaluate(self, 
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",) -> Dict[str, float]:
        torch.cuda.empty_cache()
        from repconc.models.repconc.evaluate_repconc import encode_query, batch_search, from_pq_to_ivfpq, load_index_to_gpu
        from repconc.utils.eval_utils import pytrec_evaluate

        index = self.model.pq_index
        corpus_ids, queries, qrels = self.eval_dataset

        fp16, bf16 = self.args.fp16, self.args.bf16
        self.args.fp16, self.args.bf16 = False, False
        dataloader_drop_last = self.args.dataloader_drop_last
        self.args.dataloader_drop_last = False
        query_embeds, query_ids = encode_query(queries, self.model.repconc, self.tokenizer, 512, self.args)        
        self.args.fp16, self.args.bf16 = fp16, bf16
        self.args.dataloader_drop_last = dataloader_drop_last

        torch.cuda.empty_cache()
        index = from_pq_to_ivfpq(index) # only gpuivfpq is supported
        index = load_index_to_gpu(index, 0 if self.args.local_rank < 0 else self.args.local_rank)
        all_topk_scores, all_topk_ids = batch_search(
            query_ids, query_embeds.astype(np.float32), # fp16
            corpus_ids, index, 
            topk=10, 
            batch_size=512)

        run_results = defaultdict(dict)
        for qid, topk_scores, topk_ids in zip(query_ids, all_topk_scores, all_topk_ids):
            for i, (score, docid) in enumerate(zip(topk_scores, topk_ids)):
                run_results[str(qid.item())][str(docid.item())] = score.item()
        metrics = {}
        for category, cat_metrics in pytrec_evaluate(
                qrels, 
                dict(run_results), 
                k_values =(10, ),
                mrr_k_values = (10, ),).items():
            if category == "perquery":
                continue
            for metric, score in cat_metrics.items():
                metrics[f"{metric_key_prefix}_{metric}"] = score
        torch.cuda.empty_cache()
        self.log(metrics)
       
        return metrics

    def create_optimizer(self):
        from transformers.trainer_pt_utils import (get_parameter_names,)
        from transformers.trainer_utils import ShardedDDPOption
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "centroids" not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "centroids" not in n],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if "centroids" in n],
                    "weight_decay": 0.0,
                    'lr': self.args.centroid_learning_rate
                },
            ]
            logger.info(f"optimizer_grouped_parameters: {[len(x['params']) for x in optimizer_grouped_parameters]}")
            logger.info(f"named_parameters: {len(list(self.model.named_parameters()))}")
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from fairscale.optim import OSS
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes
                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

