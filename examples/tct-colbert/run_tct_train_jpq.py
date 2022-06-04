import os
import sys
import faiss
import logging
import numpy as np
import transformers
from transformers import (
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import is_main_process

from repconc.models.jpq.finetune_jpq import (
    JPQ, 
    QueryDataset,
    JPQFinetuneArguments, 
    FinetuneQueryCollator,
    JPQFinetuner, 
    DataTrainingArguments
)
from repconc.train.run_train_jpq import (
    ModelArguments,
    load_validation_set
)

from modeling_tct import TCTTokenizerFast, tct_repconc_from_pretrained

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, JPQFinetuneArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    
    resume_from_checkpoint = False
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and any((x.startswith("checkpoint") for x in os.listdir(training_args.output_dir)))
    ):
        if not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        else:
            resume_from_checkpoint = True

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: JPQFinetuneArguments

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training parameters %s", training_args)

    assert training_args.n_gpu == 1, "Not sure whether Faiss supports distributed training with Pytorch."
    # Set seed before initializing model.
    set_seed(training_args.seed)

    ### Only here is modified
    tokenizer = TCTTokenizerFast.from_pretrained(model_args.model_name_or_path)
    repconc = tct_repconc_from_pretrained(model_args.model_name_or_path, use_constraint=False, sk_epsilon=None, sk_iters=None)
    ###

    pq_index = faiss.read_index(os.path.join(model_args.index_input_dir, "index"))
    corpus_ids = np.load(os.path.join(model_args.index_input_dir, "corpus_ids.npy"))
    valid_queries, valid_qrels = load_validation_set(data_args.valid_query_path, data_args.valid_qrel_path)

    train_set = QueryDataset(tokenizer, 
            qrel_path = data_args.qrel_path, 
            query_path = data_args.query_path, 
            max_query_len = data_args.max_query_len, 
            index_doc_ids = corpus_ids,
            rel_threshold=1, 
            verbose=is_main_process(training_args.local_rank))
    # Data collator
    data_collator = FinetuneQueryCollator(
        tokenizer=tokenizer,
        max_query_len = data_args.max_query_len, 
    )

    jpq = JPQ(
        repconc = repconc, 
        pq_index = pq_index, 
        qrels = train_set.get_qrels(), 
        neg_top_k = training_args.dynamic_topk_negative, 
        temperature = training_args.temperature,
        gpu_id = 0 if training_args.local_rank == -1 else training_args.local_rank,
    )
    # Initialize our Trainer
    trainer = JPQFinetuner(
        model=jpq,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        eval_dataset=(corpus_ids, valid_queries, valid_qrels)
    )
    # additionally save checkpoint at the end of one epoch
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # make sure load_best_model_at_end = True
    trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
