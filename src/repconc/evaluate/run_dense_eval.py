import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import field, dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments,
    set_seed)
from transformers.trainer_utils import is_main_process


from repconc.models.dense import AutoDense
from repconc.models.dense.evaluate_dense import (
    encode_dense_corpus, 
    encode_dense_query,
    batch_dense_search,
    create_index,
)
from repconc.utils.eval_utils import (
    load_corpus,
    load_queries,
    pytrec_evaluate
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    corpus_path: str = field()
    out_corpus_dir: str = field()

    query_path: str = field()
    out_query_dir: str = field()
    qrel_path: str = field(default=None)

    save_corpus_embed: bool = field(default=False)
    save_query_embed: bool = field(default=False)


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    max_seq_length: Optional[int] = field(default=512)
    similarity_metric: str = field(default=None, metadata={"help": "if is None, keep the original values"})
    pooling: str = field(default=None, metadata={"help": "if None, keep the original values", "choices": ["cls", "mean"]})


@dataclass
class EvalArguments(TrainingArguments):
    topk : int = field(default=100)
    search_threads: int = field(default=60)
    search_batch: int = field(default=1200)

    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )


def load_or_encode_query(model, tokenizer, query_path, out_query_dir, model_args, data_args, eval_args):
    out_query_embed_path = os.path.join(out_query_dir, "query_embeds.npy")
    out_query_ids_path = os.path.join(out_query_dir, "qids.npy")
    if os.path.exists(out_query_embed_path) and os.path.exists(out_query_ids_path):
        if is_main_process(eval_args.local_rank):
            query_embeds = np.load(out_query_embed_path)
            query_ids = np.load(out_query_ids_path)
            logger.info("Load pre-computed query representations")
        else:
            query_embeds, query_ids = None, None
    else:
        queries = load_queries(query_path)
        query_embeds, query_ids = encode_dense_query(queries, model, tokenizer, model_args.max_seq_length, eval_args)
        if is_main_process(eval_args.local_rank) and data_args.save_query_embed:
            os.makedirs(out_query_dir, exist_ok=True)
            np.save(out_query_embed_path, query_embeds)
            np.save(out_query_ids_path, query_ids)
    return query_embeds, query_ids


def load_or_encode_corpus(model, tokenizer, model_args, data_args, eval_args):
    out_corpus_embed_path = os.path.join(data_args.out_corpus_dir, "corpus_embeds.npy")
    out_corpus_ids_path = os.path.join(data_args.out_corpus_dir, "corpus_ids.npy")
    if os.path.exists(out_corpus_embed_path) and os.path.exists(out_corpus_ids_path):
        if is_main_process(eval_args.local_rank):
            corpus_embeds = np.load(out_corpus_embed_path)
            corpus_ids = np.load(out_corpus_ids_path)
            logger.info("Load pre-computed corpus representations")
        else:
            corpus_embeds, corpus_ids = None, None
    else:
        corpus = load_corpus(data_args.corpus_path, tokenizer.sep_token, verbose=is_main_process(eval_args.local_rank))
        corpus_embeds, corpus_ids = encode_dense_corpus(corpus, model, tokenizer, model_args.max_seq_length, eval_args)
        if is_main_process(eval_args.local_rank) and data_args.save_corpus_embed:
            os.makedirs(data_args.out_corpus_dir, exist_ok=True)
            np.save(out_corpus_embed_path, corpus_embeds)
            np.save(out_corpus_ids_path, corpus_ids)
    return corpus_embeds, corpus_ids


def search_and_compute_metrics(corpus_embeds, corpus_ids, query_embeds, query_ids, out_metric_path, out_query_dir, qrel_path, eval_args):
    index = create_index(corpus_embeds)
    all_topk_scores, all_topk_ids = batch_dense_search(
        query_ids, query_embeds, 
        corpus_ids, index, eval_args.topk, 
        batch_size=eval_args.search_batch)
    out_run_path = os.path.join(out_query_dir, "run.tsv")
    with open(out_run_path, 'w') as output:
        for qid, topk_scores, topk_ids in zip(query_ids, all_topk_scores, all_topk_ids):
            for i, (score, docid) in enumerate(zip(topk_scores, topk_ids)):
                output.write(f"{qid.item()}\tQ0\t{docid.item()}\t{i+1}\t{score.item()}\tSystem\n")

    if qrel_path is None:
        return
    metric_scores = pytrec_evaluate(qrel_path, out_run_path)
    for k in metric_scores.keys():
        if k != "perquery":
            logger.info(metric_scores[k])
    json.dump(metric_scores, open(out_metric_path, 'w'), indent=1)
    

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(eval_args.local_rank) else logging.WARN,
    )
    if is_main_process(eval_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    set_seed(2022)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.similarity_metric is not None:
        config.similarity_metric = model_args.similarity_metric
    if model_args.pooling is not None:
        config.pooling = model_args.pooling
    model = AutoDense.from_pretrained(model_args.model_name_or_path, config=config)

    corpus_embeds, corpus_ids = load_or_encode_corpus(model, tokenizer, model_args, data_args, eval_args)

    query_embeds, query_ids = load_or_encode_query(model, tokenizer, data_args.query_path, data_args.out_query_dir, model_args, data_args, eval_args)
    out_metric_path = os.path.join(data_args.out_query_dir, "metric.json")
    torch.cuda.empty_cache()
    if is_main_process(eval_args.local_rank) and not os.path.exists(out_metric_path):
        os.makedirs(data_args.out_query_dir, exist_ok=True)
        search_and_compute_metrics(corpus_embeds, corpus_ids, query_embeds, query_ids, out_metric_path, data_args.out_query_dir, data_args.qrel_path, eval_args)
    else:
        # only main process can print, so this line is only printed when the metric file exists
        logger.info("Skip search process because metric.json file already exists. ")


if __name__ == "__main__":
    main()