import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
import faiss.contrib.torch_utils
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer_utils import is_main_process
from transformers import TrainingArguments, Trainer

from .modeling_dense import BertDense, RobertaDense
from repconc.utils.eval_utils import TextDataset, get_collator_func

logger = logging.getLogger(__name__)


class DenseEvaluater(Trainer):
    def prediction_step(
        self,
        model: Union[BertDense, RobertaDense],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                logits = model(**inputs).detach().contiguous()
        return (loss, logits, None)


def encode_dense_corpus(corpus: Dict[Union[str, int], str], model: Union[BertDense, RobertaDense], 
        tokenizer, max_seq_length: int, eval_args: TrainingArguments, split_corpus_num=20):
    '''
    Avoid out-of-memory error (dense embeds are memory-inefficient)
    '''
    logger.info("Sorting Corpus by document length (Longest first)...")
    corpus_ids = np.array(sorted(corpus, key=lambda k: len(corpus[k].split()), reverse=True))
    if is_main_process(eval_args.local_rank):
        corpus_embeds = np.empty((len(corpus_ids), model.config.hidden_size), dtype=np.float32)
        write_num = 0
    else:
        corpus_embeds = None
    for doc_ids in tqdm(np.array_split(corpus_ids, split_corpus_num), 
            disable=not is_main_process(eval_args.local_rank), desc="Split corpus encoding"):
        doc_text = [corpus[did] for did in doc_ids]
        doc_dataset = TextDataset(
            doc_text, 
        )
        doc_out = DenseEvaluater(
            model=model,
            args=eval_args,
            data_collator=get_collator_func(tokenizer, max_seq_length),
            tokenizer=tokenizer,
        ).predict(doc_dataset)
        if is_main_process(eval_args.local_rank):
            doc_embeds = doc_out.predictions
            assert len(doc_embeds) == len(doc_text)
            corpus_embeds[write_num:write_num+len(doc_embeds)] = doc_embeds 
            write_num += len(doc_embeds)
    return corpus_embeds, corpus_ids


def encode_dense_query(queries: Dict[Union[str, int], str], model: Union[BertDense, RobertaDense], tokenizer, max_seq_length: int, eval_args: TrainingArguments):
    logger.info("Encoding Queries...")
    query_ids = sorted(list(queries.keys()))
    queries_text = [queries[qid] for qid in query_ids]
    query_dataset = TextDataset(queries_text)
    query_out = DenseEvaluater(
        model=model,
        args=eval_args,
        data_collator=get_collator_func(tokenizer, max_seq_length),
        tokenizer=tokenizer,
    ).predict(query_dataset)
    query_embeds = query_out.predictions
    assert len(query_embeds) == len(queries_text)
    return query_embeds, np.array(query_ids)


def dense_search(query_ids: np.ndarray, query_embeds:np.ndarray, 
    corpus_ids: np.ndarray, index: faiss.IndexFlatIP, topk: int):
    topk_scores, topk_idx = index.search(query_embeds, topk)
    topk_ids = np.vstack([corpus_ids[x] for x in topk_idx])
    assert len(query_ids) == len(topk_scores) == len(topk_ids)
    return topk_scores, topk_ids
  

def batch_dense_search(query_ids: np.ndarray, query_embeds:np.ndarray, 
    corpus_ids: np.ndarray, index: faiss.IndexFlatIP, 
    topk: int, batch_size: int):

    all_topk_scores, all_topk_ids = [], []
    iterations = math.ceil(len(query_ids) / batch_size)
    for query_id_iter, query_embeds_iter in tqdm(zip(
        np.array_split(query_ids, iterations), 
        np.array_split(query_embeds, iterations),
    ), total=iterations, desc="Batch search"):
        topk_scores, topk_ids = dense_search(
            query_id_iter, query_embeds_iter,
            corpus_ids, index, topk
        )
        all_topk_scores.append(topk_scores)
        all_topk_ids.append(topk_ids)
    all_topk_scores = np.concatenate(all_topk_scores, axis=0)
    all_topk_ids = np.concatenate(all_topk_ids, axis=0)
    return all_topk_scores, all_topk_ids


def create_index(corpus_embeds: np.ndarray, single_gpu_id=None):
    index = faiss.IndexFlatIP(corpus_embeds.shape[1])
    if faiss.get_num_gpus() == 1 or single_gpu_id is not None:
        res = faiss.StandardGpuResources()  # use a single GPU
        res.setTempMemory(128*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False
        if single_gpu_id is None:
            single_gpu_id = 0
        index = faiss.index_cpu_to_gpu(res, single_gpu_id, index, co)
    else:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = False
        co.useFloat16 = False
        index = faiss.index_cpu_to_all_gpus(index, co)
    index.add(corpus_embeds)
    return index
