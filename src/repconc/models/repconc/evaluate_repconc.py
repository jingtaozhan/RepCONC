import os
import math
import torch
import faiss
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import is_main_process

from .modeling_repconc import RepCONC, QuantizeOutput
from repconc.utils.eval_utils import TextDataset, get_collator_func

logger = logging.getLogger(__name__)
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)
    doc_encoder_path: str = field(default=None)
    query_encoder_path: str = field(default=None)
    max_seq_length: int = field(default=None)

    def __post_init__(self):
        if self.model_name_or_path is not None:
            assert self.doc_encoder_path is None and self.query_encoder_path is None
            self.doc_encoder_path, self.query_encoder_path = self.model_name_or_path, self.model_name_or_path
 

@dataclass
class EvalArguments(TrainingArguments):
    topk : int = field(default=1000)
    threads: int = field(default=1)
    search_batch: int = field(default=1200)
    cpu_search: bool = field(default=False)

    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

class RepCONCEvaluater(Trainer):
    def __init__(self, output_format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert output_format in ["code", "continuous_embedding"]
        self.output_format = output_format

    def prediction_step(
        self,
        model: RepCONC,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert prediction_loss_only == False
        assert ignore_keys is None
        inputs = self._prepare_inputs(inputs)
        text_ids = inputs['text_ids']
        del inputs['text_ids']

        with torch.no_grad():
            loss = None
            with self.autocast_smart_context_manager():
                quant_out: QuantizeOutput = model(return_code=True,**inputs)
            if self.output_format == "code":
                logits = quant_out.discrete_codes.detach().type(torch.uint8)
            elif self.output_format == "continuous_embedding":
                logits = quant_out.continuous_embeds.detach()
            else:
                raise NotImplementedError()
        
        return (loss, logits, text_ids)


def initialize_index(model:RepCONC):
    D, M = model.config.hidden_size, model.config.MCQ_M
    assert model.config.MCQ_K == 256
    index = faiss.IndexPQ(D, M, 8, faiss.METRIC_INNER_PRODUCT)
    index.is_trained = True
    # set centroid values 
    centroids = model.centroids.data.detach().cpu().numpy()
    faiss.copy_array_to_vector(centroids.ravel(), index.pq.centroids)
    return index


def add_docs(index: faiss.IndexPQ, new_codes: np.ndarray):
    M = index.pq.code_size
    ntotal = index.ntotal
    new_n = len(new_codes)
    assert new_codes.shape == (new_n, M)
    index.codes.resize((ntotal+new_n)*M)
    codes = faiss.vector_to_array(index.codes)
    codes.reshape(-1, M)[-new_n:] = new_codes
    faiss.copy_array_to_vector(codes, index.codes)
    index.ntotal += new_n


def from_pq_to_ivfpq(indexpq: faiss.IndexPQ) -> faiss.IndexIVFPQ:
    coarse_quantizer = faiss.IndexFlatL2(indexpq.pq.d)
    coarse_quantizer.is_trained = True
    coarse_quantizer.add(np.zeros((1, indexpq.pq.d), dtype=np.float32))
    ivfpq = faiss.IndexIVFPQ(
            coarse_quantizer,
            indexpq.pq.d, 1, indexpq.pq.M, indexpq.pq.nbits, 
            indexpq.metric_type)
    ivfpq.pq = indexpq.pq
    ivfpq.is_trained = True
    ivfpq.precompute_table()
    codes = faiss.vector_to_array(indexpq.codes).reshape(-1, indexpq.pq.M)
    codes = np.ascontiguousarray(codes)
    ids = np.ascontiguousarray(np.arange(len(codes), dtype=np.int64))
    ivfpq.invlists.add_entries(0, len(codes), 
        faiss.swig_ptr(ids.ravel()), 
        faiss.swig_ptr(codes.ravel()))
    return ivfpq


def load_index_to_gpu(index: faiss.IndexIVFPQ, single_gpu_id = None):
    if faiss.get_num_gpus() == 1 or single_gpu_id is not None:
        res = faiss.StandardGpuResources()
        res.setTempMemory(128*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = index.pq.M >= 56
        if single_gpu_id is None:
            single_gpu_id = 0
        index = faiss.index_cpu_to_gpu(res, single_gpu_id, index, co)
    else:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = False
        co.useFloat16 = index.pq.M >= 56
        index = faiss.index_cpu_to_all_gpus(index, co)
    return index


def encode_corpus(corpus: Dict[Union[str, int], str], model: RepCONC, tokenizer, max_seq_length: int, eval_args: EvalArguments):
    logger.info("Sorting Corpus by document length (Longest first)...")
    corpus_ids = np.array(sorted(corpus, key=lambda k: len(corpus[k]), reverse=True))
    corpus_text = [corpus[cid] for cid in corpus_ids]
    corpus_dataset = TextDataset(
        corpus_text, 
        text_ids=list(range(len(corpus_text)))
    )
    logger.info("begin evaluate")
    corpus_out = RepCONCEvaluater(
        output_format="code",
        model=model,
        args = eval_args,
        data_collator=get_collator_func(tokenizer, max_seq_length),
        tokenizer=tokenizer,
    ).predict(corpus_dataset)
    corpus_codes = corpus_out.predictions
    assert len(corpus_codes) == len(corpus)

    index = initialize_index(model)
    add_docs(index, corpus_codes)
    return index, corpus_ids


def encode_query(queries: Dict[int, str], model: RepCONC, tokenizer, 
    max_seq_length: int, eval_args: EvalArguments):
    logger.info("Encoding Queries...")
    query_ids = sorted(list(queries.keys()))
    queries_text = [queries[qid] for qid in query_ids]
    query_dataset = TextDataset(queries_text, list(range(len(queries))))
    query_out = RepCONCEvaluater(
        output_format="continuous_embedding",
        model=model,
        args=eval_args,
        data_collator=get_collator_func(tokenizer, max_seq_length),
        tokenizer=tokenizer,
    ).predict(query_dataset)
    query_embeds = query_out.predictions
    assert len(query_embeds) == len(queries_text), f"{query_embeds.shape}, {len(queries_text)}"
    return query_embeds, np.array(query_ids)


def search(query_ids: np.ndarray, query_embeds:np.ndarray, 
    corpus_ids: np.ndarray, index: faiss.IndexPQ, topk: int):
    topk_scores, topk_idx = index.search(query_embeds, topk)
    topk_ids = np.vstack([corpus_ids[x] for x in topk_idx])
    assert len(query_ids) == len(topk_scores) == len(topk_ids)
    return topk_scores, topk_ids
  

def batch_search(query_ids: np.ndarray, query_embeds:np.ndarray, 
    corpus_ids: np.ndarray, index: faiss.IndexPQ, 
    topk: int, batch_size: int):

    all_topk_scores, all_topk_ids = [], []
    iterations = math.ceil(len(query_ids) / batch_size)
    for query_id_iter, query_embeds_iter in tqdm(zip(
        np.array_split(query_ids, iterations), 
        np.array_split(query_embeds, iterations),
    ), total=iterations, desc="Batch search"):
        topk_scores, topk_ids = search(
            query_id_iter, query_embeds_iter,
            corpus_ids, index, topk
        )
        all_topk_scores.append(topk_scores)
        all_topk_ids.append(topk_ids)
    all_topk_scores = np.concatenate(all_topk_scores, axis=0)
    all_topk_ids = np.concatenate(all_topk_ids, axis=0)
    return all_topk_scores, all_topk_ids
    