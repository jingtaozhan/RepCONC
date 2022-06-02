import logging
import os
import sys
import torch
import faiss
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed, )

from repconc.models.dense import AutoDense
from repconc.models.repconc import RepCONC

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    input_corpus_embed_path: str = field(
        metadata = {
            "help": "Path to corpus embeddings"
        }
    )  
    input_corpus_ids_path: str = field(
        metadata = {
            "help": "Path to corpus ids. It will be copied to new place."
        }
    )  
    output_model_dir: str = field(
        metadata = {
            "help": "Where to save the repconc model."
        }
    )
    output_index_path: str = field(
        metadata = {
            "help": "Where to save the index."
        }
    )
    output_corpus_ids_path: str = field(
        metadata = {
            "help": "Where to save the corpus ids."
        }
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata = {
            "help": "Path to dense encoder"
        }
    )
    MCQ_M: int = field(
        metadata = {
            "help": "Number of sub-vectors per text."
        }
    )
    similarity_metric: str = field(
        default = None, 
        metadata = {
            "help": "If None, use the original value.",
            "choices": ["METRIC_CENTROID_COS", "METRIC_IP", "METRIC_COS"]
        }
    )
    pooling: str = field(
        default=None, 
        metadata = {
            "help": "if None, keep the original values",
            "choices": ["cls", "mean"]
        }
    )
    MCQ_K: int = field(
        default = 256,
        metadata = {
            "help": "Number of clusters per sub-vector."
        }
    )


def warmup_from_embeds(
        corpus_embeds: np.ndarray, 
        repconc: RepCONC, 
    ):
    MCQ_M, MCQ_K = repconc.config.MCQ_M, repconc.config.MCQ_K, 
    assert MCQ_K == 256, "256 is a standard setting for K. "

    res = faiss.StandardGpuResources()
    res.setTempMemory(1024*1024*512)
    co = faiss.GpuClonerOptions()
    co.useFloat16 = MCQ_M >= 56

    faiss.omp_set_num_threads(32)
    index = faiss.index_factory(corpus_embeds.shape[1], 
        f"OPQ{MCQ_M},PQ{MCQ_M}x8", faiss.METRIC_INNER_PRODUCT)
    opq = faiss.downcast_VectorTransform(index.chain.at(0))
    pq_regular = faiss.ProductQuantizer(opq.d_out, opq.M, 8)
    assign_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(pq_regular.dsub), co)
    pq_regular.assign_index = assign_index
    opq.pq = pq_regular

    pq = faiss.downcast_index(index.index).pq
    pq.assign_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(pq.dsub), co)

    pq_regular.verbose = True
    pq.verbose = True
    opq.verbose = True
    
    index.train(corpus_embeds)
    index.add(corpus_embeds)

    pq.assign_index = faiss.index_gpu_to_cpu(pq.assign_index)
    pq_regular.assign_index = faiss.index_gpu_to_cpu(pq_regular.assign_index)

    vt = faiss.downcast_VectorTransform(index.chain.at(0))            
    assert isinstance(vt, faiss.LinearTransform)
    rotation = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
    repconc.rotation.copy_(torch.from_numpy(rotation))

    centroids = faiss.vector_to_array(pq.centroids)
    centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
    assert pq.nbits == 8
    repconc.centroids.data.copy_(torch.from_numpy(centroids))

    if repconc.config.similarity_metric == "METRIC_CENTROID_COS":
        repconc.normalize_centrodis()

    return repconc, index


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    model_args: ModelArguments
    data_args: DataArguments

    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed before initializing model.
    set_seed(2022)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
   
    config.MCQ_M = model_args.MCQ_M
    config.MCQ_K = model_args.MCQ_K
    if model_args.similarity_metric is not None:
        config.similarity_metric = model_args.similarity_metric
    if model_args.pooling is not None:
        config.pooling = model_args.pooling

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
    )
    dense_encoder = AutoDense.from_pretrained(model_args.model_name_or_path, config=config)
    repconc = RepCONC(
        config, 
        dense_encoder, 
        use_constraint=False, 
        sk_epsilon=None, 
        sk_iters=None)

    corpus_embeds = np.load(data_args.input_corpus_embed_path)
    repconc, index = warmup_from_embeds(
        corpus_embeds,
        repconc,
    )
    os.makedirs(data_args.output_model_dir, exist_ok=True)
    repconc.save_pretrained(data_args.output_model_dir)
    tokenizer.save_pretrained(data_args.output_model_dir)

    os.makedirs(os.path.dirname(data_args.output_index_path), exist_ok=True)
    os.makedirs(os.path.dirname(data_args.output_corpus_ids_path), exist_ok=True)
    faiss.write_index(faiss.downcast_index(index.index), data_args.output_index_path)
    corpus_ids = np.load(data_args.input_corpus_ids_path)
    np.save(data_args.output_corpus_ids_path, corpus_ids)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
