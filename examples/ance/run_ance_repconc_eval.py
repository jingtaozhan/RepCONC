import os
import faiss
import logging
import numpy as np
import transformers
from transformers import (
    HfArgumentParser, 
    set_seed)
from transformers.trainer_utils import is_main_process

from repconc.models.repconc.evaluate_repconc import (
    ModelArguments, 
    EvalArguments, 
)
from repconc.utils.eval_utils import (
    load_corpus, 
    load_queries,
    DataArguments, 
    load_beir_corpus,
    load_beir_queries,
)
from repconc.models.repconc.evaluate_repconc import (
    ModelArguments, 
    EvalArguments, 
    encode_corpus, 
    encode_query,
)
from repconc.evaluate.run_repconc_eval import (
    search_and_compute_metrics,
)

from modeling_ance import ANCETokenizerFast, ance_repconc_from_pretrained
logger = logging.getLogger(__name__)


def load_or_encode_corpus(model_args: ModelArguments, data_args: DataArguments, eval_args: EvalArguments):
    out_index_path = os.path.join(data_args.out_corpus_dir, "index")
    out_corpus_ids_path = os.path.join(data_args.out_corpus_dir, "corpus_ids.npy")
    if os.path.exists(out_index_path) and os.path.exists(out_corpus_ids_path):
        index = faiss.read_index(out_index_path)
        corpus_ids = np.load(out_corpus_ids_path)
        logger.info("Load pre-computed corpus representations")
    else:
        doc_tokenizer = ANCETokenizerFast.from_pretrained(model_args.doc_encoder_path)
        doc_encoder = ance_repconc_from_pretrained(model_args.doc_encoder_path, False, None, None)
        if data_args.data_format == "msmarco":
            corpus = load_corpus(data_args.corpus_path, doc_tokenizer.sep_token, verbose=is_main_process(eval_args.local_rank))
        elif data_args.data_format == "beir":
            corpus = load_beir_corpus(data_args.corpus_path, doc_tokenizer.sep_token, verbose=is_main_process(eval_args.local_rank))
        else:
            raise NotImplementedError()
        index, corpus_ids = encode_corpus(corpus, doc_encoder, doc_tokenizer, model_args.max_seq_length, eval_args)
        if is_main_process(eval_args.local_rank):
            os.makedirs(data_args.out_corpus_dir, exist_ok=True)
            faiss.write_index(index, out_index_path)
            np.save(out_corpus_ids_path, corpus_ids)
    return index, corpus_ids


def load_or_encode_queries(model_args: ModelArguments, data_args: DataArguments, eval_args: EvalArguments):
    out_query_code_path = os.path.join(data_args.out_query_dir, "codes.npy")
    out_query_ids_path = os.path.join(data_args.out_query_dir, "qids.npy")
    if os.path.exists(out_query_code_path) and os.path.exists(out_query_ids_path):
        query_embeds = np.load(out_query_code_path)
        query_ids = np.load(out_query_ids_path)
        logger.info("Load pre-computed query representations")
    else:
        query_tokenizer = ANCETokenizerFast.from_pretrained(model_args.query_encoder_path)
        query_encoder = ance_repconc_from_pretrained(model_args.query_encoder_path, False, None, None)
        if data_args.data_format == "msmarco":
            queries = load_queries(data_args.query_path)
        elif data_args.data_format == "beir":
            queries = load_beir_queries(data_args.query_path)
        else:
            raise NotImplementedError()
        query_embeds, query_ids = encode_query(queries, query_encoder, query_tokenizer, model_args.max_seq_length, eval_args)
        if is_main_process(eval_args.local_rank):
            os.makedirs(data_args.out_query_dir, exist_ok=True)
            np.save(out_query_code_path, query_embeds)
            np.save(out_query_ids_path, query_ids)
    return query_embeds, query_ids


def replace_pq_centroids(index: faiss.IndexPQ, query_encoder_path: str):
    query_encoder = ance_repconc_from_pretrained(query_encoder_path, False, None, None)
    centroids = query_encoder.centroids.data.detach().cpu().numpy().ravel()
    faiss.copy_array_to_vector(centroids, index.pq.centroids)
    return index


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataArguments
    eval_args: EvalArguments

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
    faiss.omp_set_num_threads(eval_args.threads)

    index, corpus_ids = load_or_encode_corpus(model_args, data_args, eval_args)
    query_embeds, query_ids = load_or_encode_queries(model_args, data_args, eval_args)
    index = replace_pq_centroids(index, model_args.query_encoder_path)

    if is_main_process(eval_args.local_rank):
        search_and_compute_metrics(index, corpus_ids, query_embeds, query_ids, data_args, eval_args)
        

if __name__ == "__main__":
    main()