import os
import torch
import logging
import transformers
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed)
from transformers.trainer_utils import is_main_process

from repconc.evaluate.run_dense_eval import (
    DataArguments, 
    ModelArguments, 
    EvalArguments,
    load_or_encode_query,
    load_or_encode_corpus,
    search_and_compute_metrics
)

from modeling_tct import TCTEncoder, TCTTokenizerFast

logger = logging.getLogger(__name__)

    
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

    ### Only here is modified for ANCE
    tokenizer = TCTTokenizerFast.from_pretrained(model_args.model_name_or_path)
    model = TCTEncoder.from_pretrained(model_args.model_name_or_path)
    ###

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