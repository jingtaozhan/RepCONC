import os
import faiss
import logging
import numpy as np
from transformers import (
    HfArgumentParser,
    set_seed, )

from repconc.models.repconc import RepCONC

from repconc.train.run_warmup import (
    DataArguments,
    ModelArguments,
    warmup_from_embeds
)

from modeling_tct import TCTEncoder, TCTTokenizerFast

logger = logging.getLogger(__name__)


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
    
    ### Only here is modified
    tokenizer = TCTTokenizerFast.from_pretrained(
        model_args.model_name_or_path, 
    )
    dense_encoder = TCTEncoder.from_pretrained(model_args.model_name_or_path)
    config = dense_encoder.config
    config.MCQ_M = model_args.MCQ_M
    config.MCQ_K = model_args.MCQ_K
    ### 

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
