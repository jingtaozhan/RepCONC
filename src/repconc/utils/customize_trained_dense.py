import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from dataclasses import field, dataclass

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed)

from repconc.models.dense import AutoDense

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
            metadata={
                "help": "The dense model needed to be customized."
            }
        )
    similarity_metric: str = field(
            metadata={
                "help": "How the dense model comptues similarity",
                "choices": ["METRIC_IP", "METRIC_COS"]
            }
        )
    pooling: str = field(
            metadata={
                "help": "How the dense model aggregates representations to extract text embeddings", 
                "choices": ["cls", "mean"]
            }
        )
    output_dir: str = field(
            metadata={
                "help": "Where to save the customized model", 
            }
        )

    
def main():
    parser = HfArgumentParser(ModelArguments)
    model_args, = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO 
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    set_seed(2022)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.similarity_metric = model_args.similarity_metric
    config.pooling = model_args.pooling
    model = AutoDense.from_pretrained(model_args.model_name_or_path, config=config)

    tokenizer.save_pretrained(model_args.output_dir)
    model.save_pretrained(model_args.output_dir)


if __name__ == "__main__":
    main()