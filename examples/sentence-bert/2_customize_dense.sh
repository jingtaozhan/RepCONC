#!/bin/bash

set -e

echo "Dense retrieval models in the literature use different pooling strategies and similarity metrics. This code will add several fields to the config.json file to denote these specific design choices."

python -m repconc.utils.customize_trained_dense \
    --model_name_or_path sentence-transformers/msmarco-bert-base-dot-v5 \
    --similarity_metric METRIC_IP \
    --pooling mean \
    --output_dir ./data/sbert-marco-passage/dense_encoder