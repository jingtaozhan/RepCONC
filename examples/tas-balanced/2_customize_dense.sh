#!/bin/bash

set -e

echo "Dense retrieval models in the literature use different pooling strategies and similarity metrics. This code will add several fields to the config.json file to denote these specific design choices."

python -m repconc.utils.customize_trained_dense \
    --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
    --similarity_metric METRIC_IP \
    --pooling cls \
    --output_dir ./data/tas-b-marco-passage/dense_encoder