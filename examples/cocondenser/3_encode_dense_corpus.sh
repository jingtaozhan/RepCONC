#!/bin/bash

set -e

echo "Evaluate the customized dense retrieval model, so that: "
echo "- we can know whether we sucessfully reproduce the retrieval results."
echo "- For RepCONC, the corpus encodings will be reused for warmup centroids and generating hard negatives."
echo "- For JPQ, the corpus encodings will be reused for warmup centroids and training and inference."

echo "Pass the gpu counts (default: 1)"

gpus=${1:-1}

root="./data/cocondenser-marco-passage"
dataset_dir="${root}/dataset"
model_path="${root}/dense_encoder"
output_dir="${root}/dense_output"

sh ./examples/evaluate/dense_eval_marco.sh $gpus $dataset_dir $output_dir $model_path $model_path
