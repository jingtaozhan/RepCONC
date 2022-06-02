#!/bin/bash

set -e

echo "Evaluate the encoder trained by RepCONC. It will encode passages to discrete representations to facilitate compact index. "

gpus=$1
subvector=$2

root="./data/sbert-marco-passage"
dataset_dir="${root}/dataset"
model_path="${root}/subvector-${subvector}/repconc/encoder"
output_dir="${root}/subvector-${subvector}/repconc/encoder_output"

sh ./examples/evaluate/repconc_eval_marco.sh $gpus $dataset_dir $output_dir $model_path $model_path
