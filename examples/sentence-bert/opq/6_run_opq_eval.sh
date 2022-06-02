#!/bin/bash

set -e

echo "Since we have already run OPQ in the warmup script, now we reuse the index and evaluate the efficacy of OPQ."

gpus=$1
subvector=$2

root="./data/sbert-marco-passage"
dataset_dir="${root}/dataset"
doc_encoder_path="${root}/subvector-${subvector}/warmup"
query_encoder_path="${root}/subvector-${subvector}/warmup"
output_dir="${root}/subvector-${subvector}/warmup_output"

sh ./examples/evaluate/repconc_eval_marco.sh $gpus $dataset_dir $output_dir $doc_encoder_path $query_encoder_path