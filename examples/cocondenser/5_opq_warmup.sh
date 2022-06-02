#!/bin/bash

set -e

echo "JPQ and RepCONC both uses Product Quantization technique to compress the index"
echo "To accelerate convergence, we use OPQ to first warmup the centroids."

subvector=$1

echo "If original dimension == 768, then the compression ratio is 768/${subvector}."

root="./data/cocondenser-marco-passage"

input_corpus_embed_path="${root}/dense_output/corpus/corpus_embeds.npy"
input_corpus_ids_path="${root}/dense_output/corpus/corpus_ids.npy"
output_model_dir="${root}/subvector-${subvector}/warmup"
output_index_path="${root}/subvector-${subvector}/warmup_output/corpus/index"
output_corpus_ids_path="${root}/subvector-${subvector}/warmup_output/corpus/corpus_ids.npy"

model_name_or_path="${root}/dense_encoder"
MCQ_M=$subvector

python -m repconc.train.run_warmup \
    --input_corpus_embed_path $input_corpus_embed_path \
    --input_corpus_ids_path $input_corpus_ids_path \
    --output_model_dir $output_model_dir \
    --output_index_path $output_index_path \
    --output_corpus_ids_path $output_corpus_ids_path \
    --model_name_or_path $model_name_or_path \
    --MCQ_M $MCQ_M
    
echo "Done"