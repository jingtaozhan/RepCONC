#!/bin/bash

set -e

echo "We will sample a small corpus to efficiently validate model effectiveness during training"
echo "Top 100 passages for each dev query are merged."

root="../../data/ance-marco-passage"
input_dataset_dir="${root}/dataset"
output_dataset_dir="${root}/valid_dataset"
run_path="${root}/dense_output/dev/run.tsv"

python -m repconc.train.run_gen_valid_set \
    --input_corpus_path $input_dataset_dir/corpus.tsv \
    --input_query_path $input_dataset_dir/query.dev \
    --input_qrel_path $input_dataset_dir/qrels.dev \
    --input_run_path $run_path \
    --topk 100 \
    --output_corpus_path $output_dataset_dir/corpus.tsv \
    --output_query_path $output_dataset_dir/query.dev \
    --output_qrel_path $output_dataset_dir/qrels.dev 
    
echo "Done"