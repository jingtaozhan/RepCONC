#!/bin/bash

set -e

gpus=$1
dataset_dir=$2
output_dir=$3
doc_encoder_path=$4
query_encoder_path=$5

mkdir -p $output_dir
corpus_path="$dataset_dir/corpus.jsonl"

set -e

timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"

if [ $gpus = 1 ]
then
    distributed_cmd=" "
else
    master_port=$(eval "shuf -i 10000-15000 -n 1")
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $gpus --master_port=$master_port "
fi

python $distributed_cmd \
    -m repconc.evaluate.run_repconc_eval \
    --corpus_path $corpus_path \
    --query_path "$dataset_dir/queries.jsonl" \
    --qrel_path "$dataset_dir/qrels/test.tsv" \
    --output_dir $output_dir \
    --out_corpus_dir $output_dir \
    --out_query_dir $output_dir \
    --doc_encoder_path $doc_encoder_path \
    --query_encoder_path $query_encoder_path \
    --per_device_eval_batch_size 64 \
    --max_seq_length 512 \
    --dataloader_num_workers 1 \
    --data_format beir \
    |& tee -a $log_path   

