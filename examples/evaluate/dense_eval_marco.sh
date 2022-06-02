#!/bin/bash

set -e

gpus=$1
dataset_dir=$2 
output_dir=$3
model_name_or_path=$4

corpus_path="${dataset_dir}/corpus.tsv"

mkdir -p $output_dir
out_corpus_dir="${output_dir}/corpus"

if [ $gpus = 1 ]
then
    distributed_cmd=" "
else
    master_port=$(eval "shuf -i 10000-15000 -n 1")
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $gpus --master_port=$master_port "
fi

timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

for mode in "trec19" "trec20" "dev" ; do
    python $distributed_cmd \
        -m repconc.evaluate.run_dense_eval \
        --corpus_path $corpus_path \
        --query_path "${dataset_dir}/query.${mode}" \
        --output_dir $output_dir \
        --out_corpus_dir $out_corpus_dir \
        --out_query_dir "${output_dir}/${mode}" \
        --qrel_path "${dataset_dir}/qrels.${mode}" \
        --model_name_or_path $model_name_or_path \
        --per_device_eval_batch_size 64 \
        --dataloader_num_workers 4 \
        --save_corpus_embed \
        --save_query_embed \
        |& tee $log_path   
done
  