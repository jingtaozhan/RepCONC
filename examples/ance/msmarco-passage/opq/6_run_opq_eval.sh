#!/bin/bash

set -e

echo "Since we have already run OPQ in the warmup script, now we reuse the index and evaluate the efficacy of OPQ."

gpus=$1
subvector=$2

root="../../data/ance-marco-passage"
dataset_dir="${root}/dataset"
doc_encoder_path="${root}/subvector-${subvector}/warmup"
query_encoder_path="${root}/subvector-${subvector}/warmup"
output_dir="${root}/subvector-${subvector}/warmup_output"

mkdir -p $output_dir
out_corpus_dir="$output_dir/corpus"

corpus_path="$dataset_dir/corpus.tsv"

if [ $gpus = 1 ]
then
    distributed_cmd=" "
else
    master_port=$(eval "shuf -i 10000-15000 -n 1")
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $gpus --master_port=$master_port "
fi

for mode in "trec19" "trec20" "dev" 
do
    timestamp=`date "+%m-%d-%H:%M"`
    log_path="${output_dir}/${timestamp}.log"

    python $distributed_cmd \
        run_ance_repconc_eval.py \
        --corpus_path $corpus_path \
        --query_path "$dataset_dir/query.${mode}" \
        --qrel_path "$dataset_dir/qrels.${mode}" \
        --output_dir $output_dir \
        --out_corpus_dir $out_corpus_dir \
        --out_query_dir "$output_dir/${mode}" \
        --doc_encoder_path $doc_encoder_path \
        --query_encoder_path $query_encoder_path \
        --per_device_eval_batch_size 64 \
        --max_seq_length 512 \
        --dataloader_num_workers 1 \
        |& tee -a $log_path   
done

