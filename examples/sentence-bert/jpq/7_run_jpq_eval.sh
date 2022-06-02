#!/bin/bash

set -e

gpus=$1
subvector=$2

root="./data/sbert-marco-passage"
dataset_dir="${root}/dataset"
doc_encoder_path="${root}/subvector-${subvector}/warmup"
query_encoder_path="${root}/subvector-${subvector}/jpq/query_encoder"
output_dir="${root}/subvector-${subvector}/jpq/query_results"

mkdir -p $output_dir

# will load corpus index from this dir
out_corpus_dir="${root}/subvector-${subvector}/warmup_output/corpus"
 
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
    echo log_path: $log_path
    
    python $distributed_cmd \
        -m repconc.evaluate.run_repconc_eval \
        --corpus_path "${dataset_dir}/corpus.tsv" \
        --query_path "${dataset_dir}/query.${mode}" \
        --output_dir $output_dir \
        --out_corpus_dir $out_corpus_dir \
        --out_query_dir "${output_dir}/${mode}" \
        --qrel_path "${dataset_dir}/qrels.${mode}" \
        --doc_encoder_path $doc_encoder_path \
        --query_encoder_path $query_encoder_path \
        --per_device_eval_batch_size 64 \
        --dataloader_num_workers 4 \
        |& tee $log_path   
done
  