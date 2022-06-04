#!/bin/bash

set -e

echo "Evaluate the customized dense retrieval model, so that: "
echo "- we can know whether we sucessfully reproduce the retrieval results."
echo "- Reused to sample a small validation set. Otherwise, the original corpus is to large to efficiently validate model effectiveness during training."
echo "- For RepCONC, the corpus encodings will be reused for warmup centroids and generating hard negatives."
echo "- For JPQ, the corpus encodings will be reused for warmup centroids and training and inference."

echo "Pass the gpu counts (default: 1)"

gpus=${1:-1}

root="../../data/tct-colbert-v2-marco-passage"
dataset_dir="${root}/dataset"
model_path="castorini/tct_colbert-v2-hnp-msmarco"
output_dir="${root}/dense_output"

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

for mode in "trec19" "trec20" "dev" 
do
    timestamp=`date "+%m-%d-%H:%M"`
    log_path="${output_dir}/${timestamp}.log"
    echo log_path: $log_path

    python $distributed_cmd \
        run_tct_dense_eval.py \
        --corpus_path $corpus_path \
        --query_path "${dataset_dir}/query.${mode}" \
        --output_dir $output_dir \
        --out_corpus_dir $out_corpus_dir \
        --out_query_dir "${output_dir}/${mode}" \
        --qrel_path "${dataset_dir}/qrels.${mode}" \
        --model_name_or_path $model_path \
        --per_device_eval_batch_size 64 \
        --dataloader_num_workers 4 \
        --save_corpus_embed \
        --save_query_embed \
        |& tee $log_path   
done