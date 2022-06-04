#!/bin/bash

set -e

gpus=$1
subvector=$2

echo "Generate hard negatives for training RepCONC"
echo "Why hard negatives are necessary? Please refer to https://arxiv.org/pdf/2104.08051.pdf "

root="../../data/ance-marco-passage"
dataset_dir="${root}/dataset"
corpus_path="$dataset_dir/corpus.tsv"
query_path="$dataset_dir/query.train"
qrel_path="$dataset_dir/qrels.train"

hardneg_path="${root}/subvector-${subvector}/hardneg.json"

model_path="${root}/subvector-${subvector}/warmup"
output_dir="${root}/subvector-${subvector}/warmup_output"
out_corpus_dir="${output_dir}/corpus"
out_query_dir="${output_dir}/train"

dataloader_num_workers=1

timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"

mkdir -p $output_dir

if [ $gpus = 1 ]
then
    distributed_cmd=" "
else
    master_port=$(eval "shuf -i 10000-15000 -n 1")
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $gpus --master_port=$master_port "
fi

python $distributed_cmd \
    run_ance_repconc_eval.py \
    --corpus_path $corpus_path \
    --query_path $query_path \
    --output_dir $output_dir \
    --out_corpus_dir $out_corpus_dir \
    --out_query_dir "$out_query_dir" \
    --model_name_or_path $model_path \
    --per_device_eval_batch_size 64 \
    --max_seq_length 512 \
    --topk 200 \
    --dataloader_num_workers $dataloader_num_workers \
    |& tee -a $log_path   


python -m repconc.train.run_extract_hardneg \
    --run_path "${out_query_dir}/run.tsv" \
    --qrel_path "$qrel_path" \
    --topk 200 \
    --output_path $hardneg_path

echo "Done"