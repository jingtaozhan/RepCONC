#!/bin/bash

set -e

# Distributed training for JPQ is not supported
# because we do not observe significant efficiency gains.

subvector=$1
centroid_learning_rate=${2:-1e-4}
learning_rate=${3:-5e-6}
per_device_train_batch_size=${4:-128}
num_train_epochs=${5:-4}

root="./data/cocondenser-marco-passage"
train_data_root="${root}/dataset"
qrel_path="$train_data_root/qrels.train"
query_path="$train_data_root/query.train"

valid_data_root="${root}/valid_dataset"
valid_qrel_path="$valid_data_root/qrels.dev"
valid_query_path="$valid_data_root/query.dev"

model_name_or_path="${root}/subvector-${subvector}/warmup"
index_input_dir="${root}/subvector-${subvector}/warmup_output/corpus"
train_name="query_encoder_l${learning_rate}-cl${centroid_learning_rate}-b${per_device_train_batch_size}"
output_dir="${root}/subvector-${subvector}/jpq/${train_name}"

echo output_dir: $output_dir
mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

python -m repconc.train.run_train_jpq \
  --qrel_path $qrel_path \
  --query_path $query_path \
  --valid_qrel_path $valid_qrel_path \
  --valid_query_path $valid_query_path \
  --output_dir $output_dir \
  --model_name_or_path $model_name_or_path \
  --index_input_dir $index_input_dir \
  --logging_steps 100 \
  --max_query_len 16 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --temperature 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate $learning_rate \
  --centroid_learning_rate $centroid_learning_rate \
  --num_train_epochs $num_train_epochs \
  --dataloader_drop_last \
  --overwrite_output_dir \
  --dataloader_num_workers 0 \
  --weight_decay 0 \
  --lr_scheduler_type "constant" \
  --metric_for_best_model MRR@10 \
  --save_total_limit 2 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --eval_steps 1000 \
  --load_best_model_at_end \
  --optim adamw_torch \
  |& tee $log_path   

cd "${root}/subvector-${subvector}/jpq"
ln -sf "$train_name" "query_encoder"