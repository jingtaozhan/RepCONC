#!/bin/bash

set -e

echo "Run RepCONC training stage #1 (https://arxiv.org/pdf/2110.05789.pdf) "
echo "Hint for tuning hyper-parameters"
echo "- If mse_loss_weight is too large, MSE loss dominates and harms ranking effectiveness. If too small, PQ centroids and dense representations are loosely bounded and performance degenerates."
echo "- Learning rate for centroids should be much larger than that for dense encoder. (Roughly 10x - 30x for bert-base encoder)."
echo "- We sample negative_per_query negatives for each query. It affects training efficiency and effectiveness. If it is to small, training is fast but effectiveness is not very good. 11 is relatively good for MS MARCO Passage. You can also use 7 or 3 if you have limited machines."
echo "- full_batch_size is the number of queries per batch. We use relatively large one (4096). We also observe that 2048 also works."
echo "- cache_chunk_size is the batch size per forward. If you have limited cuda memory, you may not need to use smaller batch size but just need to set a smaller cache_chunk_size. "

gpus=$1
subvector=$2
mse_loss_weight=${3:-1e-4}
centroid_learning_rate=${4:-5e-4}
learning_rate=${5:-2e-5}
negative_per_query=${6:-11}
full_batch_size=${7:-4096}
num_train_epochs=${8:-4}

per_device_train_batch_size=$(( $full_batch_size / $gpus ))

echo negative_per_query: $negative_per_query

if [ $((gpus * per_device_train_batch_size)) != $full_batch_size ]
then
    echo "Per device batch size not a integer"
    exit
fi

echo full_batch_size: $full_batch_size

root="./data/cocondenser-marco-passage"
train_data_root="${root}/dataset"
qrel_path="$train_data_root/qrels.train"
query_path="$train_data_root/query.train"
corpus_path="$train_data_root/corpus.tsv"

valid_data_root="${root}/valid_dataset"
valid_qrel_path="$valid_data_root/qrels.dev"
valid_query_path="$valid_data_root/query.dev"
valid_corpus_path="$valid_data_root/corpus.tsv"

model_name_or_path="${root}/subvector-${subvector}/warmup"
hardneg_path="${root}/subvector-${subvector}/hardneg.json"
train_name="encoder_l${learning_rate}-cl${centroid_learning_rate}-b${full_batch_size}-m${mse_loss_weight}-n${negative_per_query}"
output_dir="${root}/subvector-${subvector}/repconc/${train_name}"

echo output_dir: $output_dir
mkdir -p $output_dir
timestamp=`date "+%m-%d-%H:%M"`
log_path="${output_dir}/${timestamp}.log"
echo log_path: $log_path

if [ $gpus = 1 ]
then
    distributed_cmd=" "
else
    master_port=$(eval "shuf -i 10000-15000 -n 1")
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $gpus --master_port=$master_port "
fi

python $distributed_cmd \
  -m repconc.train.run_train_conc \
  --qrel_path $qrel_path \
  --query_path $query_path \
  --corpus_path $corpus_path \
  --valid_qrel_path $valid_qrel_path \
  --valid_query_path $valid_query_path \
  --valid_corpus_path $valid_corpus_path \
  --output_dir $output_dir \
  --model_name_or_path $model_name_or_path \
  --logging_steps 5 \
  --max_query_len 16 \
  --max_doc_len 128 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size 32 \
  --temperature 1 \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --negative_per_query $negative_per_query \
  --dynamic_topk_hard_negative $negative_per_query \
  --learning_rate $learning_rate \
  --centroid_learning_rate $centroid_learning_rate \
  --num_train_epochs $num_train_epochs \
  --dataloader_drop_last \
  --overwrite_output_dir \
  --dataloader_num_workers 0 \
  --weight_decay 0 \
  --lr_scheduler_type "constant" \
  --cache_chunk_size 64 \
  --mse_loss_weight $mse_loss_weight \
  --negative $hardneg_path \
  --sk_epsilon 0.003 \
  --sk_iters 100 \
  --metric_for_best_model MRR@10 \
  --save_total_limit 2 \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --eval_steps 40 \
  --save_steps 40 \
  --load_best_model_at_end \
  --optim adamw_torch \
  |& tee $log_path   

cd "${root}/subvector-${subvector}/repconc"
if [ -f "encoder" ]; then
    echo "encoder symbolic already exists."
else 
    ln -s "$train_name" "encoder"
fi