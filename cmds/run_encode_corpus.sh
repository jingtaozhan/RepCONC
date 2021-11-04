set -e 

dataset="passage"
batch_size=128
m=48
max_doc_length=256
doc_encoder_dir="./data/${dataset}/official_doc_encoders/m${m}.marcopass.pq.encoder"
query_encoder_dir="./data/${dataset}/official_query_encoders/m${m}.marcopass.query.encoder"
output_path="./data/${dataset}/run_encode/m${m}-l1.marcopass.ivfpq.index"

python ./run_encode.py \
    --preprocess_dir ./data/${dataset}/preprocess \
    --doc_encoder_dir $doc_encoder_dir \
    --output_path $output_path \
    --batch_size $batch_size \
    --max_doc_length $max_doc_length \
    --query_encoder_dir $query_encoder_dir


dataset="doc"
batch_size=128
m=48
max_doc_length=512
doc_encoder_dir="./data/${dataset}/official_doc_encoders/m${m}.marcodoc.pq.encoder"
query_encoder_dir="./data/${dataset}/official_query_encoders/m${m}.marcodoc.query.encoder"
output_path="./data/${dataset}/run_encode/m${m}-l1.marcodoc.ivfpq.index"

python ./run_encode.py \
    --preprocess_dir ./data/${dataset}/preprocess \
    --doc_encoder_dir $doc_encoder_dir \
    --output_path $output_path \
    --batch_size $batch_size \
    --max_doc_length $max_doc_length \
    --query_encoder_dir $query_encoder_dir