set -e 

dataset="passage"
m=48
nlist=5000
nprobe=500
input_index_path="./data/${dataset}/run_encode/m${m}-l1.marcopass.ivfpq.index"
output_index_path="./data/${dataset}/build_ivf/m${m}-l${nlist}.marcopass.ivfpq.index"

# build IVF index
python -m repconc.build_ivf_index.py \
    --input_index_path $input_index_path \
    --output_index_path $output_index_path \
    --nlist $nlist \
    --nprobe $nprobe \
    --threads 32

exit # delete it if you want to evaluate the output index

# run retrieval with IVF index
mode="dev"
query_encoder_path="./data/${dataset}/official_query_encoders/m${m}.marcopass.query.encoder"
output_rank_path="./data/${dataset}/build_ivf/run.${mode}.m${m}.np${nprobe}.tsv"
python -m repconc.run_retrieve.py \
    --preprocess_dir ./data/${dataset}/preprocess \
    --index_path $output_index_path \
    --mode $mode \
    --query_encoder_dir $query_encoder_path \
    --output_path $output_rank_path \
    --batch_size 64 \
    --nprobe $nprobe
    # --gpu_search

# evaluation
python ./msmarco_eval.py ./data/${dataset}/preprocess/$mode-qrel.tsv $output_rank_path
