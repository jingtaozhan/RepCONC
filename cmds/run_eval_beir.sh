set -e

dataset=$1 # "nfcorpus" "scifact" "arguana" "scidocs" "fiqa" "trec-covid" ...
M=${2:-48} # number of sub-vectors
split=${3:-"test"} # test/dev/train
encode_batch_size=${4:-64}

echo dataset: $dataset
echo M: $M
echo split: $split

beir_data_root="./data/beir"
output_dir="./data/passage/beir_output/M${M}/${dataset}"
model_root="./data/passage"
query_encoder="${model_root}/official_query_encoders/m${M}.marcopass.query.encoder"
doc_encoder="${model_root}/official_doc_encoders/m${M}.marcopass.pq.encoder"

python -m jpq.eval_beir \
    --dataset $dataset \
    --beir_data_root $beir_data_root \
    --split $split \
    --encode_batch_size $encode_batch_size \
    --query_encoder $query_encoder \
    --doc_encoder $doc_encoder \
    --output_index_path "${output_dir}/index" \
    --output_ranking_path "${output_dir}/${split}-ranking.pickle"


