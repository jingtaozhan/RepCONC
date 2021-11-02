mode="dev"
echo "Evaluating on msmarco dev set"

batch_size=64
echo "Parameters: Batch size is set to $batch_size"

echo "Begin Evaluation\n"

for dataset in "passage" "doc"; do
    for m in 4 8 12 16 24 32 48; do
        echo "Running inference for msmarco-${dataset} dataset"
        echo "Parameters: Number of subvectors for each ${dataset} is set to ${m}"
        if [ $dataset = "passage" ] 
        then
            index_path="./data/${dataset}/official_pq_index/m${m}-l1.marcopass.ivfpq.index"
            query_encoder_path="./data/${dataset}/official_query_encoders/m${m}.marcopass.query.encoder"
        else
            index_path="./data/${dataset}/official_pq_index/m${m}-l1.marcodoc.ivfpq.index"
            query_encoder_path="./data/${dataset}/official_query_encoders/m${m}.marcodoc.query.encoder"
        fi
        output_path=./data/$dataset/test_results/${mode}/run.${mode}.m${m}.tsv
        python ./run_retrieve.py \
            --preprocess_dir ./data/${dataset}/preprocess \
            --index_path $index_path \
            --mode $mode \
            --query_encoder_dir $query_encoder_path \
            --output_path $output_path \
            --batch_size $batch_size \
            --nprobe 1 \
            --gpu_search
        
        # evaluation
        if [ $dataset = "passage" ] 
        then
            python ./msmarco_eval.py ./data/${dataset}/preprocess/$mode-qrel.tsv $output_path
        else
            python ./msmarco_eval.py ./data/${dataset}/preprocess/$mode-qrel.tsv $output_path 100
        fi
        
        echo "End experiment for m=$m"
        echo "***************************\n"
        
    done
done
