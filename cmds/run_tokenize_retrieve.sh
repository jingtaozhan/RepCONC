set -e 

batch_size=64
echo "Parameters: Batch size is set to $batch_size"

echo "Begin Evaluation\n"

for year in "2020" "2019"  ; do
    echo "Evaluating on msmarco TREC${year} set\n"
    for dataset in "doc" "passage"; do
        m=48
        echo "Running inference for msmarco-${dataset} TREC${year} dataset"
        echo "Parameters: Number of subvectors for each ${dataset} is set to ${m}"
        if [ $dataset = "passage" ] 
        then
            index_path="./data/${dataset}/official_pq_index/m${m}-l1.marcopass.ivfpq.index"
            query_encoder_path="./data/${dataset}/official_query_encoders/m${m}.marcopass.query.encoder"
        else
            index_path="./data/${dataset}/official_pq_index/m${m}-l1.marcodoc.ivfpq.index"
            query_encoder_path="./data/${dataset}/official_query_encoders/m${m}.marcodoc.query.encoder"
        fi
        trec_data_dir="./data/${dataset}/trec20-test"
        query_file_path="${trec_data_dir}/msmarco-test${year}-queries.tsv "
        pid2offset_path="./data/${dataset}/preprocess/pid2offset.pickle"
        output_path=./data/$dataset/tokenize_retrieve/official.run.trec${year}.m${m}.rank

        python -m repconc.tokenize_retrieve \
            --query_file_path $query_file_path \
            --index_path $index_path \
            --query_encoder_dir $query_encoder_path \
            --output_path $output_path \
            --output_format "trec" \
            --pid2offset_path $pid2offset_path \
            --dataset $dataset \
            --batch_size $batch_size \
            --gpu_search

        echo "Use official qrels files to compute metrics"
        
        # Evaluate TREC Test
        if [ $dataset = "passage" ] 
        then
            echo "After we completed this research, we noticed that TREC DL Track noted 'passage evaluation should add \"-l 2\" option when using trec_eval (https://trec.nist.gov/data/deep2020.html). However, we do not add this option when reporting the results in the paper. Here, we add this option. The recall values are higher than the reported values. '"
            ./data/trec_eval-9.0.7/trec_eval -c -mndcg_cut.10 -mrecall.100 -l 2 $trec_data_dir/${year}qrels-pass.txt $output_path
        else
            ./data/trec_eval-9.0.7/trec_eval -c -mndcg_cut.10 -mrecall.100 $trec_data_dir/${year}qrels-docs.txt $output_path
        fi
        
        echo "End experiment for m=$m"
        echo "***************************\n"
    done
done
