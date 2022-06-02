# Compressing TAS-Balanced on MS MARCO Passage Ranking

Here, we assume you have already follow the [previous instructions](..). 

RepCONC has two training stages. In the first stage, it trains the query encoder and passage encoder by minimizing ranking loss and quantization loss. In the second stage, it trains the query encoder and PQ centroids with only ranking loss. 

The following instructions set the number of sub-vectors to $64$. You can also set it to $48$ or other values.

### RepCONC Training STAGE-1

```bash
# For example, 
# number of gpus to use is 8 (first argument)

# Generate hard negatives
sh ./examples/tas-balanced/repconc/6_gen_hardneg.sh 8 64

# Run training
sh ./examples/tas-balanced/repconc/7_run_conc_train.sh 8 64

# Encodes the corpus
sh ./examples/tas-balanced/repconc/8_run_conc_eval.sh 8 64

# For example, here is the performance on MS MARCO dev set after stage-1 training.
more data/tas-b-marco-passage/subvector-64/repconc/encoder_output/dev/metric.json 
```


### RepCONC Training STAGE-2

```bash
# For example, 
# number of gpus to use is 1 (first argument)
# Encoding queries is fast and does not need multi-gpu inference.

# Run training
sh ./examples/tas-balanced/repconc/9_run_jpq_train.sh 64

# Evaluation
sh ./examples/tas-balanced/repconc/10_run_jpq_eval.sh 64

# For example, here is the performance on MS MARCO dev set.
more data/tas-b-marco-passage/subvector-64/repconc/query_encoder_results/dev/metric.json 
```