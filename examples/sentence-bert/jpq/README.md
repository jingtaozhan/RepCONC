# Compressing SBERT on MS MARCO Passage Ranking

Here, we assume you have already follow the [previous instructions](..). 


### JPQ Optimization

JPQ further trains the query encoder and PQ Centroids.

```bash
# JPQ uses 1 gpu for training. Multi-gpu does not provide additional training efficiency gains according to our experiments. 

# number of sub-vectors is set to 64.
sh ./examples/sentence-bert/jpq/6_run_jpq_train.sh 64

# number of sub-vectors is set to 48. 
sh ./examples/sentence-bert/jpq/6_run_jpq_train.sh 48
```


### JPQ Evaluation

JPQ re-encodes the queries. It loads the index of the warmup checkpoint and substitutes the PQ centroids with the newly trained ones.

```bash
# For example, 
# number of gpus to use is 1 (first argument)
# Encoding queries is fast and does not need multi-gpu inference.

# number of sub-vectors is set to 64.
sh ./examples/sentence-bert/jpq/7_run_jpq_eval.sh 1 64
# For example, here is the performance on MS MARCO dev set.
more data/sbert-marco-passage/subvector-64/jpq/query_results/dev/metric.json 

# number of sub-vectors is set to 48. 
sh ./examples/sentence-bert/jpq/7_run_jpq_eval.sh 1 48
# For example, here is the performance on MS MARCO dev set.
more data/sbert-marco-passage/subvector-48/jpq/query_results/dev/metric.json 
```