# Compressing TAS-Balanced on MS MARCO Passage Ranking

Here, we assume you have already follow the [previous instructions](..). Since the warmup process uses OPQ, now you can simply evaluate the effectiveness of the warmup checkpoint.


### OPQ Evaluation

```bash
# For example, 
# number of gpus to use is 1 (first argument)
# Only query encoding is required.

# number of sub-vectors is set to 64.
sh ./examples/tas-balanced/opq/6_run_opq_eval.sh 1 64
# For example, here is the performance on MS MARCO dev set.
more data/tas-b-marco-passage/subvector-64/warmup_output/dev/metric.json 

# number of sub-vectors is set to 48. 
sh ./examples/tas-balanced/opq/6_run_opq_eval.sh 1 48
# For example, here is the performance on MS MARCO dev set.
more data/tas-b-marco-passage/subvector-48/warmup_output/dev/metric.json 
```

