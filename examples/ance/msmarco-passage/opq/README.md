# Compressing ANCE on MS MARCO Passage Ranking

Here, we assume you have already follow the [previous instructions](..). Since the warmup process uses OPQ, now you can simply evaluate the effectiveness of the warmup checkpoint.


### OPQ Evaluation

```bash
# For example, 
# number of gpus to use is 1 (first argument)
# Only query encoding is required.

# number of sub-vectors is set to 48.
sh ./msmarco-passage/opq/6_run_opq_eval.sh 1 48
# For example, here is the performance on MS MARCO dev set.
more ../../data/ance-marco-passage/subvector-48/warmup_output/dev/metric.json 
```

