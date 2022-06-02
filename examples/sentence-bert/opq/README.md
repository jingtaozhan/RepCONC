# Compressing SBERT on MS MARCO Passage Ranking

Here, we assume you have already follow the [previous instructions](..). Since the warmup process uses OPQ, now you can simply evaluate the effectiveness of the warmup checkpoint.


### OPQ Evaluation

```bash
# For example, 
# number of gpus to use is 8 (first argument)

# number of sub-vectors is set to 64.
sh ./examples/sentence-bert/opq/6_run_opq_eval.sh 8 64

# number of sub-vectors is set to 48. 
sh ./examples/sentence-bert/opq/6_run_opq_eval.sh 8 48
```

