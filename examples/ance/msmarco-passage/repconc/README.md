# Compressing ANCE on MS MARCO Passage Ranking

Here, we assume you have already follow the [previous instructions](..). 

RepCONC has two training stages. In the first stage, it trains the query encoder and passage encoder by minimizing ranking loss and quantization loss. In the second stage, it trains the query encoder and PQ centroids with only ranking loss. 

The following instructions set the number of sub-vectors to $48$. You can also set it to other values.

### RepCONC Training STAGE-1

*Note: In stage-1, we use distributed training for acceleration. We train the models on 8 NVIDIA-V100 GPUs for about 3.5 hours. If you use different numbers of GPUs, it is possible that you need to tune the learning rate accordingly because: [PyTorch averages gradients across all nodes. When a model is trained on M nodes with batch=N, the gradient will be M times smaller when compared to the same model trained on a single node with batch=M*N. You can use a smaller learning rate when training with fewer gpus.](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel). That said, we find RepCONC is not sensitive to different learning rate and you can just use the learning rate in this example as default values.*

```bash
# For example, 
# number of gpus to use is 8 (first argument)

# Generate hard negatives
sh ./msmarco-passage/repconc/6_gen_hardneg.sh 8 48

# Run training
sh ./msmarco-passage/repconc/7_run_conc_train.sh 8 48

# Encodes the corpus
sh ./msmarco-passage/repconc/8_run_conc_eval.sh 8 48

# For example, here is the performance on MS MARCO dev set after stage-1 training.
more ../../data/ance-marco-passage/subvector-48/repconc/encoder_output/dev/metric.json 
```


### RepCONC Training STAGE-2

```bash
# For example, 
# number of gpus to use is 1 (first argument)
# Encoding queries is fast and does not need multi-gpu inference.

# Run training
sh ./msmarco-passage/repconc/9_run_jpq_train.sh 48

# Evaluation
sh ./msmarco-passage/repconc/10_run_jpq_eval.sh 1 48

# For example, here is the performance on MS MARCO dev set.
more ../../data/ance-marco-passage/subvector-48/repconc/query_encoder_results/dev/metric.json 
```