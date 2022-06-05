# Compressing TCT-ColBERT-v2 on MS MARCO Passage Ranking

This is the instructions about how to transfer [TCT-ColBERT-v2](https://cs.uwaterloo.ca/~jimmylin/publications/Lin_etal_2021_RepL4NLP.pdf) into a memory-efficient dense retrieval model. 

## Retrieval Effectiveness

Here is the effectiveness summarization about different compression methods.

| Models      | PQ Sub-vectors| Index Size  | Compression Ratio | MS MARCO Dev (MRR@10) | TREC 19 DL (NDCG@10) | TREC 20 DL (NDCG@10)
| -----------       | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| TCT-ColBERT-v2-hnp| -  | 26 GB  | 1x  | 0.359 | 0.716 | 0.689 |
| OPQ (Faiss)       | 64 | 541 MB | 48x | 0.321 | 0.652 | 0.636 | 
| JPQ               | 64 | 541 MB | 48x | 0.341 | 0.677 | 0.680 | 
| RepCONC           | 64 | 541 MB | 48x | 0.348 | 0.694 | 0.679 | 
| OPQ (Faiss)       | 48 | 406 MB | 64x | 0.301 | 0.605 | 0.608 | 
| JPQ               | 48 | 406 MB | 64x | 0.329 | 0.681 | 0.663 | 
| RepCONC           | 48 | 406 MB | 64x | 0.338 | 0.697 | 0.676 | 

##  Directory Format

The working directory is scheduled to be:

```
├── data/tct-colbert-v2-marco-passage
│   ├── dataset (will be downloaded)
│   ├── valid_dataset (will be generated)
│   ├── dense_encoder (Path to save TCT-ColBERT-v2 encoder)
│   ├── dense_output (Path to save the output of TCT-ColBERT-v2)
│   ├── subvector-X (X is the number of sub-vectors)
│   │   ├── warmup (OPQ Warmup of TCT-ColBERT-v2)
│   │   ├── warmup_output (OPQ Output of TCT-ColBERT-v2 warmup checkpoint)
│   │   ├── hardneg.json (hard negatives for repconc training)
│   │   ├── jpq (training directory of jpq)
│   │   ├── repconc (training directory of repconc)
```

## Training Instructions

The following is the training instructions about how to reproduce the above results. The first part is the common procedure of all three methods, such as preparing data. The remaining parts are the instructions for OPQ, JPQ, and RepCONC separately. 

### Common Procedure of OPQ/JPQ/RepCONC

```bash
cd examples/tct-colbert

# Prepare MS MARCO dataset
sh ./msmarco-passage/1_prepare_dataset.sh

# Let TCT-ColBERT-v2 encode the corpus. We can know whether we reproduce right. And the corpus encoding can be reused by warmup process or JPQ training process.
# For example, there are 8 gpus available.
sh ./msmarco-passage/3_encode_dense_corpus.sh 8

# Generate validation set. Sample a small corpus for efficient validation during training.
sh ./msmarco-passage/4_gen_valid_set.sh
```

### Index Compression (and Joint Optimization)

The passage representation will be quantized to several sub-vectors. Each sub-vector consumes $1$ byte because the default number of centroids per sub-vector is $256$. 
That is to say, since TCT-ColBERT-v2 encodes passage to a vector of $768$ dimension ($768 \times 4$ bytes), if the number of sub-vectors is $48$ ($48$ bytes), the compression ratio is $768 \times 4/48 = 64$.

Number of sub-vectors is an important hyper-parameter that directly corresponds to the effectiveness-efficiency tradeoff. More sub-vectors result in higher effectiveness but larger memory footprint (and slower retrieval). In this example, we set it to $64$ or $48$. Other value can also be explored as long as 768 is divisible by it.

Warmup the centroids with OPQ. A good warmup accelerates convergence. 
```bash
# You can set number of sub-vectors to 64. (48x compression ratio)
sh ./msmarco-passage/5_opq_warmup.sh 64

# You can also set number of sub-vectors to 48. (64x compression ratio)
sh ./msmarco-passage/5_opq_warmup.sh 48
```

Here are three instructions about reproducing the OPQ, JPQ, and RepCONC results. They are independent from each other. Pick one and follow the instructions.
- [Evaluating OPQ performance](./opq)
- [Joint optimization with JPQ](./jpq)
- [Joint optimization with RepCONC](./repconc)
