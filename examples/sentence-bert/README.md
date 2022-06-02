# Compressing SBERT on MS MARCO Passage Ranking

This is the instructions about how to transfer [SBERT](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5/tree/main) into a memory-efficient dense retrieval model. 

## Retrieval Effectiveness

Here is the effectiveness summarization about different compression methods.

| Models      | PQ Sub-vectors| Index Size  | Compression Ratio | MS MARCO Dev (MRR@10) | TREC 19 DL (NDCG@10) | TREC 20 DL (NDCG@10)
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| SBERT       | -  | 26 GB  | 1x  | 0.381 | 0.707 | 0.726 |
| OPQ (Faiss) | 64 | 541 MB | 48x | 0.345 | 0.693 | 0.686 | 
| JPQ         | 64 | 541 MB | 48x | 0.355 | 0.705 | 0.707 | 
| RepCONC     | 64 | 541 MB | 48x | 0.370 | 0.705 | 0.705 | 
| OPQ (Faiss) | 48 | 406 MB | 64x | 0.335 | 0.690 | 0.664 | 
| JPQ         | 48 | 406 MB | 64x | 0.351 | 0.723 | 0.696 | 
| RepCONC     | 48 | 406 MB | 64x | 0.363 | 0.709 | 0.702 | 

##  Directory Format

The working directory is scheduled to be:

```
├── data/sbert-marco-passage
│   ├── dataset (will be downloaded)
│   ├── valid_dataset (will be generated)
│   ├── dense_encoder (Path to save SBERT encoder)
│   ├── dense_output (Path to save the output of SBERT)
│   ├── subvector-X (X is the number of sub-vectors)
│   │   ├── warmup (OPQ Warmup of SBERT)
│   │   ├── warmup_output (OPQ Output of SBERT warmup checkpoint)
│   │   ├── hardneg.json (hard negatives for repconc training)
│   │   ├── jpq (training directory of jpq)
│   │   ├── repconc (training directory of repconc)
```

## Training Instructions

The following is the training instructions about how to reproduce the above results. The first part is the common procedure of all three methods, such as preparing data. The remaining parts are the instructions for OPQ, JPQ, and RepCONC separately. 

### Common Procedure of OPQ/JPQ/RepCONC

```bash
# Prepare MS MARCO dataset
sh ./examples/sentence-bert/1_prepare_dataset.sh

# SBERT uses mean pooling and inner product. Add the two fields to the config.json and save the model.
sh ./examples/sentence-bert/2_customize_dense.sh

# Let SBERT encode the corpus. We can know whether we reproduce right. And the corpus encoding can be reused by warmup process or JPQ training process.
# For example, there are 8 gpus available.
sh ./examples/sentence-bert/3_encode_dense_corpus.sh 8

# Generate validation set. Sample a small corpus for efficient validation during training.
sh ./examples/sentence-bert/4_gen_valid_set.sh

# Use OPQ to compute centroids. It will be used to warmup JPQ and RepCONC.
# For example, number of sub-vectors is set to 64.
# That is, the compression ratio is 768/64=48. 
sh ./examples/sentence-bert/5_opq_warmup.sh 64
```

### Index Compression (and Joint Optimization)

The passage representation will be quantized to several sub-vectors. Each sub-vector consumes $1$ byte because the default number of centroids per sub-vector is $256$. 
That is to say, since SBERT encodes passage to a vector of $768$ dimension ($768 \times 4$ bytes), if the number of sub-vectors is $48$ ($48$ bytes), the compression ratio is $768 \times 4/48 = 64$.

Number of sub-vectors is an important hyper-parameter that directly corresponds to the effectiveness-efficiency tradeoff. More sub-vectors result in higher effectiveness but larger memory footprint (and slower retrieval). In this example, we set it to $64$ or $48$. Other value can also be explored as long as 768 is divisible by it.

Warmup the centroids with OPQ. A good warmup accelerates convergence. 
```bash
# Here number of sub-vectors is set to 64. (48x compression ratio)
sh ./examples/sentence-bert/5_opq_warmup.sh 64
# You can also set number of sub-vectors to 48. (64x compression ratio)
sh ./examples/sentence-bert/5_opq_warmup.sh 48
```

Here are three instructions about reproducing the OPQ, JPQ, and RepCONC results. They are independent from each other. Pick one and follow the instructions.
- [Evaluating OPQ performance](./opq)
- [Joint optimization with JPQ](./jpq)
- [Joint optimization with RepCONC](./repconc)
