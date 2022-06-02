# Compressing coCondenser on MS MARCO Passage Ranking

This is the instructions about how to transfer [coCondenser](https://arxiv.org/pdf/2108.0..pdf) into a memory-efficient dense retrieval model. 

*Note: coCondenser additionally uses the titles of passages during training and inference, which is not included in the benchmark dataset. This has caused some [debate](https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor) since the original benchmark does not contain the title field. If the original corpus is used, the MRR@10 drops from 0.38 to 0.35. Anyway, this example aims to show how to compress the index of coCondenser. We keep its original design to use the modified corpus.*

## Retrieval Effectiveness

Here is the effectiveness summarization about different compression methods.

| Models      | PQ Sub-vectors| Index Size  | Compression Ratio | MS MARCO Dev (MRR@10) | TREC 19 DL (NDCG@10) | TREC 20 DL (NDCG@10)
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| coCondenser | -  | 26 GB  | 1x  | 0.381 | 0.712 | 0.680 |
| OPQ (Faiss) | 64 | 541 MB | 48x | 0.356 | 0.672 | 0.642 | 
| JPQ         | 64 | 541 MB | 48x | 0.370 | 0.693 | 0.675 | 
| RepCONC     | 64 | 541 MB | 48x | 0.373 | 0.705 | 0.692 | 
| OPQ (Faiss) | 48 | 406 MB | 64x | 0.348 | 0.688 | 0.650 | 
| JPQ         | 48 | 406 MB | 64x | 0.364 | 0.678 | 0.669 | 
| RepCONC     | 48 | 406 MB | 64x | 0.368 | 0.679 | 0.676 | 


##  Directory Format

The working directory is scheduled to be:

```
├── data/cocondenser-marco-passage
│   ├── dataset (will be downloaded)
│   ├── valid_dataset (will be generated)
│   ├── dense_encoder (Path to save coCondenser encoder)
│   ├── dense_output (Path to save the output of coCondenser)
│   ├── subvector-X (X is the number of sub-vectors)
│   │   ├── warmup (OPQ Warmup of coCondenser)
│   │   ├── warmup_output (OPQ Output of coCondenser warmup checkpoint)
│   │   ├── hardneg.json (hard negatives for repconc training)
│   │   ├── jpq (training directory of jpq)
│   │   ├── repconc (training directory of repconc)
```

## Training Instructions

The following is the training instructions about how to reproduce the above results. The first part is the common procedure of all three methods, such as preparing data. The remaining parts are the instructions for OPQ, JPQ, and RepCONC separately. 

### Common Procedure of OPQ/JPQ/RepCONC

```bash
# Prepare MS MARCO dataset
sh ./examples/cocondenser/1_prepare_dataset.sh

# coCondenser uses cls pooling and inner product. Add the two fields to the config.json and save the model.
sh ./examples/cocondenser/2_customize_dense.sh

# Let coCondenser encode the corpus. We can know whether we reproduce right. And the corpus encoding can be reused by warmup process or JPQ training process.
# For example, there are 8 gpus available.
sh ./examples/cocondenser/3_encode_dense_corpus.sh 8

# Generate validation set. Sample a small corpus for efficient validation during training.
sh ./examples/cocondenser/4_gen_valid_set.sh

# Use OPQ to compute centroids. It will be used to warmup JPQ and RepCONC.
# For example, number of sub-vectors is set to 64.
# That is, the compression ratio is 768/64=48. 
sh ./examples/cocondenser/5_opq_warmup.sh 64
```

### Index Compression (and Joint Optimization)

The passage representation will be quantized to several sub-vectors. Each sub-vector consumes $1$ byte because the default number of centroids per sub-vector is $256$. 
That is to say, since coCondenser encodes passage to a vector of $768$ dimension ($768 \times 4$ bytes), if the number of sub-vectors is $48$ ($48$ bytes), the compression ratio is $768 \times 4/48 = 64$.

Number of sub-vectors is an important hyper-parameter that directly corresponds to the effectiveness-efficiency tradeoff. More sub-vectors result in higher effectiveness but larger memory footprint (and slower retrieval). In this example, we set it to $64$ or $48$. Other value can also be explored as long as it 768 is divisible by it.

Warmup the centroids with OPQ. A good warmup accelerates convergence. 
```bash
# Here number of sub-vectors is set to 64. (48x compression ratio)
sh ./examples/cocondenser/5_opq_warmup.sh 64
# You can also set number of sub-vectors to 48. (64x compression ratio)
sh ./examples/cocondenser/5_opq_warmup.sh 48
```

Here are three instructions about reproducing the OPQ, JPQ, and RepCONC results. They are independent from each other. Pick one and follow the instructions.
- [Evaluating OPQ performance](./opq)
- [Joint optimization with JPQ](./jpq)
- [Joint optimization with RepCONC](./repconc)
