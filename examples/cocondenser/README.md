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


Instructions coming soon.