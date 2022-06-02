# Compressing TAS-Balanced on MS MARCO Passage Ranking

This is the instructions about how to transfer [TAS-Balanced](https://arxiv.org/pdf/2104.0..pdf) into a memory-efficient dense retrieval model. 

## Retrieval Effectiveness

Here is the effectiveness summarization about different compression methods.

| Models      | PQ Sub-vectors| Index Size  | Compression Ratio | MS MARCO Dev (MRR@10) | TREC 19 DL (NDCG@10) | TREC 20 DL (NDCG@10)
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| TAS-B       | -  | 26 GB  | 1x  | 0.344 | 0.721 | 0.685 |
| OPQ (Faiss) | 64 | 541 MB | 48x | 0.312 | 0.676 | 0.633 | 
| JPQ         | 64 | 541 MB | 48x | 0.331 | 0.706 | 0.654 | 
| RepCONC     | 64 | 541 MB | 48x | 0.346 | 0.699 | 0.677 | 
| OPQ (Faiss) | 48 | 406 MB | 64x | 0.298 | 0.645 | 0.630 | 
| JPQ         | 48 | 406 MB | 64x | 0.320 | 0.667 | 0.673 | 
| RepCONC     | 48 | 406 MB | 64x | 0.344 | 0.710 | 0.661 | 


Instructions coming soon.