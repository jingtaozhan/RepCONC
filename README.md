# RepCONC

This is the official repo for our WSDM'22 paper, [Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval](https://arxiv.org/pdf/2110.05789.pdf) (**Best Paper Award**). 

## Quick Links

  - [Quick Tour](#quick-tour)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Citation](#citation)

## Quick Tour 

In this work, we propose RepCONC, which models quantization process as CONstrained Clustering and end-to-end trains the dual-encoders and the quantization method. Constrained clustering involves a clustering loss and a uniform clustering constraint. The clustering loss requires the embeddings to be around the quantization centroids to support end-to-end optimization, and the constraint forces the embeddings to be uniformly clustered to all centroids to maximize distinguishability. 
The training process and the clustering constraint are visualized as follows:

Training process   |  Constrained Clustering
:-------------------------:|:-------------------------:
<img src="./figures/workflow.png" width="80%">  | <img src="./figures/cons_cluster.png" width="80%"> 

RepCONC achieves huge compression ratios ranging from 64x to 768x. It supports fast embedding search thanks to the adoption of IVF (inverted file system). With these designs, it outperforms a wide range of first-stage retrieval methods in terms of effectiveness, memory efficiency, and time efficiency. 
RepCONC also substantially boosts the second-stage ranking performance, as shown below:
<p align="center">
<img src="./figures/psg_vs_cplx_qps.png" width="50%">  
</p>

## Installation

```bash
git clone https://github.com/jingtaozhan/RepCONC
cd RepCONC
pip install . --use-feature=in-tree-build # built in-place without first copying to a temporary directory.
```


## Examples

We provide examples about how to use RepCONC and JPQ to compress index for different dense retrieval models. These examples are helpful if you want to use RepCONC/JPQ for your dense retrieval models.

* [Compressing index of Sentence BERT on MS MARCO Passage Ranking](./examples/sentence-bert) 
* [Compressing index of coCondenser on MS MARCO Passage Ranking](./examples/cocondenser)
* [Compressing index of TAS-Balanced on MS MARCO Passage Ranking](./examples/tas-balanced)
* [Compressing index of ANCE on MS MARCO Passage Ranking](./examples/ance/msmarco-passage)
* [Compressing index of TCT-ColBERT-v2 on MS MARCO Passage Ranking](./examples/tct-colbert/msmarco-passage)


## Citation
If you find this repo useful, please consider citing our work:
```
@inproceedings{zhan2022learning,
author = {Zhan, Jingtao and Mao, Jiaxin and Liu, Yiqun and Guo, Jiafeng and Zhang, Min and Ma, Shaoping},
title = {Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval},
year = {2022},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3488560.3498443},
doi = {10.1145/3488560.3498443},
booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
pages = {1328–1336},
numpages = {9},
location = {Virtual Event, AZ, USA},
series = {WSDM '22}
}
```

