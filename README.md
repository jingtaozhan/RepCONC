# RepCONC

This is the official repo for our WSDM'22 paper, [Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval](https://arxiv.org/pdf/2110.05789.pdf). 

**************************** **Updates** ****************************
* 11/3: We released code for [encoding corpus](#encode-corpus) and [IVF acceleration](#build-ivf-index).
* 11/2: We released our [model checkpoints](#models-and-indexes) and [retrieval code](#retrieval).
* 10/13: Our paper has been accepted by WSDM! Please check out the [preprint paper](https://arxiv.org/pdf/2110.05789.pdf).

## Quick Links

  - [Quick Tour](#quick-tour)
  - [Requirements](#requirements)
  - [Preprocess Data](#preprocess)
  - [Evaluate Open-sourced Checkpoints](#evaluate-open-sourced-checkpoints)
    - [Model Checkpoints](#models-and-indexes)
    - [Encode Corpus](#encode-corpus)
    - [Build IVF Index](#build-ivf-index)
    - [Run Retrieval](#retrieval)
  - [Citation](#citation)
  - [Related Work](#related-work)

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

## Requirements

This repo needs the following libraries (Python 3.x):
```
torch == 1.9.0
transformers == 4.3.3
faiss-gpu == 1.7.1
boto3
```

## Preprocess
Here are the commands to for preprocessing/tokenization. 

If you do not have MS MARCO dataset, run the following command:
```
bash download_data.sh
```
Preprocessing (tokenizing) only requires a simple command:
```
python preprocess.py --data_type doc; python preprocess.py --data_type passage
```
It will create two directories, i.e., `./data/passage/preprocess` and `./data/doc/preprocess`. We map the original qid/pid to new ids, the row numbers in the file. The mapping is saved to `pid2offset.pickle` and `qid2offset.pickle`, and new qrel files (`train/dev/test-qrel.tsv`) are generated. The passages and queries are tokenized and saved in the numpy memmap file. 

Note: RepCONC, as long as most of our [prior models](#related-work), utilizes Transformers 2.x version to tokenize text. However, when Transformers library updates to 3.x or 4.x versions, the RobertaTokenizer behaves differently. 
To support REPRODUCIBILITY, we copy the RobertaTokenizer source codes from 2.x version to [star_tokenizer.py](star_tokenizer.py). During preprocessing, we use `from star_tokenizer import RobertaTokenizer` instead of `from transformers import RobertaTokenizer`. It is also **necessary** for you to do this if you use our trained RepCONC models on other datasets. 

## Evaluate Open-sourced Checkpoints
### Models and Indexes

You can download the query encoders and indexes from our [dropbox link](https://www.dropbox.com/sh/4xqve2ixf0nrva3/AABm6Z1Ase2AC0ZpJhSrVJzGa?dl=0). After opening this link in your browser, you can see two folder, `doc` and `passage`. They correspond to MSMARCO passage ranking and document ranking. There are also four folders in either of them:
* Encoders: 
  * `official_doc_encoders`:  The unified query/document encoder output from the first-stage training. RepCONC adopts [STAR](https://arxiv.org/pdf/2104.08051.pdf) negative sampling method in this stage. 
  * `official_query_encoders`: The query encoder output from the second-stage training. RepCONC adopts [ADORE](https://arxiv.org/pdf/2104.08051.pdf) negative sampling method in this stage.
* Indexes (Note, the `pid` in the index is actually the row number of a passage in the `collection.tsv` file instead of the official pid provided by MS MARCO.): 
  * `official_pq_index`: PQ indexes.   
  * `official_ivf_index`: IVF accelerated PQ indexes. The number of inverted lists is set to 5000. 

Different query encoders and indexes correspond to different compression ratios. For example, the query encoder named `m32.marcopass.query.encoder.tar.gz` means 32 bytes per doc, i.e., `768*4/32=96x` compression ratio.

We provide several scripts to help you download these data.
```bash
sh ./cmds/download_query_encoder.sh
sh ./cmds/download_doc_encoder.sh
sh ./cmds/download_index.sh
```
### Encode Corpus

In this section, we provide commands about how to encode the corpus to compact indexes with our provided encoders. 
Note, you can skip this section and download the open-sourced indexes by running (only once): 
```bash
sh ./cmds/download_index.sh
```

To encode the corpus:

First, you need to [preprocess the dataset](#preprocess).

Second, please download the [open-sourced query and document encoders](#models-and-indexes). Here are two scripts to help you download them.
```bash
sh ./cmds/download_query_encoder.sh
sh ./cmds/download_doc_encoder.sh
```

Finally, run [run_encode.py](run_encode.py) to encode corpus. You can refer to the example commands in [cmds/run_encode_corpus.sh](cmds/run_encode_corpus.sh).
Arguments for [run_encode.py](run_encode.py) script are as follows,
* `--preprocess_dir`: preprocess dir
    * `./data/passage/preprocess`: default dir for passage preprocessing.
    * `./data/doc/preprocess`: default dir for document preprocessing.
* `--doc_encoder_dir`: The unified query/document encoder trained in the first-stage training process. The script uses it to generate Index Assignments for all passages/documents.  
* `--query_encoder_dir`:  The query encoder trained in the second-stage training process. The script uses it to set the centroid embeddings. If it is not provided, the centroid embeddings are set according the `--doc_encoder_dir` model. 
* `--output_path`:  Output index path.
* `--max_doc_length`: Max passage/document length. Set it to 256 for passage and 512 for document, respectively.
* `--batch_size`: Encoding batch size.

### Build IVF Index
In this section, we provide commands about how to use IVF to accelerate search. The IVF index is built upon the PQ index output by [run_encode.py](run_encode.py).
Note, you can skip this section and download the open-sourced indexes by running (only once): 
```bash
sh ./cmds/download_index.sh
```

We provide an example command in [run_build_ivf_index.sh](cmds/run_build_ivf_index.sh). It builds an IVFPQ index for MS MARCO Passage Ranking task. It calls [build_ivf_index.py](build_ivf_index.py). Arguments for this script are as follows,
* `--input_index_path`: The path for index output by [run_encode.py](run_encode.py).
* `--output_index_path`: The output index path.
* `--nlist`: The number of inverted lists. Large nlist improves accuracy at the cost of computation overhead.
* `--nprobe`: The number of searched lists during online retrieval. The ideal IVF speedup ratio equals to nlist/nprobe.
* `--threads`: The number of threads.
* `--by_residual`: Whether to combine IVF and PQ embeds. Default: False.


### Retrieval

In this section, we provide commands about how to reproduce the retrieval results with our open-sourced indexes and query encoders. 
Run the following command to evaluate the retrieval results on dev set.
```bash
sh ./cmds/run_retrieval.sh
```
or this command to evaluate the ivf accelerated search results:
```bash
sh ./cmds/run_ivf_accelerate_retrieval.sh
```
Both of them will call [run_retrieve.py](run_retrieve.py) to retrieve candidates.
Arguments for this evaluation script are as follows,
* `--preprocess_dir`: preprocess dir
    * `./data/passage/preprocess`: default dir for passage preprocessing.
    * `./data/doc/preprocess`: default dir for document preprocessing.
* `--mode`: Evaluation mode
    * `dev` run retrieval for msmarco development queries.
    * `test`: run retrieval for TREC 2019 DL Track queries.
    * `lead`: run retrieval for leaderboard queries.
* `--index_path`: Index path (can be either PQ index or IVF accelerated index)
* `--query_encoder_dir`:  Query encoder dir, which involves `config.json` and `pytorch_model.bin`.
* `--output_path`:  Output ranking file path, formatted following msmarco guideline (qid\tpid\trank).
* `--max_query_length`: Max query length, default: 32.
* `--batch_size`: Encoding and retrieval batch size at each iteration.
* `--topk`: Retrieve topk passages/documents.
* `--gpu_search`: Whether to use gpu for embedding search.
* `--nprobe`: How many inverted lists to probe. This value shoule lie in [1, number of inverted lists].
 
## Citation
If you find this repo useful, please consider citing our work:
```
@article{zhan2021learning,
  title={Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval},
  author={Zhan, Jingtao and Mao, Jiaxin and Liu, Yiqun and Guo, Jiafeng and Zhang, Min and Ma, Shaoping},
  journal={arXiv preprint arXiv:2110.05789},
  year={2021}
}
```

## Related Work

* **CIKM 2021: [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance](https://arxiv.org/abs/2108.00644) \[[code](https://github.com/jingtaozhan/JPQ)\]: It presents JPQ and greatly improves the efficiency of Dense Retrieval. RepCONC utilizes JPQ for second-stage training.**

* **SIGIR 2021: [Optimizing Dense Retrieval Model Training with Hard Negatives](https://arxiv.org/abs/2104.08051) \[[code](https://github.com/jingtaozhan/DRhard)\]: It provides theoretical analysis on different negative sampling strategies and greatly improves the effectiveness of Dense Retrieval with hard negative sampling. The proposed negative sampling methods are adopted by RepCONC.**

* **ARXIV 2020: [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval](https://arxiv.org/pdf/2006.15498.pdf) \[[code](https://github.com/jingtaozhan/RepBERT-Index)\]: It is one of the pioneer studies about Dense Retrieval.**
