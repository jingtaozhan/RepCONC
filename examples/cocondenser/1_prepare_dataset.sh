#!/bin/bash

set -e

root="./data/cocondenser-marco-passage"
dataset_dir="${root}/dataset"

mkdir -p $dataset_dir
cd $dataset_dir


echo "download MSMARCO passage data"

echo "start downloading corpus from RocketQA"

echo "Note: coCondenser (and its major baseline, RocketQA) cleans the corpus and adds passages with titles. This has caused some debate since the original benchmark does not contain the title field (see https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor). If the original corpus is used, the MRR@10 drops from 0.38 to 0.35. Anyway, this example aims to show the ability of RepCONC to compress the index of coCondenser. We keep its original design to use the modified corpus, but we also agree that such modification may result in unjustified model comparisons."

wget -nc --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar  --skip-old-files -zxvf marco.tar.gz


if [ -f "corpus.tsv" ]; then
    echo "corpus.tsv exists."
else 
    # https://github.com/texttron/tevatron/blob/2eb047255bb0ad973da130a2d7c7e76fc9e26dd4/examples/coCondenser-marco/get_data.sh#L11
    # During test, the command works for zsh but not bash.
    join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 marco/para.txt) <(sort -k1,1 marco/para.title.txt) | sort -k1,1 -n > corpus.tsv
fi

echo "start downloading data from MS, this may take some time depending on the network"
wget -nc --no-check-certificate https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar --skip-old-files -zxvf collectionandqueries.tar.gz -C ./

echo "start downloading queries and qrels"

wget -nc --no-check-certificate https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip ./msmarco-test2019-queries.tsv.gz 

wget -nc --no-check-certificate https://trec.nist.gov/data/deep/2019qrels-pass.txt

wget -nc --no-check-certificate https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip ./msmarco-test2020-queries.tsv.gz 

wget -nc --no-check-certificate https://trec.nist.gov/data/deep/2020qrels-pass.txt

echo "finish downloading"

echo "Create symbolic link"

ln -sf queries.train.tsv query.train
ln -sf qrels.train.tsv qrels.train

ln -sf queries.dev.small.tsv query.dev
ln -sf qrels.dev.small.tsv qrels.dev

ln -sf msmarco-test2019-queries.tsv query.trec19
ln -sf 2019qrels-pass.txt qrels.trec19

ln -sf msmarco-test2020-queries.tsv query.trec20
ln -sf 2020qrels-pass.txt qrels.trec20

echo "Done"

