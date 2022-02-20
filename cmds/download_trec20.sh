set -e 

mkdir -p data/passage/trec20-test
cd data/passage/trec20-test

# download MSMARCO passage data

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip ./msmarco-test2020-queries.tsv.gz 

wget https://trec.nist.gov/data/deep/2020qrels-pass.txt

# also download trec19 for convenience when validating the correctness of tokenize-retrieve script
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip ./msmarco-test2019-queries.tsv.gz 

wget https://trec.nist.gov/data/deep/2019qrels-pass.txt

cd ../../../
mkdir -p data/doc/trec20-test
cd data/doc/trec20-test

# download MSMARCO doc data

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip msmarco-test2020-queries.tsv.gz

wget https://trec.nist.gov/data/deep/2020qrels-docs.txt

# also download trec19 for convenience when validating the correctness of tokenize-retrieve script
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz

wget https://trec.nist.gov/data/deep/2019qrels-docs.txt