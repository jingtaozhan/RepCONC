set -e 

cd data

wget https://trec.nist.gov/trec_eval/trec_eval-9.0.7.tar.gz
tar -xzvf trec_eval-9.0.7.tar.gz

cd trec_eval-9.0.7
make