set -e 

root_dir="./data"

# download for passage dataset

doc_encoder_passage_dir="${root_dir}/passage/official_doc_encoders "
mkdir -p $doc_encoder_passage_dir
cd $doc_encoder_passage_dir 

wget https://www.dropbox.com/s/l4e6pgk6qy5o86b/m4.marcopass.pq.encoder.tar.gz?dl=0 -O m4.marcopass.pq.encoder.tar.gz
tar -xzvf m4.marcopass.pq.encoder.tar.gz

wget https://www.dropbox.com/s/8s6kzpxmcv4pqjf/m8.marcopass.pq.encoder.tar.gz?dl=0 -O m8.marcopass.pq.encoder.tar.gz
tar -xzvf m8.marcopass.pq.encoder.tar.gz

wget https://www.dropbox.com/s/3mwhyue1gpueu1p/m12.marcopass.pq.encoder.tar.gz?dl=0 -O m12.marcopass.pq.encoder.tar.gz
tar -xzvf m12.marcopass.pq.encoder.tar.gz

wget https://www.dropbox.com/s/ta71ha3px1jd4u5/m16.marcopass.pq.encoder.tar.gz?dl=0 -O m16.marcopass.pq.encoder.tar.gz
tar -xzvf m16.marcopass.pq.encoder.tar.gz

wget https://www.dropbox.com/s/ryrvtn68bdnmawh/m24.marcopass.pq.encoder.tar.gz?dl=0 -O m24.marcopass.pq.encoder.tar.gz
tar -xzvf m24.marcopass.pq.encoder.tar.gz

wget https://www.dropbox.com/s/9zbzj569wgzvgmk/m32.marcopass.pq.encoder.tar.gz?dl=0 -O m32.marcopass.pq.encoder.tar.gz
tar -xzvf m32.marcopass.pq.encoder.tar.gz

wget https://www.dropbox.com/s/wj7gfwytzd0fkgp/m48.marcopass.pq.encoder.tar.gz?dl=0 -O m48.marcopass.pq.encoder.tar.gz
tar -xzvf m48.marcopass.pq.encoder.tar.gz

# download for document dataset

cd ../..
doc_encoder_doc_dir="doc/official_doc_encoders "
mkdir -p $doc_encoder_doc_dir
cd $doc_encoder_doc_dir

wget https://www.dropbox.com/s/nk6blrll9d6jxqi/m4.marcodoc.pq.encoder.tar.gz?dl=0 -O m4.marcodoc.pq.encoder.tar.gz
tar -xzvf m4.marcodoc.pq.encoder.tar.gz

wget https://www.dropbox.com/s/2hqr4ccnff35g8q/m8.marcodoc.pq.encoder.tar.gz?dl=0 -O m8.marcodoc.pq.encoder.tar.gz
tar -xzvf m8.marcodoc.pq.encoder.tar.gz

wget https://www.dropbox.com/s/0py48j37zt4gqid/m12.marcodoc.pq.encoder.tar.gz?dl=0 -O m12.marcodoc.pq.encoder.tar.gz
tar -xzvf m12.marcodoc.pq.encoder.tar.gz

wget https://www.dropbox.com/s/i2gki5wcfmv5kpa/m16.marcodoc.pq.encoder.tar.gz?dl=0 -O m16.marcodoc.pq.encoder.tar.gz
tar -xzvf m16.marcodoc.pq.encoder.tar.gz

wget https://www.dropbox.com/s/4shx0ptvenp0qu1/m24.marcodoc.pq.encoder.tar.gz?dl=0 -O m24.marcodoc.pq.encoder.tar.gz
tar -xzvf m24.marcodoc.pq.encoder.tar.gz

wget https://www.dropbox.com/s/lvkh0zjpzoxi5c3/m32.marcodoc.pq.encoder.tar.gz?dl=0 -O m32.marcodoc.pq.encoder.tar.gz
tar -xzvf m32.marcodoc.pq.encoder.tar.gz

wget https://www.dropbox.com/s/lk61salc8a44fly/m48.marcodoc.pq.encoder.tar.gz?dl=0 -O m48.marcodoc.pq.encoder.tar.gz
tar -xzvf m48.marcodoc.pq.encoder.tar.gz



