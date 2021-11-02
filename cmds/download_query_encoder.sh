root_dir="./data"

# download for passage dataset

query_encoder_passage_dir="${root_dir}/passage/official_query_encoders "
mkdir -p $query_encoder_passage_dir
cd $query_encoder_passage_dir 

wget https://www.dropbox.com/s/ynnx5qzxxjzgni6/m4.marcopass.query.encoder.tar.gz?dl=0 -O m4.marcopass.query.encoder.tar.gz
tar -xzvf m4.marcopass.query.encoder.tar.gz

wget https://www.dropbox.com/s/o6te6ra8eusonj8/m8.marcopass.query.encoder.tar.gz?dl=0 -O m8.marcopass.query.encoder.tar.gz
tar -xzvf m8.marcopass.query.encoder.tar.gz

wget https://www.dropbox.com/s/gk82r9szdxv4e21/m12.marcopass.query.encoder.tar.gz?dl=0 -O m12.marcopass.query.encoder.tar.gz
tar -xzvf m12.marcopass.query.encoder.tar.gz

wget https://www.dropbox.com/s/ps176d3kno8pjqz/m16.marcopass.query.encoder.tar.gz?dl=0 -O m16.marcopass.query.encoder.tar.gz
tar -xzvf m16.marcopass.query.encoder.tar.gz

wget https://www.dropbox.com/s/yilrlfbrxtzonvs/m24.marcopass.query.encoder.tar.gz?dl=0 -O m24.marcopass.query.encoder.tar.gz
tar -xzvf m24.marcopass.query.encoder.tar.gz

wget https://www.dropbox.com/s/x2lhjn8fu1a5mpm/m32.marcopass.query.encoder.tar.gz?dl=0 -O m32.marcopass.query.encoder.tar.gz
tar -xzvf m32.marcopass.query.encoder.tar.gz

wget https://www.dropbox.com/s/jym4juvd0j98w5x/m48.marcopass.query.encoder.tar.gz?dl=0 -O m48.marcopass.query.encoder.tar.gz
tar -xzvf m48.marcopass.query.encoder.tar.gz


# download for document dataset

cd ../..
query_encoder_doc_dir="doc/official_query_encoders "
mkdir -p $query_encoder_doc_dir
cd $query_encoder_doc_dir

wget https://www.dropbox.com/s/1atamawqr3f8wx6/m4.marcodoc.query.encoder.tar.gz?dl=0 -O m4.marcodoc.query.encoder.tar.gz
tar -xzvf m4.marcodoc.query.encoder.tar.gz

wget https://www.dropbox.com/s/uisl6el92o8y9g5/m8.marcodoc.query.encoder.tar.gz?dl=0 -O m8.marcodoc.query.encoder.tar.gz
tar -xzvf m8.marcodoc.query.encoder.tar.gz

wget https://www.dropbox.com/s/nzmrzbp5r4ghhct/m12.marcodoc.query.encoder.tar.gz?dl=0 -O m12.marcodoc.query.encoder.tar.gz
tar -xzvf m12.marcodoc.query.encoder.tar.gz

wget https://www.dropbox.com/s/vwk0r2rfhht7ntd/m16.marcodoc.query.encoder.tar.gz?dl=0 -O m16.marcodoc.query.encoder.tar.gz
tar -xzvf m16.marcodoc.query.encoder.tar.gz

wget https://www.dropbox.com/s/rb2cc6bkj226n0z/m24.marcodoc.query.encoder.tar.gz?dl=0 -O m24.marcodoc.query.encoder.tar.gz
tar -xzvf m24.marcodoc.query.encoder.tar.gz

wget https://www.dropbox.com/s/l5ww8hy44kyk7bz/m32.marcodoc.query.encoder.tar.gz?dl=0 -O m32.marcodoc.query.encoder.tar.gz
tar -xzvf m32.marcodoc.query.encoder.tar.gz

wget https://www.dropbox.com/s/bsz4v3qu9w3pyuq/m48.marcodoc.query.encoder.tar.gz?dl=0 -O m48.marcodoc.query.encoder.tar.gz
tar -xzvf m48.marcodoc.query.encoder.tar.gz






