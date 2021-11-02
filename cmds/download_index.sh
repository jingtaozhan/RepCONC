root_dir="./data"

# download for passage dataset

ivf_passage_dir="${root_dir}/passage/official_ivf_index"
mkdir -p $ivf_passage_dir
cd $ivf_passage_dir 

wget https://www.dropbox.com/s/tz1bn4gk952nqau/m48-l5000.marcopass.ivfpq.index?dl=0 -O m48-l5000.marcopass.ivfpq.index

echo "Finish downloading all ivf indexes for msmarco-passage dataset"

cd ..
pq_passage_dir="official_pq_index"
mkdir -p $pq_passage_dir
cd $pq_passage_dir

wget https://www.dropbox.com/s/uz0qhmlpnpjgtcl/m4-l1.marcopass.ivfpq.index?dl=0 -O m4-l1.marcopass.ivfpq.index

wget https://www.dropbox.com/s/1gzqyu40zthmpll/m8-l1.marcopass.ivfpq.index?dl=0 -O m8-l1.marcopass.ivfpq.index

wget https://www.dropbox.com/s/d2hrr3csl74d5dj/m12-l1.marcopass.ivfpq.index?dl=0 -O m12-l1.marcopass.ivfpq.index

wget https://www.dropbox.com/s/ey4qdwmthrmubcj/m16-l1.marcopass.ivfpq.index?dl=0 -O m16-l1.marcopass.ivfpq.index

wget https://www.dropbox.com/s/ctbrygs32biabvy/m24-l1.marcopass.ivfpq.index?dl=0 -O m24-l1.marcopass.ivfpq.index
 
wget https://www.dropbox.com/s/ij57i9p3dvlw79s/m32-l1.marcopass.ivfpq.index?dl=0 -O m32-l1.marcopass.ivfpq.index

wget https://www.dropbox.com/s/uizj1jqcj6uk8y9/m48-l1.marcopass.ivfpq.index?dl=0 -O m48-l1.marcopass.ivfpq.index

echo "Finish downloading all pq indexes for msmarco-passage dataset"

# download for document dataset

cd ../../
ivf_doc_dir="doc/official_ivf_index"
mkdir -p $ivf_doc_dir
cd $ivf_doc_dir

wget https://www.dropbox.com/s/jbtofskbd7ub73g/m48-l5000.marcodoc.ivfpq.index?dl=0 -O m48-l5000.marcodoc.ivfpq.index

echo "Finish downloading all ivf indexes for msmarco-doc dataset"

cd ..
pq_doc_dir="official_pq_index"
mkdir -p $pq_doc_dir
cd $pq_doc_dir

wget https://www.dropbox.com/s/lq2clm4wpvl3khp/m4-l1.marcodoc.ivfpq.index?dl=0 -O m4-l1.marcodoc.ivfpq.index

wget https://www.dropbox.com/s/jh85cjwynzmqj2h/m8-l1.marcodoc.ivfpq.index?dl=0 -O m8-l1.marcodoc.ivfpq.index

wget https://www.dropbox.com/s/9an52ixpxy6fst8/m12-l1.marcodoc.ivfpq.index?dl=0 -O m12-l1.marcodoc.ivfpq.index

wget https://www.dropbox.com/s/xzhqgttd0p3r5wg/m16-l1.marcodoc.ivfpq.index?dl=0 -O m16-l1.marcodoc.ivfpq.index

wget https://www.dropbox.com/s/2t1riq6dyyhv9yo/m24-l1.marcodoc.ivfpq.index?dl=0 -O m24-l1.marcodoc.ivfpq.index

wget https://www.dropbox.com/s/tx9fs93wke6tox5/m32-l1.marcodoc.ivfpq.index?dl=0 -O m32-l1.marcodoc.ivfpq.index

wget https://www.dropbox.com/s/k9sxe8kwkx0ftz5/m48-l1.marcodoc.ivfpq.index?dl=0 -O m48-l1.marcodoc.ivfpq.index

echo "Finish downloading all pq indexes for msmarco-doc dataset"




