"""
Given a PQ index, this script shows how to construct an IVF accelerated PQ index.
I use a brute-force method by firstly reconstructing all full-dimension embeddings
from the PQ index and then running the IVF-PQ algorithm to build the IVFPQ index. 
It is fast, requiring only several minutes for 8.8 million data pts.

Creation Date : 11/3/2021
Author : Jingtao Zhan <jingtaozhan@gmail.com>
"""
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
import faiss

def reconstruct_embeds_from_pq(ivf_pq_index):
    assert ivf_pq_index.nlist == 1
    invlists = faiss.extract_index_ivf(ivf_pq_index).invlists
    ls = invlists.list_size(0)
    list_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    list_codes = list_codes.reshape(-1, invlists.code_size) 

    centroids = faiss.vector_to_array(ivf_pq_index.pq.centroids)
    centroids = centroids.reshape(invlists.code_size, 256, -1)

    full_dim, sub_dim = centroids.shape[0]*centroids.shape[-1], centroids.shape[-1]
    psg_embeds = np.zeros((len(list_codes), full_dim), dtype=np.float32)

    for i in tqdm(range(48), desc="reconstruct embeds"):
        begin, end = i*sub_dim, (i+1)*sub_dim
        sub_centroids = centroids[i]
        psg_embeds[:, begin:end] = sub_centroids[list_codes[:, i]]
    return psg_embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_index_path", type=str, required=True)
    parser.add_argument("--output_index_path", type=str, required=True)
    parser.add_argument("--nlist", type=int, default=5000, 
        help="The number of inverted lists. more lists -> more fine-grained "
        "segmentation of vector space -> better accuracy. Also more lists -> "
        "more computation overhead.")
    parser.add_argument("--nprobe", type=int, default=500,
        help="This value can be dynamically set during retrieval. "
        "You can change it to any value lying in [1, nlist] now or at a later time. "
        "Ideally, the IVF speedup ratio is nlist/nprobe if the computation overhead "
        "is overlooked (selecting the nearest nprobe lists to the query vector).")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--by_residual", action="store_true")
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)    

    input_index = faiss.read_index(args.input_index_path)
    psg_embeds = reconstruct_embeds_from_pq(input_index)

    coarse_quantizer = faiss.IndexFlatL2(768)
    index = faiss.IndexIVFPQ(coarse_quantizer, 768, args.nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
    
    index.verbose = True
    index.pq.verbose = True
    index.by_residual = args.by_residual
    train_start = timer()
    index.train(psg_embeds)
    add_start = timer()
    index.add(psg_embeds)
    index.nprobe = args.nprobe # default to 10x IVF speedup

    os.makedirs(os.path.dirname(args.output_index_path), exist_ok=True)
    faiss.write_index(index, args.output_index_path)

    
    
    
        
    
    