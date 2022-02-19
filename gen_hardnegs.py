import sys
sys.path.append('./')
import os
import re
import json
import pickle
import faiss
import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from dataset import load_rel, load_rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=str, required=True)
    parser.add_argument("--qrel", type=str, required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rank_dict = load_rank(args.rank)
    rel_dict = load_rel(args.qrel)
    query_ids_set = sorted(rank_dict.keys() | rel_dict.keys())
    for k in tqdm(query_ids_set): # the train query size
        v = rank_dict[k]
        v = list(filter(lambda x:x not in rel_dict[k], v))
        assert len(v) >= args.top, "Not enough irrelevant documents after removing relevant ones"
        v = v[:200]
        rank_dict[k] = v
    json.dump(rank_dict, open(args.output, 'w'))
                    