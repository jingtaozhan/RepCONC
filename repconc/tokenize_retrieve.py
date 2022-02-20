# coding=utf-8
"""
This script tokenizes queries and then runs retrieval, 
while run_retrieval.py uses pre-tokenized queries and only runs retrieval.
"""
import os
import faiss
import torch
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig
from timeit import default_timer as timer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from jpq.star_tokenizer import RobertaTokenizer
from repconc.model import QuantDot
from repconc.dataset import get_collate_function


class TRECQueryDataset(Dataset):
    def __init__(self, query_file_path, max_query_length):
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", do_lower_case = True, cache_dir=None)
        self.text_queries = []
        for line in open(query_file_path, 'r'):
            qid, query = line.split("\t")
            qid, query = int(qid), query.strip()
            self.text_queries.append((qid, query))
        self.max_query_length = max_query_length

    def __len__(self):  
        return len(self.text_queries)

    def __getitem__(self, item):
        qid, text = self.text_queries[item]
        input_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.max_query_length,
            truncation=True)
        attention_mask = [1]*len(input_ids)

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": qid,
        }
        return ret_val


def query_inference(model, index, args):
    query_dataset = TRECQueryDataset(
        query_file_path=args.query_file_path,
        max_query_length=args.max_query_length
    )
    
    model = model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.batch_size,
        collate_fn=get_collate_function(args.max_query_length),
        drop_last=False,
    )
    batch_size = dataloader.batch_size
    num_examples = len(dataloader.dataset)
    print("  Num examples = %d", num_examples)
    print("  Batch size = %d", batch_size)

    model.eval()

    all_query_ids = []
    all_search_results_pids, all_search_results_scores = [], []
    for inputs, ids in tqdm(dataloader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        all_query_ids.extend(ids)
        with torch.no_grad():
            query_embeds = model(**inputs).detach().cpu().numpy()
            batch_results_scores, batch_results_pids = index.search(query_embeds, args.topk)
            all_search_results_pids.extend(batch_results_pids.tolist())
            all_search_results_scores.extend(batch_results_scores.tolist())
    return all_query_ids, all_search_results_scores, all_search_results_pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_file_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--query_encoder_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_format", type=str, choices=["msmarco", "trec"], required=True)

    # these two arguments are used simply for converting offset pids to official pids
    parser.add_argument("--pid2offset_path", type=str, required=True)
    # msmarco doc use D... as document ids, if dataset == "doc", we need to explicitly add "D" as prefix
    # preprocess.py shoud have saved this D in the pid2offset.pickle ...
    parser.add_argument("--dataset", type=str, choices=["doc", "passage"], required=True)

    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--nprobe", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--gpu_search", action="store_true")
    
    args = parser.parse_args()


    if args.gpu_search:
        args.device = torch.device("cuda:0")
        args.n_gpu = 1
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    
    config_class, model_class = RobertaConfig, QuantDot
    
    config = config_class.from_pretrained(args.query_encoder_dir)
    model = model_class.from_pretrained(args.query_encoder_dir, config=config,)
    index = faiss.read_index(args.index_path)

    if args.nprobe is None:
        args.nprobe = index.nlist
        print(f"Automatically set nprobe to {args.nprobe} = nlist")
    else:
        assert args.nprobe > 0, "nprobe should be a positive number"
        assert args.nprobe <= index.nlist, f"nprobe should range from [1, nlist(index.nlist)]" 
    index.nprobe = args.nprobe

    if not args.gpu_search:
        faiss.omp_set_num_threads(32)
    else:
        if index.nlist > 1:
            print("Faiss requires setting by_residual=True for GPU acceleration. \
                However, this option is set to False when generating IVF accelerated RepCONC indexes. \
                And setting it to True during evaluation may hurt ranking effectiveness.")
            print("Setting by_residual to True and precomputing tables ...")
            index.by_residual = True
            index.precompute_table()
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024)
        index = faiss.index_cpu_to_gpu(res, 0, index)

    all_query_ids, all_search_results_scores, all_search_results_pids = \
        query_inference(model, index, args)

    if args.dataset == "doc":
        pid2offset = pickle.load(open(args.pid2offset_path, 'rb'))
        offset2pid = {v:f"D{k}" for k, v in pid2offset.items()}
    else:
        assert args.dataset == "passage"
        pid2offset = {i:i for i in range(8841823)}
        offset2pid = {v:k for k, v in pid2offset.items()}

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as outputfile:
        for qid, scores, poffsets in zip(all_query_ids, 
                all_search_results_scores, all_search_results_pids):
            for idx, (score, poffset) in enumerate(zip(scores, poffsets)):
                rank = idx+1
                pid = offset2pid[poffset]
                if args.output_format == "msmarco":
                    outputfile.write(f"{qid}\t{pid}\t{rank}\n")
                else:
                    assert args.output_format == "trec" 
                    index_name = os.path.basename(args.index_path)
                    outputfile.write(f"{qid} Q0 {pid} {rank} {score} JPQ-{index_name}\n")


if __name__ == "__main__":
    main()
