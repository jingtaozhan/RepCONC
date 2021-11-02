# coding=utf-8
import os
from urllib import parse
import torch
import faiss
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig
from model.modeling import QuantDot, QuantDot
from timeit import default_timer as timer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from dataset import TextTokenIdsCache, SequenceDataset, get_collate_function

logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_index(index_path, use_cuda, faiss_gpu_index):
    index = faiss.read_index(index_path)
    if use_cuda:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024)
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index)
    return index


def query_inference(model, index, args):
    query_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(
            data_dir=args.preprocess_dir, 
            prefix=f"{args.mode}-query"),
        max_seq_length=args.max_query_length
    )
    
    model = model.to(args.device)
    dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.batch_size,
        collate_fn=get_collate_function(args.max_query_length),
        drop_last=False,
    )
    batch_size = dataloader.batch_size
    num_examples = len(dataloader.dataset)
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)
    model.eval()

    all_search_results = []
    for inputs, ids in tqdm(dataloader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        with torch.no_grad():
            query_embeds = model(**inputs).detach().cpu().numpy()
            batch_results = index.search(query_embeds, args.topk)[1]
            all_search_results.extend(batch_results.tolist())
    return all_search_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["dev", "test"], required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--query_encoder_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--gpu_search", action="store_true")
    parser.add_argument("--nprobe", type=int, default=None)
    args = parser.parse_args()

    args.device = torch.device("cuda:0")
    args.n_gpu = 1
    
    config_class, model_class = RobertaConfig, QuantDot
    
    config = config_class.from_pretrained(args.query_encoder_dir)
    model = model_class.from_pretrained(args.query_encoder_dir, config=config,)

    index = faiss.read_index(args.index_path)

    if args.nprobe is None:
        args.nprobe = index.nlist
        logger.info(f"Automatically set nprobe to {args.nprobe} = nlist")
    else:
        assert args.nprobe > 0, "nprobe should be a positive number"
        assert args.nprobe <= index.nlist, f"nprobe should range from [1, nlist(index.nlist)]" 
    index.nprobe = args.nprobe

    if not args.gpu_search:
        faiss.omp_set_num_threads(32)
    else:
        if index.nlist > 1:
            logger.warning("Faiss requires setting by_residual=True for GPU acceleration. \
                However, this option is set to False when generating IVF accelerated RepCONC indexes. \
                And setting it to True during evaluation may hurt ranking effectiveness.")
            logger.warning("Setting by_residual to True and precomputing tables ...")
            index.by_residual = True
            index.precompute_table()
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024)
        index = faiss.index_cpu_to_gpu(res, 0, index)

    all_search_results = query_inference(model, index, args)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as outputfile:
        for qid, pids in enumerate(all_search_results):
            for idx, pid in enumerate(pids):
                rank = idx+1
                outputfile.write(f"{qid}\t{pid}\t{rank}\n")


if __name__ == "__main__":
    main()
