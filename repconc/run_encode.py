# coding=utf-8
import os
import torch
import faiss
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from repconc.model import QuantDot
from repconc.dataset import TextTokenIdsCache, SequenceDataset, get_collate_function

logger = logging.Logger(__name__, level=logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s- %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def gen_fake_index(M, D):
    # N: number of data points
    # D: dim of each data point
    # M: number of sub-vectors

    # We generate a IVFPQ index, and set the nlists=1
    # So in fact, it is a PQ index
    # We add IVF because IVFPQ supports GPU search
    # while PQ does not.
    coarse_quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFPQ(coarse_quantizer, D, 1, M, 
        8, faiss.METRIC_INNER_PRODUCT)
    fake_train_pts = np.random.random((10000, D)).astype(np.float32)
    index.train(fake_train_pts) # fake training
    return index
    

def doc_inference(model, index, args):
    doc_dataset = SequenceDataset(
        ids_cache=TextTokenIdsCache(
            data_dir=args.preprocess_dir, 
            prefix=f"passages"),
        max_seq_length=args.max_doc_length
    )
    
    model = model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    dataloader = DataLoader(
        doc_dataset,
        sampler=SequentialSampler(doc_dataset),
        batch_size=args.batch_size,
        collate_fn=get_collate_function(args.max_doc_length),
        drop_last=False,
    )
    batch_size = dataloader.batch_size
    num_examples = len(dataloader.dataset)
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    for inputs, ids in tqdm(dataloader):        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(args.device)
        model.eval()
        with torch.no_grad():
            embeds = model(**inputs).detach().cpu().numpy()
            index.add(embeds)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--doc_encoder_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--query_encoder_dir", type=str, default=None,
        help="RepCONC has two-stage training. In the first-stage, it trains a unified query/document encoder"
        " using constrained clustering. In the second-stage, it further optimizes the query"
        " encoder and centroid embeddings. If this argument is given, the centroid embeddings"
        " will be replaced and the ranking effectiveness should be better.")
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    faiss.omp_set_num_threads(args.threads)

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    
    config_class, model_class = RobertaConfig, QuantDot
    
    config = config_class.from_pretrained(args.doc_encoder_dir)
    model = model_class.from_pretrained(args.doc_encoder_dir, config=config)

    fake_index = gen_fake_index(config.MCQ_M, config.hidden_size)
    # discard coarse quantizer
    coarse_quantizer = faiss.downcast_index(fake_index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    coarse_embeds[:] = 0
    faiss.copy_array_to_vector(coarse_embeds.ravel(), coarse_quantizer.xb)
    # set centroid values 
    centroids = faiss.vector_to_array(fake_index.pq.centroids)
    centroids = centroids.reshape(config.MCQ_M, 256, -1)
    centroids[:] = model.centroids.data.detach().cpu().numpy()
    faiss.copy_array_to_vector(centroids.ravel(), fake_index.pq.centroids)
    # some other stuffs
    fake_index.precomputed_table.clear()
    fake_index.precompute_table()

    index = fake_index
    # let's begin encoding corpus!
    index = doc_inference(model, index, args)

    if args.query_encoder_dir is not None:
        print("Replace centroid embeddings using the provided query encoder")
        config = config_class.from_pretrained(args.query_encoder_dir)
        query_encoder = model_class.from_pretrained(args.query_encoder_dir, config=config)
        centroids = faiss.vector_to_array(index.pq.centroids)
        centroids = centroids.reshape(config.MCQ_M, 256, -1)
        centroids[:] = query_encoder.centroids.data.detach().cpu().numpy()
        faiss.copy_array_to_vector(centroids.ravel(), index.pq.centroids)
    else:
        print("No query encoder provided. Using centroids from the document encoder.")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    faiss.write_index(index, args.output_path)


if __name__ == "__main__":
    main()
