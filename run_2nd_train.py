import sys
import os
import torch
import random
import time
import faiss
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from transformers import (AdamW, 
    get_linear_schedule_with_warmup,
    RobertaConfig)
from dataset import (TextTokenIdsCache, 
    SequenceDataset, load_rel, pack_tensor_2D)
from modeling import QuantDot
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))


class TrainQueryDataset(SequenceDataset):
    def __init__(self, queryids_cache, 
            rel_file, max_query_length):
        SequenceDataset.__init__(self, queryids_cache, max_query_length)
        self.reldict = load_rel(rel_file)

    def __getitem__(self, item):
        ret_val = super().__getitem__(item)
        ret_val['rel_pids'] = self.reldict[item]
        ret_val['rel_pid'] = random.choice(self.reldict[item])
        return ret_val


def get_query_collate_function(max_seq_length):
    def collate_function(batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids, default=1, 
                dtype=torch.int64, length=None),
            "attention_mask": pack_tensor_2D(attention_mask, default=0, 
                dtype=torch.int64, length=None),
        }
        qoffsets = [x['id'] for x in batch]
        rel_pids = [x["rel_pid"] for x in batch]
        all_rel_pids = [x["rel_pids"] for x in batch]
        return data, qoffsets, rel_pids, all_rel_pids
    return collate_function  


def get_doc_embeds(psg_ids, pq_codes, centroids):
    M = centroids.shape[0]
    first_indices = torch.arange(M).to(centroids.device)
    first_indices = first_indices.expand(len(psg_ids), M).reshape(-1)
    second_indices = pq_codes[psg_ids].reshape(-1)
    embeddings = centroids[first_indices, second_indices].reshape(len(psg_ids), -1)
    return embeddings


def get_scores(query_embeds, psg_ids, pq_codes, centroids):
    # query_embeds: bs, dim; psg_ids: bs, k
    # pq_codes: N, M; centroids: M, 256, _dim
    bs, M, cent_per_subv = len(query_embeds), centroids.size(0), centroids.size(1)
    query_embeds = query_embeds.reshape(len(query_embeds), M, 1, -1) 
    centroid_scores = (query_embeds
        * centroids.unsqueeze(0)).sum(-1) # bs, M, 256
    # centroid_scores = centroid_scores.unsqueeze(1).expand(
    #         bs, psg_ids.size(1), M, cent_per_subv) # bs, k, M, 256
    psg_pq_codes = pq_codes[psg_ids.reshape(-1)].reshape(*psg_ids.shape, M) # bs, k, M
    # all_scores = torch.gather(centroid_scores, 3, psg_pq_codes.unsqueeze(3)).squeeze(3).sum(-1) # bs, k
    all_scores = torch.gather(centroid_scores, 2, psg_pq_codes.transpose(1,2)).transpose(1,2).sum(-1)
    return all_scores


def eval_mrr_recall(batch_neighbors, all_rel_pids):
    mrr, recall = 0, 0
    for retrieve_pids, cur_rel_pids in zip(
        batch_neighbors, all_rel_pids):
        for idx, pid in enumerate(retrieve_pids[:10]):
            if pid in cur_rel_pids:
                mrr += 1/(idx+1) 
                recall += 1  # wrong in a strict sense
                break
    mrr /= len(batch_neighbors)
    recall /= len(batch_neighbors)
    return mrr, recall


def compute_loss(query_embeddings, pq_codes, centroids, 
        batch_neighbors, all_rel_pids, device, loss_choice):
    loss = 0
    neg_masks = torch.FloatTensor([[0] + [1 if x in rels else 0 for x in nn] 
        for rels, nn in zip(all_rel_pids, batch_neighbors)]).to(device)
    batch_neighbors = torch.from_numpy(batch_neighbors).to(device)
    sampled_rels = torch.tensor([random.choice(rels) for rels in all_rel_pids]
        , dtype=torch.long, device=device)[:, None]
    all_pids = torch.hstack((sampled_rels, batch_neighbors))
    all_scores = get_scores(query_embeddings, all_pids, pq_codes, centroids)
    if loss_choice == "lambda": # compute pair weights
        with torch.no_grad():
            rel_rank = all_scores.size(1) - (
                all_scores[:, :1] >= all_scores[:, 1:]).sum(dim=-1, keepdim=True) # bs, 1
            neg_rank = torch.arange(1, all_scores.size(1), device=rel_rank.device).unsqueeze(0) # 1, neg
            weights = torch.abs(1/rel_rank - 1/neg_rank).reshape(-1)
    all_scores -= neg_masks * 100000.0
    if loss_choice in ["pair", "lambda"]:
        bs, hard_num = len(all_scores), all_scores.size(1)-1
        rel_scores = all_scores[:, :1].expand(bs, hard_num).reshape(-1)
        logit_matrix = torch.cat([rel_scores.unsqueeze(1),
                    all_scores[:, 1:].reshape(-1).unsqueeze(1)], dim=1)  
        lsm = torch.nn.functional.log_softmax(logit_matrix, dim=1)
        sep_loss = -1.0 * lsm[:, 0] # pairwise loss
        if loss_choice == "lambda":
            loss = (sep_loss * weights).sum() / len(sep_loss) # weight pairs
        else:
            loss = sep_loss.mean()
    elif loss_choice == "list":
        loss = torch.nn.CrossEntropyLoss()(all_scores, torch.zeros(len(all_scores), dtype=torch.long, device=device))
    else:
        raise NotImplementedError()
    return loss


def create_optimizer_and_scheduler(model, args, num_training_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                if "centroids" not in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.001, # same as static hard negative training
        },
        {
            "params": [p for n, p in model.named_parameters() 
                if "centroids" not in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {   "params": [p for n, p in model.named_parameters() 
            if "centroids" in n ],
            "weight_decay": 0.0, 'lr': args.centroid_lr
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
        lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps)
    return optimizer, scheduler

        
def train(args, model, pq_codes, ivf_index):
    """ Train the model """
    res = faiss.StandardGpuResources()
    res.setTempMemory(128*1024*1024)
    co = faiss.GpuClonerOptions()
    co.useFloat16 = ivf_index.pq.M >= 56
    gpu_ivf_index = None
    gpu_ivf_index = faiss.index_cpu_to_gpu(res, 0, ivf_index, co)

    tb_writer = SummaryWriter(args.log_dir)

    train_dataset = TrainQueryDataset(
        TextTokenIdsCache(args.preprocess_dir, "train-query"),
        os.path.join(args.preprocess_dir, "train-qrel.tsv"),
        args.max_seq_length
    )

    train_sampler = RandomSampler(train_dataset) 
    collate_fn = get_query_collate_function(args.max_seq_length)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(args.warmup_ratio * t_total)
    print("Warmup steps: ", args.warmup_steps)

    optimizer, scheduler = create_optimizer_and_scheduler(model, args, t_total)
    optimizer.load_state_dict(
        torch.load(os.path.join(args.init_model_path, "optimizer.pt"), map_location=args.model_device)
    )
    optimizer.param_groups[0]['initial_lr'] = args.lr
    optimizer.param_groups[1]['initial_lr'] = args.lr
    optimizer.param_groups[2]['initial_lr'] = args.centroid_lr
    optimizer.param_groups[0]['lr'] = 0.0
    optimizer.param_groups[1]['lr'] = 0.0
    optimizer.param_groups[2]['lr'] = 0.0

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step, global_iterate = 0, 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)  
    
    tr_loss, logging_loss = 0.0, 0.0
    tr_mrr, logging_mrr = 0.0, 0.0
    tr_recall, logging_recall = 0.0, 0.0
    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        
        for step, (batch, _, _, all_rel_poffsets) in enumerate(epoch_iterator):

            batch = {k:v.to(args.model_device) for k, v in batch.items()}
            model.train()            
            query_embeds = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"])
            
            retrieve_scores, batch_neighbors = gpu_ivf_index.search(
                    query_embeds.detach().cpu().numpy(), args.neg_topk)
            mrr, recall = eval_mrr_recall(batch_neighbors, all_rel_poffsets)

            loss = compute_loss(
                    query_embeds, 
                    pq_codes, model.centroids,
                    batch_neighbors, all_rel_poffsets, 
                    model.device, args.loss)   
        
            tr_mrr += mrr
            tr_recall += recall     
            tr_loss += loss.item()

            loss /= args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # if args.n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                faiss.copy_array_to_vector(
                    model.centroids.detach().cpu().numpy().ravel(), 
                    ivf_index.pq.centroids)
                gpu_ivf_index = None
                gpu_ivf_index = faiss.index_cpu_to_gpu(res, 0, ivf_index, co)
            
            global_iterate += 1
            if args.logging_steps > 0 and global_iterate % args.logging_steps == 0:
                log_step = global_iterate * args.train_batch_size
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], log_step)
                cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                tb_writer.add_scalar('train/all_loss', cur_loss, log_step)
                logging_loss = tr_loss

                cur_mrr =  (tr_mrr - logging_mrr)/(args.logging_steps)
                tb_writer.add_scalar('train/mrr_10', cur_mrr, log_step)
                logging_mrr = tr_mrr

                cur_recall =  (tr_recall - logging_recall)/(args.logging_steps)
                tb_writer.add_scalar('train/recall_10', cur_recall, log_step)
                logging_recall = tr_recall
        
        save_model(model, args.model_save_dir, f'epoch-{epoch_idx+1}', args)
        faiss.write_index(ivf_index,
            os.path.join(args.model_save_dir, f'epoch-{epoch_idx+1}', "index"))


def run_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--model_save_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--init_index_path", type=str, required=True)
    parser.add_argument("--init_model_path", type=str, required=True)
    parser.add_argument("--loss", choices=["list", "pair", "lambda"], required=True)

    parser.add_argument("--centroid_lr", type=float, required=True)
    parser.add_argument("--lr", default=2e-6, type=float)
    parser.add_argument("--gpu_search", action="store_true")

    parser.add_argument("--neg_topk", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", default=0.05, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=6, type=int)
    parser.add_argument("--use_cross_entropy_loss", action="store_true")

    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)
    os.makedirs(args.model_save_dir, exist_ok=True)
    return args


def main():
    args = run_parse_args()
    logger.info(args)

    # Setup CUDA, GPU 
    args.model_device = torch.device(f"cuda:0")
    args.n_gpu = 1

    # Setup logging
    logger.warning("Model Device: %s, n_gpu: %s", args.model_device, args.n_gpu)

    # Set seed
    set_seed(args)
    
    config = RobertaConfig.from_pretrained(args.init_model_path)
    config.return_dict = False
    config.gradient_checkpointing = True
    model = QuantDot.from_pretrained(args.init_model_path, config=config)

    ivf_index = faiss.read_index(args.init_index_path)

    # extract Index Assignments from the index
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)
    pq_codes = torch.LongTensor(pq_codes).to(args.model_device)

    # check the index parameters and model parameters match each other
    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
    centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
    centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
    faiss.copy_array_to_vector(centroid_embeds.ravel(), ivf_index.pq.centroids)
    assert (coarse_embeds == 0).all()
    centroid_embeds = torch.FloatTensor(centroid_embeds)
    assert (model.centroids == centroid_embeds).all()

    model.to(args.model_device)
    train(args, model, pq_codes, ivf_index)
    

if __name__ == "__main__":
    main()
