import logging
import torch
import json
import csv
import pytrec_eval
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    corpus_path: str = field()
    query_path: str = field()
    out_corpus_dir: str = field()
    out_query_dir: str = field()
    qrel_path: str = field(default=None)

    data_format: Optional[str] = field(
        default="msmarco", 
        metadata={"choices": ["msmarco", "beir"]}
    )


def concat_title_body(doc: Dict[str, str]):
    body = doc['text'].strip()
    if "title" in doc and len(doc['title'].strip())> 0:
        title = doc['title'].strip()
        if title[-1] in "!.?。！？":
            text = title + " " + body
        else:
            text = title + ". " + body
    else:
        text = body
    return text


def load_corpus(corpus_path, sep_token, verbose=True):
    corpus = {}
    for line in tqdm(open(corpus_path), mininterval=10, disable=not verbose):
        splits = line.strip().split("\t")
        corpus_id, text_fields = splits[0], splits[1:]
        text = f'{sep_token}'.join((t.strip() for t in text_fields))
        corpus[corpus_id] = text[:10000]
    return corpus


def load_queries(query_path):
    queries = {}
    for line in open(query_path):
        qid, text = line.split("\t")
        queries[qid] = text
    return queries


def load_beir_corpus(corpus_path, sep_token, verbose=True):
    corpus = {}
    for line in tqdm(open(corpus_path), mininterval=10, disable=not verbose):
        data = json.loads(line)
        corpus_id = data['_id']
        text = concat_title_body(data)
        corpus[corpus_id] = text[:10000]
    return corpus


def load_beir_queries(query_path):
    queries = {}
    for line in open(query_path):
        data = json.loads(line)
        qid, text = data['_id'], data['text'].strip()
        queries[qid] = text
    return queries


def load_beir_qrels(qrel_path):
    reader = csv.reader(open(qrel_path, encoding="utf-8"), 
            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    qrels = dict()
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels


class TextDataset(Dataset):
    def __init__(self, text_lst: List[str], text_ids: List[int]=None):
        self.text_lst = text_lst
        self.text_ids = text_ids
        assert self.text_ids is None or len(text_lst) == len(text_ids)
    
    def __len__(self):
        return len(self.text_lst)

    def __getitem__(self, item):
        if self.text_ids is not None:
            return self.text_ids[item], self.text_lst[item]
        else:
            return self.text_lst[item]


def get_collator_func(tokenizer, max_length):
    def collator_fn(batch):
        if isinstance(batch[0], tuple):
            ids = torch.LongTensor([x[0] for x in batch])
            features = tokenizer([x[1] for x in batch], padding=True, truncation=True, max_length=max_length)
            return {
                'input_ids': torch.LongTensor(features['input_ids']),
                'attention_mask': torch.LongTensor(features['attention_mask']),
                'text_ids': ids,
            }
        else:
            assert isinstance(batch[0], str)
            features = tokenizer(batch, padding=True, truncation=True, max_length=max_length)
            return {
                'input_ids': torch.LongTensor(features['input_ids']),
                'attention_mask': torch.LongTensor(features['attention_mask']),
            }
    return collator_fn


def truncate_run(run: Dict[str, Dict[str, float]], topk: int):
    new_run = dict()
    for qid, pid2scores in run.items():
        rank_lst = sorted(pid2scores.items(), key=lambda x: x[1], reverse=True)
        new_run[qid] = dict(rank_lst[:topk])
    return new_run


def pytrec_evaluate(
        qrel: Union[str, Dict[str, Dict[str, int]]], 
        run: Union[str, Dict[str, Dict[str, float]]],
        k_values =(1,3,5,10,100),
        mrr_k_values = (10, 100),
        relevance_level = 1,
        ):
    ndcg, map, recall, precision, mrr = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    if isinstance(qrel, str):
        with open(qrel, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)
    if isinstance(run, str):
        with open(run, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {map_string, ndcg_string, recall_string, precision_string}, relevance_level=relevance_level)
    query_scores = evaluator.evaluate(run)
    
    for query_id in query_scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += query_scores[query_id]["ndcg_cut_" + str(k)]
            map[f"MAP@{k}"] += query_scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += query_scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += query_scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(query_scores), 5)
        map[f"MAP@{k}"] = round(map[f"MAP@{k}"]/len(query_scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(query_scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(query_scores), 5)

    mrr_evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {"recip_rank"}, relevance_level=relevance_level)
    for mrr_cut in mrr_k_values:
        mrr_query_scores = mrr_evaluator.evaluate(truncate_run(run, mrr_cut))
        for query_id in mrr_query_scores.keys():
            s = mrr_query_scores[query_id]["recip_rank"]
            mrr[f"MRR@{mrr_cut}"] += s
            query_scores[query_id][f"recip_rank_{mrr_cut}"] = s
        mrr[f"MRR@{mrr_cut}"] = round(mrr[f"MRR@{mrr_cut}"]/len(mrr_query_scores), 5)

    ndcg, map, recall, precision, mrr = dict(ndcg), dict(map), dict(recall), dict(precision), dict(mrr)
    metric_scores = {
        "ndcg": ndcg,
        "map": map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "perquery": query_scores
    }
    return dict(metric_scores)


if __name__ == "__main__":
    metrics = pytrec_evaluate(
        "./data/datasets/msmarco-passage/qrels.dev",
        "./data/model_runs/tct_colbert-v2-hnp/dev.trec"
    )
    print(metrics["mrr"])
    print(metrics["ndcg"])
    print(metrics["map"])