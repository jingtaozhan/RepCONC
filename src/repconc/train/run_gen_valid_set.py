import os
import random
import argparse
import subprocess
from typing import Set
from tqdm import tqdm
            

def sample_docs_from_topics(qrel_path: str, run_path: str, topk: int):
    all_qids, sampled_pids = set(), set()
    for line in open(qrel_path):
        qid, _, pid, _ = line.split()
        all_qids.add(qid)
        sampled_pids.add(pid)
    for line in open(run_path):
        qid, _, pid, rank, _, _ = line.split()
        if int(rank) <= topk and qid in all_qids:
            sampled_pids.add(pid)
    return sampled_pids


def output_corpus(in_corpus_path: str, out_corpus_path: str, sampled_docids: Set[str]):
    cnt = 0
    with open(out_corpus_path, 'w') as output_file:
        for line in tqdm(open(in_corpus_path)):
            pid = line.split("\t", maxsplit=1)[0]
            if pid in sampled_docids:
                output_file.write(line)
                cnt += 1
    print(f"Write Cnt: {cnt}, Sample Cnt: {len(sampled_docids)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_corpus_path", type=str, required=True, 
        help="Path to corpus, each line separated by tab, and the first element is id.") 
    parser.add_argument("--input_query_path", type=str, required=True, 
        help="Path to queries, each line separated by tab, and the first element is id.") 
    parser.add_argument("--input_qrel_path", type=str, required=True, 
        help="Path to TREC-format qrel file.") 
    parser.add_argument("--input_run_path", type=str, required=True, 
        help="Path to TREC-format qrel file.")
    parser.add_argument("--topk", type=int, required=True, 
        help="Topk passages/docs per query.")
    parser.add_argument("--output_corpus_path", type=str, required=True, 
        help="Output path of corpus.") 
    parser.add_argument("--output_query_path", type=str, required=True, 
        help="Output path to queries.") 
    parser.add_argument("--output_qrel_path", type=str, required=True, 
        help="Output path to TREC-format qrel file.") 
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_corpus_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_query_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_qrel_path), exist_ok=True)

    subprocess.check_call(["cp", args.input_qrel_path, args.output_qrel_path])
    subprocess.check_call(["cp", args.input_query_path, args.output_query_path])
    docids = sample_docs_from_topics(args.output_qrel_path, args.input_run_path, args.topk)
    output_corpus(
        args.input_corpus_path,
        args.output_corpus_path,
        docids,
    )


if __name__ == "__main__":
    main()