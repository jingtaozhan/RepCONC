import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--qrel_path", type=str, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    qrel = defaultdict(set)

    for line in tqdm(open(args.qrel_path)):
        query_id, _, object_id, relevance = line.strip().split()
        if int(relevance) > 0:
            qrel[query_id].add(object_id)

    hardneg = defaultdict(list)

    for line in tqdm(open(args.run_path)):
        query_id, _, object_id, ranking, score, _ = line.strip().split()
        if int(ranking) <= args.topk and object_id not in qrel[query_id]:
            hardneg[query_id].append(object_id)

    json.dump(hardneg, open(args.output_path, 'w'))


if __name__ == "__main__":
    main()