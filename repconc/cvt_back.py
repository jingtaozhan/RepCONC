import os
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["train", "dev", "test", "lead"], required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["passage", "doc"], required=True)
    args = parser.parse_args()

    pid2offset = pickle.load(open(os.path.join(args.preprocess_dir, "pid2offset.pickle"), 'rb'))
    offset2pid = {v:k for k, v in pid2offset.items()}
    qid2offset = pickle.load(open(os.path.join(args.preprocess_dir, f"{args.mode}-qid2offset.pickle"), 'rb'))
    offset2qid = {v:k for k, v in qid2offset.items()}

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as output:
        for line in open(args.input_path):
            if args.mode == "test": # TREC Format
                qoffset, _, poffset, rank, score, system = line.split()
                qoffset, poffset, rank = int(qoffset), int(poffset), int(rank)
                qid, pid = offset2qid[qoffset], offset2pid[poffset]
                if args.dataset == "doc":
                    output.write(f"{qid} Q0 D{pid} {rank} {score} {system}\n")
                else:
                    output.write(f"{qid} Q0 {pid} {rank} {score} {system}\n")
            else:
                qoffset, poffset, rank = line.split()
                qoffset, poffset, rank = int(qoffset), int(poffset), int(rank)
                qid, pid = offset2qid[qoffset], offset2pid[poffset]
                if args.dataset == "doc":
                    output.write(f"{qid}\tD{pid}\t{rank}\n")
                else:
                    output.write(f"{qid}\t{pid}\t{rank}\n")