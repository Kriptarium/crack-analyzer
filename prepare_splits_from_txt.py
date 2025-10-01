#!/usr/bin/env python3
import argparse, json
from pathlib import Path
def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]
parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--train", default="train.txt")
parser.add_argument("--val", default="val.txt")
parser.add_argument("--test", default="test.txt")
parser.add_argument("--out", default="splits_from_txt.json")
args = parser.parse_args()
root = Path(args.root)
def make_records(fn):
    lines = read_list(root/fn)
    recs=[]
    for ln in lines:
        img = root/"images"/ln if not (root/ln).exists() else root/ln
        mask = root/"masks"/(Path(img).stem+".png")
        recs.append({"image":str(img),"mask":str(mask),"mask_exists":mask.exists(),"mask_nonzero":mask.exists()})
    return recs
splits={"train":make_records(args.train),"val":make_records(args.val),"test":make_records(args.test)}
open(args.out,"w").write(json.dumps(splits,indent=2))
print("Wrote",args.out)