import argparse, json
from pathlib import Path
def readlist(p): return [l.strip() for l in open(p) if l.strip()]
parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--train", default="train.txt")
parser.add_argument("--val", default="val.txt")
parser.add_argument("--test", default="test.txt")
parser.add_argument("--out", default="splits.json")
args = parser.parse_args()
root = Path(args.root)
def make(fn):
    p = root/fn
    lines = readlist(p) if p.exists() else []
    recs=[]
    for l in lines:
        img = root/"images"/l if (root/"images"/l).exists() else (root/l if (root/l).exists() else l)
        mask = root/"masks"/(Path(img).stem+".png")
        recs.append({"image":str(img),"mask":str(mask),"mask_exists":mask.exists(),"mask_nonzero":mask.exists()})
    return recs
spl = {"train":make(args.train),"val":make(args.val),"test":make(args.test)}
open(args.out,"w").write(json.dumps(spl,indent=2))
print("Wrote",args.out)