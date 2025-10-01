#!/usr/bin/env python3
import json, sys, os, math, argparse, time
from pathlib import Path
from PIL import Image
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score, precision_recall_curve, auc
from torchvision import transforms, models
import torch
from postprocess_improved import improved_postprocess_from_probs

def load_model(path, device):
    model = models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256,1,kernel_size=1)
    model.aux_classifier=None
    ck = torch.load(path, map_location=device)
    sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def infer_prob(model, pil, device):
    tf = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    x = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)['out']
        prob = torch.sigmoid(out).cpu().numpy()[0,0]
    return prob

def flatten_pair(gt, prob, thr=0.5):
    gtf = (gt.flatten()>127).astype(int)
    pf = (prob.flatten()>thr).astype(int)
    return gtf, pf

def save_example(img_path, gt_mask, prob, out_folder, name_prefix, thr=0.5):
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    im = Image.open(img_path).convert("RGB")
    prob_img = (prob*255).astype("uint8")
    mask_pp = improved_postprocess_from_probs(prob, thresh=thr)
    # save overlay and prob images
    overlay = np.array(im.resize((512,512))).copy()
    overlay_mask = np.zeros_like(overlay)
    overlay_mask[(mask_pp>127)] = [255,0,0]
    import cv2
    over = cv2.addWeighted(overlay,1.0,overlay_mask,0.6,0)
    Image.fromarray(over).save(Path(out_folder)/f"{name_prefix}_overlay.png")
    Image.fromarray(prob_img).save(Path(out_folder)/f"{name_prefix}_prob.png")
    Image.fromarray(mask_pp).save(Path(out_folder)/f"{name_prefix}_mask.png")
    # save original resized
    Image.fromarray(np.array(im.resize((512,512)))).save(Path(out_folder)/f"{name_prefix}_img.png")

def run_report(splits_json, model_pth, out_dir, thr=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(splits_json))
    test = splits.get("test", [])
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = load_model(model_pth, device)
    all_gt = []
    all_prob = []
    per_image = []
    t0 = time.time()
    for r in test:
        img = r["image"]; mask = r["mask"]
        if not Path(img).exists() or not Path(mask).exists(): 
            print("Skipping missing:", img); continue
        pil = Image.open(img).convert("RGB")
        gt = np.array(Image.open(mask).convert("L").resize((512,512)))
        prob = infer_prob(model, pil, device)
        all_gt.append((gt.flatten()>127).astype(int))
        all_prob.append(prob.flatten())
        # compute per-image IoU at thr and store FP/FN counts
        pred_bin = (prob>thr).astype(int)
        gt_bin = (gt>127).astype(int)
        iou = jaccard_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=1)
        prec = precision_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=1)
        rec = recall_score(gt_bin.flatten(), pred_bin.flatten(), zero_division=1)
        per_image.append({"image":img, "mask":mask, "iou":float(iou), "prec":float(prec), "rec":float(rec)})
        # save examples for worst/best later
    # flatten for overall metrics
    all_gt_flat = np.concatenate(all_gt)
    all_prob_flat = np.concatenate(all_prob)
    # compute precision-recall curve & AUC-PR
    precision, recall, thresholds = precision_recall_curve(all_gt_flat, all_prob_flat)
    pr_auc = auc(recall, precision)
    # choose best/worst images by IoU
    per_image_sorted = sorted(per_image, key=lambda x: x["iou"])
    worst = per_image_sorted[:5]
    best = per_image_sorted[-5:]
    # save examples
    for i,r in enumerate(worst+best):
        name = ("worst%d"%i) if i < len(worst) else ("best%d"%(i-len(worst)))
        save_example(r["image"], r["mask"], np.array(all_prob[per_image.index(r)]) if r in per_image else np.zeros((512,512)), out_dir, name, thr=thr)
    # write markdown report
    md = []
    md.append("# Crack500 Validation Report")
    md.append(f"Generated: {time.ctime()}")
    md.append("## Summary metrics (threshold = %.3f)"%thr)
    # compute overall binary at thr
    pred_bin_flat = (all_prob_flat>thr).astype(int)
    try:
        overall_iou = jaccard_score(all_gt_flat, pred_bin_flat, zero_division=1)
        overall_prec = precision_score(all_gt_flat, pred_bin_flat, zero_division=1)
        overall_rec = recall_score(all_gt_flat, pred_bin_flat, zero_division=1)
    except Exception as e:
        overall_iou = overall_prec = overall_rec = 0.0
    md.append(f"- Images evaluated: {len(per_image)}")
    md.append(f"- Pixel IoU: **{overall_iou:.4f}**")
    md.append(f"- Pixel Precision: **{overall_prec:.4f}**")
    md.append(f"- Pixel Recall: **{overall_rec:.4f}**")
    md.append(f"- PR AUC: **{pr_auc:.4f}**")
    md.append("## PR curve")
    # save PR figure
    plt.figure(figsize=(6,4)); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve"); plt.grid(True)
    plt.savefig(out_dir/"pr_curve.png", bbox_inches="tight"); plt.close()
    md.append("![PR curve](pr_curve.png)")
    md.append("## Example worst images (low IoU)")
    for i,r in enumerate(worst):
        name = f"worst{i}"
        md.append(f"### {r['image']}  IoU={r['iou']:.4f}  P={r['prec']:.4f}  R={r['rec']:.4f}")
        md.append(f"![{name}]({name}_overlay.png)")
    md.append("## Example best images (high IoU)")
    for i,r in enumerate(best):
        name = f"best{i}"
        md.append(f"### {r['image']}  IoU={r['iou']:.4f}  P={r['prec']:.4f}  R={r['rec']:.4f}")
        md.append(f"![{name}]({name}_overlay.png)")
    # write to file
    out_md = out_dir/"report.md"
    open(out_md,"w").write("\\n".join(md))
    print("Report written to", out_md)
    print("PR AUC:", pr_auc, "Overall IoU:", overall_iou)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="report_out")
    parser.add_argument("--thr", type=float, default=0.5)
    args = parser.parse_args()
    run_report(args.splits, args.model, args.out, thr=args.thr)
