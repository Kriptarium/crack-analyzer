#!/usr/bin/env python3
import sys, os, torch, numpy as np
from PIL import Image
from torchvision import transforms, models
from postprocess_improved import improved_postprocess_from_probs

def load_model(pth, device):
    model = models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    model.aux_classifier = None
    ck = torch.load(pth, map_location=device)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

def run(model, img_path, out_prob="prob.png", out_mask="mask.png", device="cpu", post_thresh=0.6):
    im = Image.open(img_path).convert("RGB")
    tf = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    x = tf(im).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)["out"]
        prob = torch.sigmoid(out).cpu().numpy()[0,0]
    Image.fromarray((prob*255).astype("uint8")).save(out_prob)
    mask_pp = improved_postprocess_from_probs(prob, thresh=post_thresh, gaussian_ksize=5, min_area=500, min_skel_len=60, min_elongation=3.0)
    Image.fromarray(mask_pp).save(out_mask)
    print("Saved", out_prob, out_mask)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: infer.py model.pth image.jpg [out_prob out_mask]")
        sys.exit(1)
    pth = sys.argv[1]; img = sys.argv[2]
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = load_model(pth, device)
    run(model, img, sys.argv[3] if len(sys.argv)>3 else "prob.png", sys.argv[4] if len(sys.argv)>4 else "mask.png", device=device, post_thresh=0.6)
