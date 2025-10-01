
import os, io, json, time
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, models
from postprocess_improved import improved_postprocess_from_probs
import cv2

st.set_page_config(page_title="Crack500 Explorer & Demo", layout="wide")
st.title("Crack500 Explorer — Model demo + evaluation")

# Sidebar: dataset and model config
st.sidebar.header("Dataset & Model")
dataset_root = st.sidebar.text_input("Dataset root (folder with images/ and masks/ and train/val/test txt)", value=str(Path.home()/ "datasets" / "crack500"))
use_splits = st.sidebar.checkbox("Use splits.json (if present)", value=True)
model_url = st.sidebar.text_input("MODEL_URL (optional public URL to download best_crack500.pth)", value=os.getenv("MODEL_URL",""))
download_model = st.sidebar.button("Download model now")
device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
st.sidebar.write("Device: %s" % device)

@st.cache_resource(show_spinner=False)
def maybe_download_model(url, dst="best_crack500.pth"):
    if not url:
        return None
    if Path(dst).exists():
        return dst
    try:
        import subprocess
        subprocess.check_call(["curl","-L",url,"-o",dst])
        return dst
    except Exception as e:
        st.sidebar.error(f"Model download failed: {e}")
        return None

@st.cache_resource
def load_model(path):
    if path is None or not Path(path).exists():
        return None
    model = models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256,1,kernel_size=1)
    model.aux_classifier=None
    ck = torch.load(path, map_location=device)
    sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

if download_model and model_url:
    model_file = maybe_download_model(model_url)
else:
    model_file = "best_crack500.pth" if Path("best_crack500.pth").exists() else (maybe_download_model(model_url) if model_url else None)

model = load_model(model_file) if model_file else None
if model_file:
    st.sidebar.success(f"Model ready: {model_file}")
else:
    st.sidebar.info("No model available — will use classical detector fallback for demo.")

# Postprocess controls
st.sidebar.header("Postprocess")
thresh = st.sidebar.slider("Prob threshold", 0.5, 0.95, 0.65, 0.01)
min_area = st.sidebar.slider("Min area px", 50, 5000, 700, 50)
min_skel = st.sidebar.slider("Min skeleton length", 10, 300, 60, 5)
min_elong = st.sidebar.slider("Min elongation", 1.0, 10.0, 3.0, 0.1)
spur_iters = st.sidebar.slider("Prune spur iters", 0, 30, 8, 1)

# Utilities
def classical_detector(img_pil):
    arr = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5,5), 0)
    edges = cv2.Canny(g, 50, 150)
    prob = edges.astype("float32")/255.0
    prob = cv2.GaussianBlur(prob, (5,5), 1.0)
    return prob

def infer_with_model(img_pil):
    tf = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    x = tf(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)['out']
        prob = torch.sigmoid(out).cpu().numpy()[0,0]
    return prob

def overlay(img_pil, mask):
    img = np.array(img_pil.convert("RGB").resize((mask.shape[1], mask.shape[0])))
    mask3 = np.zeros_like(img)
    mask3[mask>127] = [255,0,0]
    out = cv2.addWeighted(img, 1.0, mask3, 0.6, 0)
    return Image.fromarray(out)

# Load dataset listings
rootp = Path(dataset_root)
images_dir = rootp / "images"
masks_dir = rootp / "masks"
splits_file = rootp / "splits.json"
use_txt_lists = False
if use_splits and splits_file.exists():
    splits = json.load(open(splits_file))
    test_list = splits.get("test", [])
    image_paths = [p["image"] for p in test_list]
else:
    # try train/val/test txt
    for f in ["test.txt","test","testlist.txt"]:
        p = rootp / f
        if p.exists():
            use_txt_lists = True
            lines = [l.strip() for l in open(p) if l.strip()]
            image_paths = [str(images_dir / l) if (images_dir / l).exists() else l for l in lines]
            break
    else:
        # fallback: list images dir
        image_paths = sorted([str(p) for p in images_dir.glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]])

if not image_paths:
    st.error("No images found. Check dataset_root and ensure images/ exists.")
    st.stop()

st.write(f"Found {len(image_paths)} images (using dataset root: {dataset_root})")

# Main layout: selector + display
col1, col2 = st.columns([1,2])
with col1:
    st.header("Image list (test)")
    idx = st.number_input("Index", min_value=0, max_value=len(image_paths)-1, value=0, step=1)
    chosen = st.selectbox("Pick image", options=range(len(image_paths)), format_func=lambda i: Path(image_paths[i]).name, index=0)
    st.write(Path(image_paths[chosen]).name)
    if st.button("Run evaluation on test set (batch, may be slow)"):
        st.session_state["do_batch"] = True

with col2:
    img_path = Path(image_paths[chosen])
    st.subheader("Original image")
    img = Image.open(img_path).convert("RGB")
    st.image(img, use_column_width=True)

    # show GT mask if exists
    mask_path = masks_dir / (img_path.stem + ".png")
    if not mask_path.exists():
        # try common alternatives
        for ext in [".jpg",".jpeg"]:
            if (masks_dir / (img_path.stem + ext)).exists():
                mask_path = masks_dir / (img_path.stem + ext)
                break
    if mask_path.exists():
        gt = Image.open(mask_path).convert("L")
        st.subheader("Ground truth mask")
        st.image(gt, use_column_width=True)
    else:
        st.info("No ground-truth mask found for this image (looking for masks/<stem>.png)")

    # Run inference for selected image
    if st.button("Run inference on selected image"):
        with st.spinner("Running inference..."):
            if model is not None:
                prob = infer_with_model(img)
            else:
                prob = classical_detector(img)
            st.subheader("Probability map")
            st.image((prob*255).astype("uint8"), width=420)
            mask_pp = improved_postprocess_from_probs(prob, thresh=thresh, gaussian_ksize=5,
                                                      min_area=min_area, min_skel_len=min_skel,
                                                      min_elongation=min_elong, spur_prune_iters=spur_iters)
            st.subheader("Postprocessed mask")
            st.image(mask_pp, width=420)
            st.subheader("Overlay")
            st.image(overlay(img, mask_pp), use_column_width=True)

# Batch evaluation
if st.session_state.get("do_batch", False):
    st.info("Running batch inference on test set — this may take time.")
    results = []
    device0 = device
    for i, ip in enumerate(image_paths):
        if i % 10 == 0:
            st.write(f"Processing {i}/{len(image_paths)}")
        p = Path(ip)
        try:
            pil = Image.open(p).convert("RGB")
        except Exception as e:
            continue
        if model is not None:
            prob = infer_with_model(pil)
        else:
            prob = classical_detector(pil)
        mask_pp = improved_postprocess_from_probs(prob, thresh=thresh, gaussian_ksize=5,
                                                  min_area=min_area, min_skel_len=min_skel,
                                                  min_elongation=min_elong, spur_prune_iters=spur_iters)
        # compute simple IoU if gt exists
        gtpath = masks_dir / (p.stem + ".png")
        if gtpath.exists():
            gt = np.array(Image.open(gtpath).convert("L").resize(mask_pp.shape[::-1]))
            pred_bin = (mask_pp>127).astype(int)
            gt_bin = (gt>127).astype(int)
            inter = (pred_bin & gt_bin).sum()
            union = (pred_bin | gt_bin).sum()
            iou = inter/union if union>0 else 1.0
        else:
            iou = None
        results.append({"image":str(p), "iou": iou})
    # show summary
    ious = [r["iou"] for r in results if r["iou"] is not None]
    if ious:
        st.success(f"Batch done. Mean IoU on available GT: {np.mean(ious):.4f} (N={len(ious)})")
    else:
        st.warning("Batch done. No GT masks found to compute IoU.")
    st.session_state["do_batch"] = False
