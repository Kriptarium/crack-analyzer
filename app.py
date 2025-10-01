import os, io, time, sys
import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from postprocess_improved import improved_postprocess_from_probs
import tempfile

st.set_page_config(page_title="Crack Analyzer (Model Demo + Report)", layout="wide")
st.title("Crack Analyzer — Demo + Report generator")

st.sidebar.header("Model settings")
MODEL_PATH = "best_crack500.pth"
MODEL_URL = os.getenv("MODEL_URL","").strip()
device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
st.sidebar.write("Device: %s" % device)

@st.cache_resource(show_spinner=False)
def download_model_if_needed(url, dst="best_crack500.pth"):
    if os.path.exists(dst):
        return dst
    if not url:
        return None
    import subprocess
    st.sidebar.write("Downloading model...")
    try:
        subprocess.check_call(["curl","-L",url,"-o",dst])
        return dst
    except Exception as e:
        st.sidebar.error("Model download failed: %s" % str(e))
        return None

@st.cache_resource
def load_model(path):
    if path is None or not os.path.exists(path):
        return None
    model = models.segmentation.deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256,1,kernel_size=1)
    model.aux_classifier = None
    ck = torch.load(path, map_location=device)
    sd = ck.get("model_state", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(device).eval()
    return model

model_file = download_model_if_needed(MODEL_URL, MODEL_PATH) if MODEL_URL else (MODEL_PATH if os.path.exists(MODEL_PATH) else None)
if model_file:
    st.sidebar.success("Model available: %s" % os.path.basename(model_file))
else:
    st.sidebar.warning("No model found. Set MODEL_URL env var or upload best_crack500.pth to repo root. App will run classical detector fallback.")

model = load_model(model_file) if model_file else None

st.sidebar.header("Postprocess sliders (for demo)")
thresh = st.sidebar.slider("Prob threshold", 0.5, 0.95, 0.65, 0.01)
min_area = st.sidebar.slider("Min area (px)", 50, 5000, 700, 50)
min_skel = st.sidebar.slider("Min skeleton length", 10, 300, 60, 5)
min_elong = st.sidebar.slider("Min elongation", 1.0, 10.0, 3.0, 0.1)
spur_iters = st.sidebar.slider("Prune spur iters", 0, 30, 8, 1)

st.markdown("Upload an image to run model inference. If model isn't available, app runs a classical edge-based detector as fallback.")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
def classical_detector(img_pil):
    import cv2
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

def overlay_mask(img_pil, mask, color=(255,0,0), alpha=0.6):
    img = np.array(img_pil.convert("RGB")).astype("uint8")
    mask3 = np.zeros_like(img)
    mask_bool = (mask>127)
    mask3[mask_bool] = color
    out = cv2.addWeighted(img,1.0,mask3,alpha,0)
    return Image.fromarray(out)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.subheader("Original")
    st.image(img, use_column_width=True)
    if model is not None:
        prob_map = infer_with_model(img)
    else:
        prob_map = classical_detector(img)
    st.subheader("Probability / edge map")
    st.image((prob_map*255).astype("uint8"), width=420)
    mask = improved_postprocess_from_probs(prob_map, thresh=thresh, gaussian_ksize=5,
                                          min_area=min_area, min_skel_len=min_skel,
                                          min_elongation=min_elong, spur_prune_iters=spur_iters)
    st.subheader("Post-processed mask")
    st.image(mask, width=420)
    st.subheader("Overlay")
    # overlay using cv2 (import lazily)
    import cv2
    ov = overlay_mask(img, mask)
    st.image(ov, use_column_width=True)

st.markdown("---")
st.markdown("## Report generation (offline)")
st.markdown("To generate a full validation & performance report on a test split, run the included `generate_report.py` locally (or on a server). The report will compute IoU, Precision/Recall, save example FP/FN images and produce a Markdown report.")
st.markdown("See README for exact commands.")
