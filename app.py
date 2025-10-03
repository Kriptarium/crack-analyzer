
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from postprocess_improved import improved_postprocess_from_probs

st.set_page_config(page_title="Crack Analyzer (postprocess adjustable)", layout="wide")

st.title("Crack Analyzer — Demo (improved postprocessing)")
st.markdown("Upload an image. The app runs a classical edge-based detector and then applies an improved postprocessing chain. Use the sidebar to tune sensitivity.")

# Sidebar sliders for live parameter tuning
st.sidebar.header("Postprocess parameters (tweak & test)")
thresh = st.sidebar.slider("Probability threshold", 0.5, 0.95, 0.70, 0.01)
gauss = st.sidebar.slider("Gaussian blur kernel (odd)", 1, 11, 5, 2)
min_area = st.sidebar.slider("Min component area (px)", 50, 5000, 700, 50)
min_skel_len = st.sidebar.slider("Min skeleton length (px)", 10, 300, 80, 5)
min_elongation = st.sidebar.slider("Min elongation ratio", 1.0, 10.0, 3.5, 0.1)
spur_iters = st.sidebar.slider("Prune spur iterations", 0, 30, 10, 1)
closing_k = st.sidebar.slider("Closing kernel", 1, 11, 5, 2)
opening_k = st.sidebar.slider("Opening kernel", 0, 11, 3, 1)

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

def classical_detector(img_pil):
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3,3), 0)
    edges = cv2.Canny(g, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.dilate(edges, k, iterations=1)
    prob = edges.astype("float32")/255.0
    prob = cv2.GaussianBlur(prob, (5,5), 1.0)
    return prob

def overlay_mask(img_pil, mask, alpha=0.6, color=(255,0,0)):
    img = np.array(img_pil.convert("RGB")).astype("uint8")
    mask3 = np.zeros_like(img)
    mask_bool = (mask>127)
    mask3[mask_bool] = color
    out = img.copy()
    out = cv2.addWeighted(out, 1.0, mask3, alpha, 0)
    return Image.fromarray(out)

if uploaded:
    img = Image.open(uploaded)
    st.subheader("Original image")
    st.image(img, use_column_width=True)
    with st.spinner("Running classical detector..."):
        prob_map = classical_detector(img)
    st.subheader("Probability / edge map (classical detector)")
    st.image((prob_map*255).astype("uint8"), width=400)
    st.subheader("Applying improved post-processing...")
    pp_mask = improved_postprocess_from_probs(prob_map,
                                             thresh=thresh,
                                             gaussian_ksize=gauss if gauss%2==1 else gauss+1,
                                             min_area=min_area,
                                             min_skel_len=min_skel_len,
                                             min_elongation=min_elongation,
                                             spur_prune_iters=spur_iters,
                                             closing_k=closing_k if closing_k%2==1 else closing_k+1,
                                             opening_k=opening_k if opening_k%2==1 else opening_k+1)
    st.write("Post-processed mask (binary)")
    st.image(pp_mask, width=400)
    st.write("Overlayed result")
    st.image(overlay_mask(img, pp_mask), use_column_width=True)
    st.success("Done — tweak the sliders to reduce false positives or false negatives.")
else:
    st.info("Upload an image to run detection. Use the sidebar sliders to tune postprocessing thresholds.")
