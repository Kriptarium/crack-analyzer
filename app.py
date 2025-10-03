
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from postprocess import postprocess_mask_from_probs

st.set_page_config(page_title="Crack Analyzer (with postprocessing)", layout="wide")

st.title("Crack Analyzer — Demo (postprocessing enabled)")

st.markdown("""
This demo applies a classical crack-proposal detector (Canny-based) and then a post-processing step
to remove tiny/noisy detections. If you have a custom segmentation model file named `best_resnet18.pth`
in the repo, the app can be extended to load it (not included by default).
""")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

def classical_detector(img_pil):
    # returns probability-like map (float 0..1)
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # CLAHE to normalize illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    # slight blur
    g = cv2.GaussianBlur(g, (3,3), 0)
    # Canny edges
    edges = cv2.Canny(g, 50, 150)
    # Dilate to make edges thicker
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.dilate(edges, k, iterations=1)
    # Normalize to 0..1 float map
    prob = edges.astype("float32")/255.0
    # Smooth a bit
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
    st.image(img, caption="Uploaded image", use_column_width=True)
    with st.spinner("Running classical detector..."):
        prob_map = classical_detector(img)
    st.write("Probability / edge map (classical detector)")
    # show prob map
    st.image((prob_map*255).astype("uint8"), width=350)
    st.write("Applying post-processing to remove small/noisy detections...")
    # postprocess
    pp_mask = postprocess_mask_from_probs(prob_map, thresh=0.65, gaussian_ksize=5, min_area=500, min_len=40, min_w=3)
    st.write("Post-processed mask (binary)")
    st.image(pp_mask, width=350)
    st.write("Overlayed result")
    st.image(overlay_mask(img, pp_mask), use_column_width=True)
    st.success("Done — tweak parameters in postprocess.py if you want different sensitivity.")
else:
    st.info("Upload an image to run detection.")
