
import streamlit as st
import numpy as np
import cv2
from skimage.morphology import skeletonize
from PIL import Image
import io

st.set_page_config(page_title="Crack Analyzer - Public", layout="wide")

st.title("🔎 Crack Analyzer — Public (Render-ready)")
st.markdown("""
This app is prepared for Render deployment.

**Start command (Render):**
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
""")

col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
    px2mm = st.number_input("Scale (mm/pixel)", value=0.1, format="%.4f")
with col2:
    st.write("Tips:")
    st.write("- Take images perpendicular to the surface and include a ruler if possible.")
    st.write("- Avoid strong reflections; they can confuse detection.")

def classical_crack_mask(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.GaussianBlur(g, (5,5), 0)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    edges = cv2.Canny(tophat, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.erode(edges, None, iterations=1)
    return edges

if uploaded is not None:
    data = uploaded.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read image. Try another file.")
    else:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        mask = classical_crack_mask(img)
        skel = skeletonize(mask>0)
        px_length = int(skel.sum())
        length_mm = px_length * px2mm
        st.write(f"Total crack length (approx): **{length_mm:.2f} mm**")
        overlay = img.copy()
        overlay[skel] = (0,0,255)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Cracks (red)", use_column_width=True)
        # download mask
        mask_png = (mask>0).astype('uint8')*255
        success, buf = cv2.imencode('.png', mask_png)
        if success:
            st.download_button("Download mask (PNG)", data=buf.tobytes(), file_name="mask.png", mime="image/png")
        else:
            st.error("Mask creation failed.")
