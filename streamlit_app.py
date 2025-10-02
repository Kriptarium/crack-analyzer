
import streamlit as st
from pathlib import Path
import tempfile, zipfile, io
from PIL import Image
import numpy as np
import torch
from train_mini import train_model, infer_on_image, prepare_records_from_folder
from utils import overlay_mask_on_image
st.set_page_config(layout="wide", page_title="Crack Analyzer Mini (Train on few samples)")
st.title("Crack Analyzer Mini — quick train & test on a few image+mask pairs")
st.markdown("Upload a ZIP with folders `images/` and `masks/` (matching filenames). App will show previews, allow quick training (CPU/MPS) and inference. Intended for small demos (<=100 images).")
uploaded = st.file_uploader("Upload dataset zip (images/ + masks/)", type=["zip"], accept_multiple_files=False)
if uploaded is not None:
    tmpd = tempfile.TemporaryDirectory()
    zpath = Path(tmpd.name) / "u.zip"
    with open(zpath, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(tmpd.name)
    except zipfile.BadZipFile:
        st.error("Not a valid zip file")
        raise SystemExit()
    root = Path(tmpd.name)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        st.error("Zip must contain images/ and masks/ at root")
    else:
        st.success(f"Found images: {len(list(images_dir.glob('*')))} masks: {len(list(masks_dir.glob('*')))}")
        # show a few previews
        cols = st.columns(3)
        for i,imgp in enumerate(sorted(images_dir.glob('*'))[:6]):
            with cols[i%3]:
                st.image(str(imgp), caption=imgp.name, use_column_width=True)
        # training controls
        epochs = st.number_input("Epochs (small recommended)", value=3, min_value=1, max_value=50, step=1)
        batch = st.number_input("Batch size", value=2, min_value=1, max_value=8, step=1)
        lr = st.number_input("Learning rate", value=1e-4, format="%.6f")
        train_btn = st.button("Train quick model on uploaded data")
        model_file = Path(tmpd.name)/"trained_mini.pth"
        if train_btn:
            st.info("Preparing records...")
            records = prepare_records_from_folder(images_dir, masks_dir)
            if len(records)==0:
                st.error("No matching image-mask pairs found (check filenames).")
            else:
                st.info(f"Starting training on {len(records)} samples. This may take time on CPU/MPS.")
                st.write("Device:", "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
                st.progress(0)
                history = train_model(records, epochs=int(epochs), batch_size=int(batch), lr=float(lr), out_path=str(model_file), progress_cb=lambda p: st.progress(p))
                st.success("Training finished. Model saved.")
                st.write("Training history (last entries):", history[-5:])
        st.markdown('---')
        st.header("Inference: test an image")
        inf_img = st.file_uploader("Upload a single image (jpg/png) for inference", type=["jpg","png","jpeg"])
        if inf_img is not None:
            img = Image.open(inf_img).convert('RGB')
            st.image(img, caption="Input image", use_column_width=True)
            st.write("Running inference...")
            model_path = model_file if model_file.exists() else None
            pred_mask = infer_on_image(np.array(img), model_path=model_path)
            ov = overlay_mask_on_image(np.array(img), pred_mask)
            st.image(ov, caption="Overlay (prediction)", use_column_width=True)
            # show mask
            st.image(pred_mask, caption="Predicted mask (binary)", use_column_width=True)
            # allow download mask
            buf = io.BytesIO()
            Image.fromarray(pred_mask).save(buf, format="PNG")
            st.download_button("Download predicted mask", data=buf.getvalue(), file_name="pred_mask.png", mime="image/png")
