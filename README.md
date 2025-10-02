
Crack Analyzer Mini
-------------------
This package provides a lightweight Streamlit demo that:
- Accepts a ZIP with `images/` and `masks/` folders (matching filenames)
- Allows quick training of a tiny UNet on the uploaded dataset (CPU/MPS). Intended for small demo datasets (like 5-50 samples).
- Runs inference on uploaded images and shows overlays and predicted masks.
Deployment: upload files to GitHub repo and deploy to Render as a web service.
Notes: Training on Render free tier may be slow or time out. For real experiments use a GPU environment (Colab, local GPU).
