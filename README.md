
Updated Crack Analyzer repository (improved postprocessing)

Files:
- app.py : Streamlit app (with sidebar sliders to tune postprocessing)
- postprocess_improved.py : improved postprocessing functions (skeleton prune, elongation filter, length filter)
- postprocess.py : compatibility shim
- requirements.txt : minimal requirements

How to use:
1. Download/unzip the files and upload them to your GitHub repo (either replace existing files or add new ones).
2. Push to GitHub; Render will auto-deploy.
3. On the live app, use the sidebar sliders to tune threshold, area, skeleton length, etc. to reduce false positives.

Note: This demo uses a classical edge detector (Canny) as the "probability" source. If you later want to load your segmentation model weights, do not upload the .pth to GitHub; instead use a MODEL_URL environment variable and modify the start command to download the model on deploy.
