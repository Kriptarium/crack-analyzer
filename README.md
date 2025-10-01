
# Crack Analyzer (Streamlit demo with post-processing)

This repository contains a minimal Streamlit demo app that runs a classical crack-proposal detector
(Canny-based) and then applies post-processing to remove small/noisy detections.

It is intended to be uploaded to your GitHub repo (or replace files) and then deployed to Render (or run locally).

## Files
- `app.py` - Streamlit app. Upload an image, runs detection + postprocess, shows overlay.
- `postprocess.py` - Post-processing utilities (component filtering, skeleton-based filtering).
- `requirements.txt` - Minimal requirements for Render.
- `.gitignore` - Ignore large files / model weights.

## How to deploy (GitHub -> Render)
1. Upload these files to your GitHub repository (e.g. via the web interface: "Add file" -> "Upload files"), commit to `main`.
2. On Render: Create a new **Web Service**, connect GitHub, choose this repo.
   - Build command: `pip install -r requirements.txt`
   - Start command:
     ```
     streamlit run app.py --server.port $PORT --server.headless true
     ```
3. Render will build and deploy. Check logs if there are errors.

## Notes
- This demo **does not** include a trained deep learning segmentation model. If you want to use a model (e.g. `best_resnet18.pth`), do **not** upload it to GitHub; instead store it externally and modify `app.py` to load it (I can help).
- The post-processing step is the main improvement: it reduces false positives from small texture details.

If you want, I can:
- Zip these files so you can upload them through GitHub web UI.
- Or push them directly to your repo if you grant steps (I will instead provide instructions).

