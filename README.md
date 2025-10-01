
Crack500 Explorer UI - README

Files:
- app_crack500.py : Streamlit app for exploring Crack500 images, running model inference, and batch evaluation on test split.
- prepare_splits_from_txt.py : convert train/val/test txt lists into splits.json expected by the app.
- postprocess_improved.py : postprocessing utilities used by the app.
- requirements.txt : dependencies

How to use locally:
1. Put your Crack500 dataset under a folder with structure:
   <dataset_root>/images/*.jpg
   <dataset_root>/masks/*.png
   and optionally train.txt / val.txt / test.txt listing file names (one per line).

2. Create venv, install deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # if torch install fails on Mac MPS, follow PyTorch mac instructions or use CPU wheel

3. If you have lists, create splits.json (optional):
   python3 prepare_splits_from_txt.py --root /path/to/dataset --out splits.json

4. Run Streamlit app:
   streamlit run app_crack500.py
   Enter dataset root in sidebar or leave default ~/datasets/crack500

5. To deploy on Render:
   - Upload files to GitHub repo, set Render build command: pip install -r requirements.txt
   - Set Render env var MODEL_URL if you host best_crack500.pth somewhere public.
   - Start command (Render): bash -lc "if [ ! -f ./best_crack500.pth ] && [ ! -z \"$MODEL_URL\" ]; then curl -L \"$MODEL_URL\" -o best_crack500.pth; fi && streamlit run app_crack500.py --server.port \$PORT --server.headless true"

Notes:
- The app will use the model if best_crack500.pth exists in repo root or if MODEL_URL is provided; otherwise it falls back to a classical edge-based detector for demo.
- Batch evaluation computes a simple pixel IoU if ground-truth masks are present; it's not a full scientific evaluation but a quick check for reviewers.
