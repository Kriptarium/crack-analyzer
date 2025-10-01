Crack Analyzer — GitHub+Render friendly package with report generation

Contents:
- app.py : Streamlit demo web app (loads model if MODEL_URL env var is set)
- infer.py : CLI single-image inference
- generate_report.py : Runs inference on test split and writes a Markdown report with metrics and example images
- prepare_splits_from_txt.py : Create splits.json from train/val/test txt lists and images/masks folders
- postprocess_improved.py : postprocessing utilities (skeleton prune, elongation, length filters)
- requirements.txt : Python dependencies

How to use (quick):
1) Download/unzip this repo and upload files to your GitHub repo root (or add files via web UI).
2) If you have train/val/test lists (train.txt etc) and folders images/ masks/ use:
   python3 prepare_splits_from_txt.py --root /path/to/dataset --train train.txt --val val.txt --test test.txt --out splits.json
3) Place your trained model (best_crack500.pth) on a public URL (Google Drive/S3/GitHub Release) or push it to repo (not recommended).
4) On Render: set Environment variable MODEL_URL to model download URL (if you don't upload model to repo).
   Start command:
   bash -lc "if [ ! -f ./best_crack500.pth ] && [ ! -z \"$MODEL_URL\" ]; then curl -L \"$MODEL_URL\" -o best_crack500.pth; fi && streamlit run app.py --server.port \$PORT --server.headless true"
5) To generate validation report locally (recommended):
   python3 -m venv venv; source venv/bin/activate
   pip install -r requirements.txt
   python3 generate_report.py --splits splits.json --model best_crack500.pth --out report_out --thr 0.5
   The script writes report_out/report.md and images (PR curve, overlays) for reviewers.

Notes on evaluation & report:
- The report computes pixel-level IoU, precision, recall, PR AUC and saves top/bottom example images (overlay, prob map, mask).
- For reviewer reproducibility: include splits.json and a small script to reproduce predictions (infer.py).
- If you want a PDF report, open report.md and convert to PDF (pandoc or GitHub render).

If you want, I can also:
- Prepare a small Colab that fine-tunes a model on Crack500 and uploads the resulting .pth to Google Drive for easy Render deployment.
- Or prepare a PR diff to your existing repo that replaces app.py and adds generate_report.py so you just merge and deploy.

Questions? Say "Hazırım" and hangi modeli kullanmak istediğini (var olan best_resnet18.pth veya yeni model) — ben sana adım adım devamını vereyim.
