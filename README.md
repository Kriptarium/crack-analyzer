# Crack Analyzer – File Runner

Tek dosya çalıştırma arayüzü: **görüntü + model dosyası (.pth veya .onnx)** yükleyin, **çalıştır**a basın.

## Özellikler
- **Baseline (Canny)**: Model gerekmeden hızlı çatlak vurgulama.
- **PyTorch (Sınıflandırma / Segmentasyon)**: `.pth` state_dict dosyalarıyla çalışır.
- **ONNX (Sınıflandırma / Segmentasyon)**: Donanımsız (CPU) ONNX Runtime ile inference.
- **Görsel yükleme**: JPG/PNG.
- **Eşik / Overlay / Isı haritası** ayarları.

> Not: `.pth` dosyaları **mimarinizle uyumlu state_dict** içermelidir.
> Sınıflandırma için `SimpleCrackClassifier`, segmentasyon için `UNetLite` sağlanır.
> Farklı mimari kullandıysanız ONNX dışa aktarımı (export) önerilir.

## Kurulum
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Kullanım
1. Soldan **modu** seçin.
2. **Görüntü** dosyasını yükleyin.
3. (İsteğe bağlı) **Model** dosyasını yükleyin: `.pth` (PyTorch) veya `.onnx` (ONNX).
4. **Çalıştır** butonuna tıklayın.

## Model Hazırlama İpuçları
### PyTorch -> ONNX örneği (sınıflandırma)
```python
import torch
from models import SimpleCrackClassifier
model = SimpleCrackClassifier(num_classes=2)
model.load_state_dict(torch.load("your_classifier.pth", map_location="cpu"))
model.eval()
x = torch.randn(1,3,224,224)
torch.onnx.export(model, x, "classifier.onnx", input_names=["input"], output_names=["logits"], opset_version=17)
```

### PyTorch -> ONNX örneği (segmentasyon)
```python
import torch
from models import UNetLite
net = UNetLite(in_ch=3, out_ch=1)
net.load_state_dict(torch.load("your_unet.pth", map_location="cpu"))
net.eval()
x = torch.randn(1,3,256,256)
torch.onnx.export(net, x, "unet.onnx", input_names=["input"], output_names=["logits"], opset_version=17)
```

## Render / Hugging Face Spaces Dağıtımı
- **Render**: `Start Command` → `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`  
- **HF Spaces (Gradio/Streamlit)**: `sdk: streamlit`, `app_file: app.py`

## SSS
- **Model yok, çalışır mı?** Evet, *Baseline (Canny)* ile çalışır.
- **.pth yüklendi ama sonucu anlamsız**: Mimari uyuşmuyor olabilir. ONNX export deneyin.
- **Crack500 ile uyum**: Bu UI, **Crack500 formatındaki** tekil görüntüler için uygundur (256–512px kare önerilir).
