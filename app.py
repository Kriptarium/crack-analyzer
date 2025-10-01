
import streamlit as st
import numpy as np
from PIL import Image
import io, os, torch, onnx, onnxruntime as ort
import cv2
from utils import (
    read_image_as_rgb, 
    preprocess_for_classification, 
    preprocess_for_segmentation, 
    overlay_mask, 
    safe_torch_load_state_dict,
    canny_baseline
)
from models import SimpleCrackClassifier, UNetLite

st.set_page_config(page_title="Crack Analyzer – File Runner", page_icon="🧱", layout="wide")

st.title("🧱 Crack Analyzer – File Runner")
st.caption("Tek dosya çalıştırma arayüzü: model yükle, görsel seç, çalıştır. Kaggle Crack500 veya benzeri çatlak görüntüleri için.")

with st.sidebar:
    st.header("Ayarlar")
    mode = st.selectbox("Çalışma modu", ["Baseline (Canny)", "PyTorch: Sınıflandırma", "PyTorch: Segmentasyon", "ONNX: Sınıflandırma", "ONNX: Segmentasyon"])
    conf_threshold = st.slider("Mask eşik değeri (seg.)", 0.1, 0.9, 0.5, 0.05)
    show_overlay = st.checkbox("Maskeyi görüntü üzerine bindir", True)
    show_heatmap = st.checkbox("Isı haritası (seg.)", False)

st.write("### 1) Görsel yükle")
img_file = st.file_uploader("Görsel dosyası yükleyin (JPG/PNG).", type=["jpg","jpeg","png"])

st.write("### 2) Model (isteğe bağlı) yükle")
model_file = st.file_uploader("Model dosyası yükleyin (.pth veya .onnx). Seçili moda uygun olmalı.", type=["pth","onnx"])

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Girdi")
    if img_file:
        image = read_image_as_rgb(img_file)
        st.image(image, caption="Yüklenen görüntü", use_column_width=True)
    else:
        st.info("Bir görüntü seçin. Eğer seçmezseniz demo için örnek bir taş dokusu kullanılacak.")
        demo = Image.new("RGB", (512, 512), (210, 210, 210))
        st.image(demo, caption="Demo görüntü", use_column_width=True)
        image = demo

run = st.button("▶️ Çalıştır")

with col2:
    st.subheader("Çıktı")

    if run:
        if mode == "Baseline (Canny)":
            mask, vis = canny_baseline(np.array(image))
            st.write("**Basit Canny tabanlı çatlak çıkarımı**")
            st.image(vis, caption="Canny + morfoloji sonucu", use_column_width=True)

        elif mode == "PyTorch: Sınıflandırma":
            # Basit CNN ve .pth state_dict
            model = SimpleCrackClassifier(num_classes=2)
            if model_file and model_file.name.endswith(".pth"):
                loaded = safe_torch_load_state_dict(model, model_file)
                if not loaded:
                    st.warning("Model state_dict yüklenemedi. Random ağırlıklarla devam ediliyor.")
            else:
                st.info(".pth dosyası yüklemediğiniz için model rastgele ağırlıklarla çalışacak (sadece demo).")

            model.eval()
            x = preprocess_for_classification(image)  # (1,3,H,W) tensor
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred = int(probs.argmax())
            labels = ["No Crack", "Crack"]
            st.metric("Tahmin", labels[pred], delta=f"{probs[pred]*100:.1f}% güven")

        elif mode == "PyTorch: Segmentasyon":
            net = UNetLite(in_ch=3, out_ch=1)
            if model_file and model_file.name.endswith(".pth"):
                loaded = safe_torch_load_state_dict(net, model_file)
                if not loaded:
                    st.warning("UNet state_dict yüklenemedi. Random ağırlıklarla devam ediliyor (sadece demo).")
            else:
                st.info(".pth dosyası yüklenmedi; ağ rastgele ağırlıklarla (sadece demo).")

            net.eval()
            x, orig = preprocess_for_segmentation(image) # torch tensor, original np
            with torch.no_grad():
                pred = net(x)  # (1,1,h,w)
                prob = torch.sigmoid(pred).cpu().numpy()[0,0]
            mask = (prob >= conf_threshold).astype(np.uint8)*255
            if show_heatmap:
                heat = (prob * 255).astype(np.uint8)
                st.image(heat, caption="Olasılık haritası (0-255)", use_column_width=True)
            if show_overlay:
                over = overlay_mask(orig, mask)
                st.image(over, caption="Mask Overlay", use_column_width=True)
            else:
                st.image(mask, caption="İkili Maske", use_column_width=True)

        elif mode == "ONNX: Sınıflandırma":
            if not (model_file and model_file.name.endswith(".onnx")):
                st.error("Lütfen ONNX sınıflandırma modeli yükleyin (.onnx).")
            else:
                ort_sess = ort.InferenceSession(model_file.getvalue(), providers=['CPUExecutionProvider'])
                x = preprocess_for_classification(image, as_numpy=True)  # np array (1,3,H,W)
                inp = ort_sess.get_inputs()[0].name
                out = ort_sess.get_outputs()[0].name
                logits = ort_sess.run([out], {inp: x})[0]  # (1,2)
                probs = (logits - logits.max()).astype(np.float32)
                probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
                pred = int(probs.argmax())
                labels = ["No Crack", "Crack"]
                st.metric("Tahmin", labels[pred], delta=f"{probs[0,pred]*100:.1f}% güven")

        elif mode == "ONNX: Segmentasyon":
            if not (model_file and model_file.name.endswith(".onnx")):
                st.error("Lütfen ONNX segmentasyon modeli yükleyin (.onnx).")
            else:
                ort_sess = ort.InferenceSession(model_file.getvalue(), providers=['CPUExecutionProvider'])
                x, orig = preprocess_for_segmentation(image, as_numpy=True)  # np (1,3,h,w), orig np
                inp = ort_sess.get_inputs()[0].name
                out = ort_sess.get_outputs()[0].name
                logits = ort_sess.run([out], {inp: x})[0]  # (1,1,h,w)
                prob = 1/(1+np.exp(-logits[0,0]))
                mask = (prob >= conf_threshold).astype(np.uint8)*255
                if show_heatmap:
                    heat = (prob * 255).astype(np.uint8)
                    st.image(heat, caption="Olasılık haritası (0-255)", use_column_width=True)
                if show_overlay:
                    over = overlay_mask(orig, mask)
                    st.image(over, caption="Mask Overlay", use_column_width=True)
                else:
                    st.image(mask, caption="İkili Maske", use_column_width=True)

    else:
        st.info("Sol taraftan modu seçin, görüntünüzü ve (varsa) model dosyanızı yükleyin, sonra **Çalıştır** butonuna basın.")
