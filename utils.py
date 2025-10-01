
import numpy as np
import torch
import cv2
from PIL import Image

def read_image_as_rgb(file):
    img = Image.open(file).convert("RGB")
    return img

def to_numpy(img: Image.Image):
    return np.array(img)

def preprocess_for_classification(img, size=224, as_numpy=False):
    if isinstance(img, Image.Image):
        arr = np.array(img)
    else:
        arr = img
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize to [-1,1]
    chw = np.transpose(arr, (2,0,1))[None, ...]
    if as_numpy:
        return chw.astype(np.float32)
    else:
        return torch.from_numpy(chw).float()

def preprocess_for_segmentation(img, size=256, as_numpy=False):
    if isinstance(img, Image.Image):
        orig = np.array(img)
    else:
        orig = img
    arr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2,0,1))[None, ...]
    if as_numpy:
        return arr.astype(np.float32), orig
    else:
        return torch.from_numpy(arr).float(), orig

def overlay_mask(image_np, mask_np, alpha=0.5):
    image = image_np.copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    color = np.zeros_like(image)
    color[..., 1] = mask_np  # green channel
    over = cv2.addWeighted(image, 1.0, color, alpha, 0)
    return over

def safe_torch_load_state_dict(model, file_obj):
    try:
        data = file_obj.read()
        state = torch.load(
            io.BytesIO(data),
            map_location="cpu",
            weights_only=False
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return True
    except Exception as e:
        # print("load error:", e)
        return False

def canny_baseline(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Morphology to connect thin cracks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    overlay = overlay_mask(image_np, closed)
    return closed, overlay
