
import numpy as np, cv2
from PIL import Image
def overlay_mask_on_image(img_rgb, mask_binary, alpha=0.6):
    if mask_binary.ndim==2:
        mask = (mask_binary>127).astype('uint8')
    else:
        mask = (mask_binary[...,0]>127).astype('uint8')
    overlay = img_rgb.copy()
    mask3 = np.stack([mask]*3, axis=-1)
    color = np.zeros_like(overlay); color[...,0]=255
    out = np.where(mask3, (overlay*(1-alpha)+color*alpha).astype('uint8'), overlay)
    return out
