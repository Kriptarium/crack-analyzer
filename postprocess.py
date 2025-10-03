
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from PIL import Image

def remove_small_components(mask, min_area=500):
    if mask.max() > 1:
        mask = (mask > 127).astype('uint8')
    lbl = label(mask)
    out = np.zeros_like(mask)
    for r in regionprops(lbl):
        if r.area >= min_area:
            out[lbl == r.label] = 1
    return (out * 255).astype('uint8')

def remove_thin_components_by_skeleton(mask, min_length=40, min_width=3):
    if mask.max() > 1:
        binm = (mask > 127).astype('uint8')
    else:
        binm = mask.copy()
    lbl = label(binm)
    out = np.zeros_like(binm)
    for r in regionprops(lbl):
        comp = (lbl == r.label).astype('uint8')
        sk = skeletonize(comp).astype('uint8')
        sklen = int(sk.sum())
        width = (comp.sum() / (sklen + 1e-6))
        if sklen >= min_length and width >= min_width:
            out[comp == 1] = 1
    return (out * 255).astype('uint8')

def postprocess_mask_from_probs(probs, thresh=0.65, gaussian_ksize=5, min_area=500, min_len=40, min_w=3):
    # probs: numpy float [0..1], single-channel
    if gaussian_ksize > 0:
        probs = cv2.GaussianBlur((probs * 255).astype('uint8'), (gaussian_ksize, gaussian_ksize), 1.0).astype('float32') / 255.0
    mask = (probs > thresh).astype('uint8') * 255
    mask = remove_small_components(mask, min_area=min_area)
    mask = remove_thin_components_by_skeleton(mask, min_length=min_len, min_width=min_w)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def postprocess_mask_from_imagepath(mask_path, out_path=None, **kwargs):
    m = np.array(Image.open(mask_path).convert('L'))
    m_pp = postprocess_mask_from_probs(m / 255.0, **kwargs)
    out_path = out_path or mask_path.replace('.png', '_pp.png')
    Image.fromarray(m_pp).save(out_path)
    return out_path
