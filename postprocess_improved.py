
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def remove_small_components(mask, min_area=500):
    lbl = label(mask>0)
    out = np.zeros_like(mask)
    for r in regionprops(lbl):
        if r.area >= min_area:
            out[lbl==r.label] = 255
    return out

def prune_skeleton_spurs(mask, iterations=8):
    binm = (mask>0).astype('uint8')
    sk = skeletonize(binm).astype('uint8')
    K = np.ones((3,3), dtype=np.uint8); K[1,1]=0
    sk_work = sk.copy()
    for _ in range(iterations):
        nb = cv2.filter2D(sk_work.astype(np.uint8), -1, K, borderType=cv2.BORDER_CONSTANT)
        endpoints = ((sk_work==1) & (nb==1)).astype(np.uint8)
        if endpoints.sum() == 0:
            break
        sk_work[endpoints==1] = 0
    sk_dil = cv2.dilate(sk_work.astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    return (sk_dil*255).astype('uint8')

def filter_by_skeleton_length(mask, min_skel_len=40):
    out = np.zeros_like(mask)
    lbl = label(mask>0)
    for r in regionprops(lbl):
        comp = (lbl==r.label).astype('uint8')
        sk = skeletonize(comp).astype('uint8')
        sklen = int(sk.sum())
        if sklen >= min_skel_len:
            out[comp==1] = 255
    return out

def filter_by_elongation(mask, min_elongation=3.0):
    out = np.zeros_like(mask)
    lbl = label(mask>0)
    for r in regionprops(lbl):
        maj = getattr(r, "major_axis_length", 0.0) or 0.0
        minr = getattr(r, "minor_axis_length", 0.0) or 0.0
        if minr < 1e-6:
            ratio = 999.0
        else:
            ratio = maj / (minr + 1e-6)
        if ratio >= min_elongation:
            out[lbl==r.label] = 255
    return out

def improved_postprocess_from_probs(probs,
                                   thresh=0.65,
                                   gaussian_ksize=5,
                                   min_area=500,
                                   min_skel_len=60,
                                   min_elongation=3.0,
                                   spur_prune_iters=8,
                                   closing_k=3,
                                   opening_k=3):
    if gaussian_ksize and gaussian_ksize>0:
        probs_blur = cv2.GaussianBlur((probs*255).astype('uint8'), (gaussian_ksize,gaussian_ksize), 1.0).astype('float32')/255.0
    else:
        probs_blur = probs
    mask = (probs_blur > thresh).astype('uint8')*255
    if opening_k and opening_k>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_k,opening_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = remove_small_components(mask, min_area=min_area)
    if spur_prune_iters and spur_prune_iters>0:
        mask_spur = prune_skeleton_spurs(mask, iterations=spur_prune_iters)
    else:
        mask_spur = mask
    mask_len = filter_by_skeleton_length(mask_spur, min_skel_len=min_skel_len)
    mask_el = filter_by_elongation(mask_len, min_elongation=min_elongation)
    if closing_k and closing_k>0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_k,closing_k))
        mask_el = cv2.morphologyEx(mask_el, cv2.MORPH_CLOSE, k2)
    return mask_el
