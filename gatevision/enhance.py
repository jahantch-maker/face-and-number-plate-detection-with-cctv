"""Standard (non-AI) picture enhancement for the saved face and plate photos.

Only classical, honest operations: enlarge small pictures, remove noise, fix
brightness, lift local contrast and sharpen. Nothing is invented - no detail
is added that the camera did not capture - so an enhanced picture can never
show a different person or different plate characters than the original.
"""
from __future__ import annotations

import cv2
import numpy as np


def _brightness_fix(bgr, target=118.0):
    """Gamma-correct a too dark / too bright picture towards a comfortable mean."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32)
    mean = float(lum.mean()) + 1e-6
    if 85.0 <= mean <= 150.0:
        return bgr
    gamma = float(np.clip(np.log(target / 255.0) / np.log(mean / 255.0), 0.45, 1.8))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _contrast(bgr, clip=2.0, grid=4):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _unsharp(bgr, amount=0.9, sigma=1.3):
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _enlarge(bgr, min_width, max_scale=3.0):
    h, w = bgr.shape[:2]
    if w >= min_width:
        return bgr
    scale = min(max_scale, min_width / float(w))
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)


def enhance_face(bgr):
    """Clearer face: bigger, less noisy, better lit, crisper. Returns a new picture."""
    if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < 12:
        return bgr
    try:
        out = _enlarge(bgr, 240)
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 5, 15)
        out = _brightness_fix(out)
        out = _contrast(out, clip=2.0, grid=4)
        out = _unsharp(out, amount=0.7, sigma=1.3)
        return np.clip(out, 0, 255).astype(np.uint8)
    except cv2.error:
        return bgr


def enhance_plate(bgr):
    """Easier-to-read plate photo: bigger, balanced brightness, stronger contrast, crisp edges."""
    if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < 8:
        return bgr
    try:
        out = _enlarge(bgr, 320)
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 5, 15)
        out = _brightness_fix(out)
        out = _contrast(out, clip=2.5, grid=4)
        out = _unsharp(out, amount=0.8, sigma=1.1)
        return np.clip(out, 0, 255).astype(np.uint8)
    except cv2.error:
        return bgr
