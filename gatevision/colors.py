"""Colour naming for vehicles and clothing (pure NumPy/OpenCV, no AI model)."""
from __future__ import annotations

import cv2
import numpy as np

NAMES = [
    "black", "white", "silver", "gray", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "brown", "beige",
]
_IDX = {n: i for i, n in enumerate(NAMES)}

VEHICLE_COLORS = NAMES
CLOTHING_COLORS = NAMES


def name_pixels(hsv: np.ndarray) -> np.ndarray:
    """Map an (N,3) HSV array (OpenCV ranges) to colour-name indices."""
    h = hsv[:, 0].astype(np.int16)
    s = hsv[:, 1].astype(np.int16)
    v = hsv[:, 2].astype(np.int16)
    out = np.full(len(hsv), _IDX["gray"], dtype=np.int16)

    low_sat = s < 40
    out[low_sat & (v >= 195)] = _IDX["white"]
    out[low_sat & (v >= 135) & (v < 195)] = _IDX["silver"]
    out[low_sat & (v >= 60) & (v < 135)] = _IDX["gray"]

    col = ~low_sat
    out[col & ((h < 8) | (h >= 170))] = _IDX["red"]
    out[col & (h >= 8) & (h < 21)] = _IDX["orange"]
    out[col & (h >= 21) & (h < 34)] = _IDX["yellow"]
    out[col & (h >= 34) & (h < 86)] = _IDX["green"]
    out[col & (h >= 86) & (h < 131)] = _IDX["blue"]
    out[col & (h >= 131) & (h < 156)] = _IDX["purple"]
    out[col & (h >= 156) & (h < 170)] = _IDX["pink"]
    # refinements
    out[col & (h >= 5) & (h < 26) & (v < 150) & (s >= 60)] = _IDX["brown"]
    out[col & (h >= 8) & (h < 32) & (s < 95) & (v >= 150)] = _IDX["beige"]
    # very dark pixels are black whatever their hue
    out[v < 60] = _IDX["black"]
    # dark & weakly coloured => black as well (CCTV noise)
    out[(v < 85) & (s < 70)] = _IDX["black"]
    return out


def dominant_color(bgr: np.ndarray, region=(0.0, 1.0, 0.0, 1.0)) -> str | None:
    """Dominant colour name of ``bgr`` within region=(y0,y1,x0,x1) fractions."""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    y0, y1, x0, x1 = region
    sub = bgr[int(h * y0): max(int(h * y1), int(h * y0) + 1),
              int(w * x0): max(int(w * x1), int(w * x0) + 1)]
    if sub.size == 0:
        return None
    # downscale: speed and noise
    scale = 64.0 / max(sub.shape[:2])
    if scale < 1:
        sub = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    idx = name_pixels(hsv)
    counts = np.bincount(idx, minlength=len(NAMES))
    return NAMES[int(counts.argmax())]


def vehicle_color(crop_bgr: np.ndarray) -> str | None:
    """Colour of the body: central band, skipping roof/windows and wheels."""
    return dominant_color(crop_bgr, region=(0.25, 0.80, 0.10, 0.90))


def clothing_colors(person_bgr: np.ndarray) -> tuple[str | None, str | None]:
    """(upper, lower) clothing colours from a person crop."""
    upper = dominant_color(person_bgr, region=(0.18, 0.48, 0.22, 0.78))
    lower = dominant_color(person_bgr, region=(0.55, 0.90, 0.25, 0.75))
    return upper, lower
