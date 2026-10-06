"""Number-plate reading (open-source fast-alpr: plate detector + OCR, ONNX).

Pakistani plates are letters followed by digits (ASY-3549, ARG-808, DH-121,
LEA-09-1234). OCR often confuses 0/O, 1/I, 8/B, 5/S, 2/Z, so we correct each
part by position and then vote over many frames of the same vehicle.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}

# letters(2-4) + digits(1-6): covers ASY3549, ARG808, DH121, LEA091234
_PK_RE = re.compile(r"^[A-Z]{2,4}\d{1,6}$")


@dataclass
class PlateRead:
    text: str            # corrected, upper-case, no separators (e.g. ASY3549)
    raw: str
    conf: float
    valid: bool          # matches the letters+digits pattern
    box: tuple | None = None   # (x1,y1,x2,y2) relative to the image given to read()
    det_conf: float = 0.0      # how sure the detector is that this is a plate


def variants(bgr):
    """The same picture in versions that survive night-time colour casts.

    1. as is
    2. gray-world white balance + contrast boost (removes a red/orange cast)
    3. grayscale + contrast boost
    """
    yield "original", bgr
    f = bgr.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    wb = np.clip(f * (means.mean() / means), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    yield "balanced", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    g = clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    yield "gray", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def crop_variants(plate):
    """Many looks at ONE plate picture: bigger, and slightly rotated (bikes park
    at an angle). The whole plate is always kept - two-line plates carry the
    number on the bottom line."""
    h, w = plate.shape[:2]
    if h < 8 or w < 8:
        return
    scale = 2.0 if w < 300 else 1.0
    base = cv2.resize(plate, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else plate
    bh, bw = base.shape[:2]
    yield "full", base
    for ang in (-12, -6, 6, 12):
        m = cv2.getRotationMatrix2D((bw / 2, bh / 2), ang, 1.0)
        yield f"rot{ang}", cv2.warpAffine(base, m, (bw, bh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


_YEAR_NUMBER = re.compile(r"^([A-Z]{2,4})(\d{2})(\d{3,4})$")
_YEAR_ONLY = re.compile(r"^([A-Z]{2,4})(\d{2})$")


def drop_year(text: str) -> str:
    """Punjab/Islamabad plates print a small registration year after the letters
    (LEA-17-5989). Keep the letters and the big number, skip the year."""
    m = _YEAR_NUMBER.match(text or "")
    if m and 0 <= int(m.group(2)) <= 30:
        return m.group(1) + m.group(3)
    return text


def is_year_only(text: str) -> bool:
    """'MNC17' = the reader saw only the letters and the small year, not the number."""
    m = _YEAR_ONLY.match(text or "")
    return bool(m and 0 <= int(m.group(2)) <= 30)


def pretty(text: str) -> str:
    """ASY3549 -> ASY-3549 for display."""
    m = re.match(r"^([A-Z]{2,4})(\d+)$", text or "")
    return f"{m.group(1)}-{m.group(2)}" if m else (text or "")


def correct_pk_plate(raw: str) -> tuple[str, bool]:
    """Fix look-alike characters by position. Returns (text, matches_pattern)."""
    s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    if _PK_RE.match(s):
        return s, True
    best = None
    for k in (2, 3, 4):
        if len(s) <= k or len(s) - k > 6:
            continue
        head = "".join(_TO_LETTER.get(ch, ch) for ch in s[:k])
        tail = "".join(_TO_DIGIT.get(ch, ch) for ch in s[k:])
        cand = head + tail
        if not _PK_RE.match(cand):
            continue
        head_changes = sum(a != b for a, b in zip(s[:k], head))
        if head_changes > 1:       # a "plate" that is mostly digits up front is not ours
            continue
        changes = sum(a != b for a, b in zip(s, cand))
        if best is None or changes < best[0]:
            best = (changes, cand)
    if best and best[0] <= 2:
        return best[1], True
    return s, False


FALLBACK_DETECTORS = ["yolo-v9-t-640-license-plate-end2end", "yolo-v9-t-512-license-plate-end2end",
                      "yolo-v9-t-384-license-plate-end2end"]
FALLBACK_DETECTOR = FALLBACK_DETECTORS[-1]


class PlateReader:
    def __init__(self, detector_model: str, ocr_model: str, use_gpu: bool = False,
                 detector_conf_thresh: float | None = None):
        import logging
        from fast_alpr import ALPR  # lazy import

        log = logging.getLogger("gatevision.plates")
        gpu = {"detector_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
               "ocr_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]} if use_gpu else {}
        thresh = {"detector_conf_thresh": detector_conf_thresh} if detector_conf_thresh else {}
        # tolerate fast-alpr versions that do not know some keywords
        kw_sets = [{**gpu, **thresh}, thresh, gpu, {}]
        models = [detector_model] + [m for m in FALLBACK_DETECTORS if m != detector_model]
        last = None
        for model in models:
            for kw in kw_sets:
                try:
                    self.alpr = ALPR(detector_model=model, ocr_model=ocr_model, **kw)
                    self.detector_model = model
                    if model != detector_model:
                        log.warning("plate detector %s unavailable (%s) - using %s", detector_model, last, model)
                    return
                except TypeError as exc:
                    last = exc
                except Exception as exc:          # unknown model name, download problem ...
                    last = exc
                    break
        raise last

    def read(self, bgr, extra_ocr: bool = True) -> list[PlateRead]:
        """One pass. Plates the detector found but the OCR could not read come
        back with empty text, so the caller can still keep their photo."""
        out: list[PlateRead] = []
        for r in self.alpr.predict(bgr) or []:
            bb = r.detection.bounding_box
            box = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))
            det_conf = float(getattr(r.detection, "confidence", 0.0) or 0.0)
            ocr = getattr(r, "ocr", None)
            if ocr is None or not getattr(ocr, "text", None):
                out.append(PlateRead("", "", 0.0, False, box, det_conf))
                continue
            conf = getattr(ocr, "confidence", 0.0)
            if isinstance(conf, (list, tuple, np.ndarray)):
                conf = float(np.mean(conf)) if len(conf) else 0.0
            text, valid = correct_pk_plate(ocr.text)
            text = drop_year(text)
            if len(text) < 3:
                out.append(PlateRead("", ocr.text, 0.0, False, box, det_conf))
                continue
            out.append(PlateRead(text, ocr.text, float(conf), valid, box, det_conf))
        if extra_ocr:
            for r in list(out):
                if r.box and r.det_conf >= 0.35:
                    out.extend(self._extra_reads(bgr, r.box, r.det_conf))
        return out

    def _extra_reads(self, bgr, box, det_conf) -> list[PlateRead]:
        """Run only the OCR again on rotated / enlarged / top-line versions of the plate."""
        ocr_model = getattr(self.alpr, "ocr", None)
        if ocr_model is None:
            return []
        x1, y1, x2, y2 = box
        h, w = bgr.shape[:2]
        px, py = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
        crop = bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        out: list[PlateRead] = []
        for _name, img in crop_variants(crop):
            try:
                o = ocr_model.predict(img)
            except Exception:
                continue
            if o is None or not getattr(o, "text", None):
                continue
            conf = getattr(o, "confidence", 0.0)
            if isinstance(conf, (list, tuple, np.ndarray)):
                conf = float(np.mean(conf)) if len(conf) else 0.0
            text, valid = correct_pk_plate(o.text)
            text = drop_year(text)
            if len(text) >= 3 and valid:
                out.append(PlateRead(text, o.text, float(conf), valid, box, det_conf))
        return out

    def read_robust(self, bgr) -> list[PlateRead]:
        """Read the picture in several colour-corrected versions."""
        out: list[PlateRead] = []
        for _name, img in variants(bgr):
            out.extend(self.read(img))
        return out


def vote(reads: list[PlateRead], min_conf: float = 0.0):
    """Pick the most trusted plate over many frames.

    Returns (text, confidence, best_read) or (None, 0, None).
    A read's weight is its OCR confidence, boosted when it fits the plate
    pattern. Confidence is the winner's share of the total weight.
    """
    score: dict[str, float] = defaultdict(float)
    best: dict[str, PlateRead] = {}
    total = 0.0
    full_prefixes = {re.match(r"[A-Z]+", r.text).group(0) for r in reads
                     if r.text and not is_year_only(r.text) and re.match(r"[A-Z]+", r.text)}
    for r in reads:
        if r.conf < min_conf:
            continue
        if is_year_only(r.text) and re.match(r"[A-Z]+", r.text) and \
                any(p.startswith(re.match(r"[A-Z]+", r.text).group(0)[:2]) for p in full_prefixes):
            continue                      # saw only letters + year; a fuller read exists
        w = r.conf * (1.6 if r.valid else 0.6)
        score[r.text] += w
        total += w
        if r.text not in best or r.conf > best[r.text].conf:
            best[r.text] = r
    if not score:
        return None, 0.0, None
    text = max(score, key=score.get)
    return text, score[text] / total if total else 0.0, best[text]
