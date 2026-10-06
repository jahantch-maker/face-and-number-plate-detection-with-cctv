"""Number-plate reading (open-source fast-alpr: plate detector + OCR, ONNX).

Pakistani plates are letters followed by digits (ASY-3549, ARG-808, DH-121,
LEA-09-1234). OCR often confuses 0/O, 1/I, 8/B, 5/S, 2/Z, so we correct each
part by position and then vote over many frames of the same vehicle.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

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


class PlateReader:
    def __init__(self, detector_model: str, ocr_model: str, use_gpu: bool = False):
        from fast_alpr import ALPR  # lazy import

        kwargs = {}
        if use_gpu:
            kwargs.update(detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                          ocr_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        try:
            self.alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model, **kwargs)
        except TypeError:  # older fast-alpr without provider kwargs
            self.alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)

    def read(self, bgr) -> list[PlateRead]:
        out: list[PlateRead] = []
        for r in self.alpr.predict(bgr) or []:
            ocr = getattr(r, "ocr", None)
            if ocr is None or not getattr(ocr, "text", None):
                continue
            conf = getattr(ocr, "confidence", 0.0)
            if isinstance(conf, (list, tuple)):
                conf = sum(conf) / len(conf) if conf else 0.0
            text, valid = correct_pk_plate(ocr.text)
            if len(text) < 3:
                continue
            bb = r.detection.bounding_box
            out.append(PlateRead(text, ocr.text, float(conf), valid,
                                 (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))))
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
    for r in reads:
        if r.conf < min_conf:
            continue
        w = r.conf * (1.6 if r.valid else 0.6)
        score[r.text] += w
        total += w
        if r.text not in best or r.conf > best[r.text].conf:
            best[r.text] = r
    if not score:
        return None, 0.0, None
    text = max(score, key=score.get)
    return text, score[text] / total if total else 0.0, best[text]
