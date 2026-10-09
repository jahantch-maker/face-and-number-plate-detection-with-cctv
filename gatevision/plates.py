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
    char_conf: float | None = None   # mean confidence of the characters actually read (None: use conf)

    @property
    def text_conf(self) -> float:
        return self.conf if self.char_conf is None else self.char_conf


def _confs(o) -> tuple[float, float]:
    """(mean over all reader slots, mean over the characters actually read).
    The empty slots after a short plate are always sure, which lifts the first
    number even for a foot or a lamp; the second one stays low for those."""
    conf = getattr(o, "confidence", 0.0)
    if not isinstance(conf, (list, tuple, np.ndarray)):
        return float(conf or 0.0), float(conf or 0.0)
    if not len(conf):
        return 0.0, 0.0
    n = len((getattr(o, "text", "") or "").replace("_", ""))
    return float(np.mean(conf)), float(np.mean(conf[:n])) if n else 0.0


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


def split_rows(plate):
    """Split a two-line plate (letters + small year on top, big number below) into
    its two rows. Returns (top, bottom) pictures, or None for one-line plates.

    Works from where the ink is: a clear empty gap near the middle with two
    text bands of similar height. A one-line plate with a small city / name
    line underneath has bands of very different height and is NOT split.
    """
    h, w = plate.shape[:2]
    if h < 24 or w < 40:
        return None
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
    x0, x1 = int(w * 0.18), int(w * 0.97)            # skip the green emblem strip on the left
    y0, y1 = int(h * 0.08), int(h * 0.92)            # skip the plate frame
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return None
    region = cv2.GaussianBlur(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2)).apply(region), (3, 3), 0)
    t, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = region < t
    ink = dark if dark.mean() < 0.5 else ~dark       # the smaller side is the writing
    prof = ink.mean(axis=1)
    k = max(3, len(prof) // 25)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    n = len(prof)
    lo, hi = int(n * 0.30), int(n * 0.70)
    if hi <= lo:
        return None
    cut = lo + int(np.argmin(prof[lo:hi]))
    top_peak, bot_peak = float(prof[:cut].max()), float(prof[cut:].max())
    if min(top_peak, bot_peak) < 0.10 or prof[cut] > 0.50 * min(top_peak, bot_peak):
        return None                                   # no clear empty gap between two text rows

    def band_height(seg):
        rows = np.where(seg > 0.35 * seg.max())[0]
        return (rows[-1] - rows[0] + 1) if len(rows) else 0
    ht, hb = band_height(prof[:cut]), band_height(prof[cut:])
    if min(ht, hb) < 0.55 * max(ht, hb) or min(ht, hb) < 0.12 * n:
        return None                                   # unequal rows: one-line plate with a small caption
    cut_abs = y0 + cut
    pad = max(2, int(h * 0.04))
    return plate[: min(h, cut_abs + pad)], plate[max(0, cut_abs - pad):]


def compose_two_line(top_text: str, bottom_text: str):
    """Letters from the top row (the small year is skipped) + digits from the bottom row.
    Returns the plate text, or None if the two rows do not look like a plate."""
    top = re.sub(r"[^A-Z0-9]", "", (top_text or "").upper())
    bot = re.sub(r"[^A-Z0-9]", "", (bottom_text or "").upper())
    if len(top) < 2 or not bot:
        return None
    m = re.match(r"^(.*?)(\d{1,2})$", top)               # MNF17 / MNF1 -> MNF (year skipped, even half-read)
    if m and len(m.group(1)) >= 2:
        top = m.group(1)
    if len(top) == 4 and top[-1] in _TO_DIGIT:         # MNFI = MNF + a year digit read as a letter
        top = top[:3]
    letters = "".join(_TO_LETTER.get(ch, ch) for ch in top)
    digits = "".join(_TO_DIGIT.get(ch, ch) for ch in bot)
    if not re.fullmatch(r"[A-Z]{2,4}", letters) or not re.fullmatch(r"\d{1,4}", digits):
        return None
    return letters + digits


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


def _custom_ocr_files(path):
    """(model.onnx, model_config.yaml) when a trained reader is in place, else None."""
    from pathlib import Path
    if not path:
        return None
    model = Path(path)
    cfg = model.with_name(model.stem + "_config.yaml")
    return (model, cfg) if model.is_file() and cfg.is_file() else None


class PlateReader:
    def __init__(self, detector_model: str, ocr_model: str, use_gpu: bool = False,
                 detector_conf_thresh: float | None = None, ocr_custom: str | None = None):
        import logging
        from fast_alpr import ALPR  # lazy import

        log = logging.getLogger("gatevision.plates")
        gpu = {"detector_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
               "ocr_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]} if use_gpu else {}
        thresh = {"detector_conf_thresh": detector_conf_thresh} if detector_conf_thresh else {}
        ocr = {"ocr_model": ocr_model}
        self.ocr_name = ocr_model
        custom = _custom_ocr_files(ocr_custom)
        if custom:
            try:                         # check it loads before handing it to the plate finder
                from fast_plate_ocr import LicensePlateRecognizer
                LicensePlateRecognizer(onnx_model_path=custom[0], plate_config_path=custom[1])
            except Exception as exc:     # a broken or too-new trained reader must not stop the cameras
                log.warning("trained plate reader %s could not be loaded (%s) - using %s", custom[0], exc, ocr_model)
                custom = None
        if custom:                       # our own reader, trained on Pakistani plates (training/README.md)
            ocr = {"ocr_model": None, "ocr_model_path": custom[0], "ocr_config_path": custom[1]}
            self.ocr_name = str(custom[0])
            log.info("using the trained plate reader %s", custom[0])
        # tolerate fast-alpr versions that do not know some keywords
        kw_sets = [{**gpu, **thresh}, thresh, gpu, {}]
        models = [detector_model] + [m for m in FALLBACK_DETECTORS if m != detector_model]
        last = None
        for model in models:
            for kw in kw_sets:
                try:
                    self.alpr = ALPR(detector_model=model, **ocr, **kw)
                    self.detector_model = model
                    if model != detector_model:
                        log.warning("plate detector %s unavailable (%s) - using %s", detector_model, last, model)
                    return
                except TypeError as exc:
                    last = exc
                except Exception as exc:          # unknown model name, download problem ...
                    last = exc
                    break
        if custom:                       # a broken or too-new trained reader must not stop the cameras
            log.warning("trained plate reader %s could not be loaded (%s) - using %s", custom[0], last, ocr_model)
            self.__init__(detector_model, ocr_model, use_gpu, detector_conf_thresh)
            return
        raise last

    def read(self, bgr, extra_ocr: bool = True) -> list[PlateRead]:
        """One pass. Plates the detector found but the OCR could not read come
        back with empty text, so the caller can still keep their photo."""
        out: list[PlateRead] = []
        for r in self.alpr.predict(bgr) or []:
            bb = r.detection.bounding_box
            box = (int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2))
            det_conf = float(getattr(r.detection, "confidence", 0.0) or 0.0)
            if det_conf >= 0.35:
                two = self._two_line(self._crop(bgr, box))
                if two:                      # two-line plate read row by row: letters + number, year skipped
                    out.append(PlateRead(two[0], "two-line", two[1], True, box, det_conf, two[2]))
                    continue
            ocr = getattr(r, "ocr", None)
            if ocr is None or not getattr(ocr, "text", None):
                out.append(PlateRead("", "", 0.0, False, box, det_conf))
                continue
            conf, char_conf = _confs(ocr)
            text, valid = correct_pk_plate(ocr.text)
            text = drop_year(text)
            if len(text) < 3:
                out.append(PlateRead("", ocr.text, 0.0, False, box, det_conf))
                continue
            out.append(PlateRead(text, ocr.text, conf, valid, box, det_conf, char_conf))
        if extra_ocr:
            for r in list(out):
                if r.box and r.det_conf >= 0.35:
                    out.extend(self._extra_reads(bgr, r.box, r.det_conf))
        return out

    @staticmethod
    def _crop(bgr, box, pad=0.08):
        x1, y1, x2, y2 = box
        h, w = bgr.shape[:2]
        px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
        return bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]

    def _ocr_raw(self, img):
        """(raw text, confidence, character confidence) for one picture, or ("", 0, 0)."""
        ocr_model = getattr(self.alpr, "ocr", None)
        if ocr_model is None:
            return "", 0.0, 0.0
        try:
            o = ocr_model.predict(img)
        except Exception:
            return "", 0.0, 0.0
        if o is None or not getattr(o, "text", None):
            return "", 0.0, 0.0
        return (o.text, *_confs(o))

    def _two_line(self, plate_img):
        """Read a two-line plate row by row: (text, conf, char conf) or None if it is not one / unreadable."""
        rows = split_rows(plate_img)
        if rows is None:
            return None
        top_text, top_conf, top_cc = self._ocr_raw(rows[0])
        bot_text, bot_conf, bot_cc = self._ocr_raw(rows[1])
        text = compose_two_line(top_text, bot_text)
        if not text:
            return None
        return text, min(top_conf, bot_conf), min(top_cc, bot_cc)

    def _extra_reads(self, bgr, box, det_conf) -> list[PlateRead]:
        """Run only the OCR again on rotated / enlarged / top-line versions of the plate."""
        if getattr(self.alpr, "ocr", None) is None:
            return []
        crop = self._crop(bgr, box)
        out: list[PlateRead] = []
        for _name, img in crop_variants(crop):
            two = self._two_line(img)
            if two:
                out.append(PlateRead(two[0], "two-line", two[1], True, box, det_conf, two[2]))
                continue
            raw, conf, char_conf = self._ocr_raw(img)
            if not raw:
                continue
            text, valid = correct_pk_plate(raw)
            text = drop_year(text)
            if len(text) >= 3 and valid:
                out.append(PlateRead(text, raw, conf, valid, box, det_conf, char_conf))
        return out

    def read_robust(self, bgr) -> list[PlateRead]:
        """Read the picture in several colour-corrected versions."""
        out: list[PlateRead] = []
        for _name, img in variants(bgr):
            reads = self.read(img)
            out.extend(reads)
            # a clear, plate-shaped, confident read: the other colour versions would only cost time
            if any(r.text and r.valid and r.conf >= 0.9 and r.det_conf >= 0.5 for r in reads):
                break
        return out


def vote(reads: list[PlateRead], min_conf: float = 0.0):
    """Pick the most trusted plate over many frames.

    Returns (text, confidence, best_read) or (None, 0, None).
    A read's weight is its OCR confidence, boosted when it fits the plate
    pattern. Reads of the same plate that differ in a character or two
    (AWS5573, AWS5578, ARS5573 ...) are one candidate: the text is decided
    character by character, so a clear plate is not lost because each frame
    misread a different letter. Confidence is that candidate's share of the
    total weight.
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

    # group spellings of the same plate: same length, at most 2 characters apart
    groups: list[list[str]] = []
    for t in sorted(score, key=score.get, reverse=True):
        for g in groups:
            if len(g[0]) == len(t) and sum(a != b for a, b in zip(g[0], t)) <= 2:
                g.append(t)
                break
        else:
            groups.append([t])
    group = max(groups, key=lambda g: sum(score[t] for t in g))
    weight = sum(score[t] for t in group)
    text = group[0]
    if len(group) > 1:                    # character by character, weighted
        lead: dict[int, float] = defaultdict(float)      # how many letters before the number
        for t in group:
            lead[len(re.match(r"[A-Z]*", t).group(0))] += score[t]
        n_letters = max(lead, key=lead.get)
        chars = []
        for i in range(len(text)):
            votes: dict[str, float] = defaultdict(float)
            for t in group:
                ch = _TO_LETTER.get(t[i], t[i]) if i < n_letters else _TO_DIGIT.get(t[i], t[i])
                votes[ch] += score[t]
            chars.append(max(votes, key=votes.get))
        consensus, ok = correct_pk_plate("".join(chars))
        if ok:
            text = consensus
    top = max((best[t] for t in group), key=lambda r: r.conf)
    if text not in best:
        best[text] = PlateRead(text, top.raw, top.conf, True, top.box, top.det_conf, top.char_conf)
    return text, weight / total if total else 0.0, best[text]
