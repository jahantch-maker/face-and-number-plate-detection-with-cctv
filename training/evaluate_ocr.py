"""Score the plate reader on labelled plate photos: how many does it read
exactly right? Run it before and after training to see the real gain.

    evaluate_ocr.bat                                   (double-click: scores the reader in use now)
    python training/evaluate_ocr.py                    (same)
    python training/evaluate_ocr.py --model models/pk_plate_ocr.onnx
    python training/evaluate_ocr.py --labels some/labels.csv --model cct-xs-v1-global-model

--labels  data/training/plates/labels.csv by default (what label_plates.bat
          writes), or any CSV with image_path + plate_text columns.
--model   a built-in reader name, or a trained .onnx file (its settings file
          <name>_config.yaml must sit next to it).

The reader's answer is cleaned up exactly like the app does (0/O, 8/B fixes,
small year dropped) before comparing. Writes evaluate_ocr_results.csv with
every photo, the right answer and what was read, to look at the misses.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, has_text, plates_dir

BIKES = ("motorcycle", "bicycle")


def load_rows(path: Path):
    """[(image path, right answer, group)] from either labels format."""
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if "plate_text" in r:                               # training format
                text, img, group = r["plate_text"], path.parent / r["image_path"], "photos"
            else:                                               # labels.csv from label_plates
                text, img = r.get("label", ""), path.parent / "images" / r["file"]
                group = "bike / rickshaw" if r.get("vehicle") in BIKES else (r.get("vehicle") or "other")
            if has_text(text):
                out.append((img, text.upper(), group))
    return out


def make_ocr(model: str):
    from fast_alpr.default_ocr import DefaultOCR
    p = Path(model)
    if p.suffix == ".onnx":
        p = p if p.is_absolute() else ROOT / p
        cfg = p.with_name(p.stem + "_config.yaml")
        if not cfg.exists():
            raise SystemExit(f"settings file {cfg} not found next to the model")
        return DefaultOCR(hub_ocr_model=None, model_path=p, config_path=cfg)
    return DefaultOCR(hub_ocr_model=model)


def read(ocr, img):
    from gatevision.plates import correct_pk_plate, drop_year
    o = ocr.predict(img)
    if o is None or not o.text:
        return "", 0.0
    conf = o.confidence
    conf = float(np.mean(conf)) if isinstance(conf, (list, tuple, np.ndarray)) and len(conf) else float(conf or 0)
    text, _ = correct_pk_plate(o.text.replace("_", ""))
    return drop_year(text), conf


def main():
    from gatevision.config import DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=plates_dir() / "labels.csv")
    from gatevision.config import SHIPPED_OCR
    custom = next((p for p in (ROOT / DEFAULTS["models"]["plate_ocr_custom"], ROOT / SHIPPED_OCR) if p.exists()), None)
    in_use = str(custom) if custom else DEFAULTS["models"]["plate_ocr"]
    ap.add_argument("--model", default=in_use)
    ap.add_argument("--out", type=Path, default=Path("evaluate_ocr_results.csv"))
    args = ap.parse_args()
    if not args.labels.exists():
        raise SystemExit(f"no labels at {args.labels}: run collect_plates.bat and label_plates.bat first")
    rows = load_rows(args.labels)
    if not rows:
        raise SystemExit("no labelled photos yet")
    ocr = make_ocr(args.model)
    stats = defaultdict(lambda: [0, 0])          # group -> [right, total]
    results = []
    for img_path, truth, group in rows:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        got, conf = read(ocr, img)
        ok = got == truth
        for g in (group, "ALL"):
            stats[g][0] += ok
            stats[g][1] += 1
        results.append({"photo": img_path.name, "group": group, "right_answer": truth, "read": got,
                        "confidence": f"{conf:.2f}", "correct": "yes" if ok else "no"})
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"Reader: {args.model}\n")
    print(f"{'group':<18}{'photos':>8}{'exactly right':>16}")
    for g in sorted(stats, key=lambda g: (g == "ALL", g)):
        right, total = stats[g]
        print(f"{g:<18}{total:>8}{right / total:>15.1%}")
    print("\nEvery photo with its answer:", args.out.resolve())


if __name__ == "__main__":
    main()
