"""Measure how well the plate reader does on a folder of photos (no camera needed).

    python benchmark_plates.py "C:\\path\\to\\Cars"

Prints one line per photo and a summary, and writes benchmark_plates.csv
(photo, plate found?, text, confidence) so the answers can be checked by eye.
Use it on the Pakistani car photos, and re-run after every model update to see
whether the numbers went up.
"""
import csv
import sys
import time
from pathlib import Path

import cv2

from gatevision.config import DEFAULTS
from gatevision.plates import PlateReader, pretty, vote


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    folder = Path(sys.argv[1])
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print("no photos found in", folder)
        return
    m = DEFAULTS["models"]
    reader = PlateReader(m["plate_detector"], m["plate_ocr"], ocr_custom=m.get("plate_ocr_custom"))
    print("plate finder:", reader.detector_model, "| reader:", reader.ocr_name, "|", len(files), "photos\n")
    rows, found, readable = [], 0, 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 1600:                      # big camera photos: shrink like a video frame
            s = 1600 / max(h, w)
            img = cv2.resize(img, None, fx=s, fy=s)
        reads = [r for r in reader.read_robust(img) if r.det_conf >= 0.35]
        text, share, best = vote([r for r in reads if r.text], 0.2)
        has_box = bool(reads)
        found += has_box
        readable += bool(text)
        conf = round(share * (best.conf if best else 0), 2)
        rows.append([p.name, int(has_box), pretty(text) if text else "", conf])
        print(f"{i:4d}/{len(files)}  {p.name:22s} {'plate found' if has_box else 'NO PLATE':11s} {pretty(text) if text else '-':12s} {conf if text else ''}")
    with open("benchmark_plates.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["photo", "plate_found", "text", "confidence"]] + rows)
    n = len(rows)
    print(f"\nSUMMARY: {n} photos | plate found {found} ({100*found//max(n,1)}%) | text read {readable} ({100*readable//max(n,1)}%) | {time.time()-t0:.0f}s")
    print("Saved benchmark_plates.csv - open it in Excel and compare the text with the photos.")


if __name__ == "__main__":
    main()
