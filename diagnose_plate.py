"""Compare plate finders on one picture and show what each one sees.

    python diagnose_plate.py "D:\\gatevision_data\\images\\2026-10-07\\out_plate\\014339_c04401_crop.jpg"

It tries several plate-finder / reader models on the picture (normal,
colour-corrected and grayscale), prints every box and text, marks texts that
look like a real plate, and saves <name>_diag.jpg. It can take a few minutes the
first time because each model is downloaded once.
Send me the printed text and that picture.
"""
import logging
import sys
from pathlib import Path

import cv2

from gatevision.plates import PlateReader, pretty, variants

logging.disable(logging.INFO)
if len(sys.argv) < 2:
    raise SystemExit(__doc__)
path = Path(sys.argv[1])
img = cv2.imread(str(path))
if img is None:
    raise SystemExit(f"cannot read {path}")
print(f"picture: {img.shape[1]}x{img.shape[0]} px\n")

DETECTORS = ["yolo-v9-t-384-license-plate-end2end", "yolo-v9-t-512-license-plate-end2end",
             "yolo-v9-t-640-license-plate-end2end", "yolo-v9-s-608-license-plate-end2end"]
OCRS = ["cct-xs-v1-global-model", "cct-xs-v2-global-model"]
tried = [(n, v) for n, v in variants(img) if n in ("original", "gray")]

tiles, plausible = [], []
for det in DETECTORS:
    for ocr in OCRS:
        label = f"{det.replace('-license-plate-end2end', '')} + {ocr.replace('-global-model', '')}"
        try:
            reader = PlateReader(det, ocr, detector_conf_thresh=0.25)
            if reader.detector_model != det:
                print(f"== {label}: detector not available, skipped")
                continue
        except Exception as exc:                       # model not downloadable etc.
            print(f"== {label}: could not load ({type(exc).__name__}) - skipped")
            continue
        print(f"== {label}")
        for name, v in tried:
            reads = reader.read(v)
            if not reads:
                print(f"   [{name}] no plate box")
            vis = v.copy()
            for r in reads:
                x1, y1, x2, y2 = r.box
                txt = pretty(r.text) if r.text else "(no text)"
                good = bool(r.text) and r.valid and r.det_conf >= 0.35
                if good:
                    plausible.append((label, name, txt, r.conf))
                print(f"   [{name}] box {x2 - x1}px wide, finder {r.det_conf:.2f}: {txt} (reader {r.conf:.2f})"
                      + ("   <== looks like a plate" if good else ""))
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0) if good else (0, 0, 255), 3)
                cv2.putText(vis, f"{txt} {r.conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if good else (0, 0, 255), 2)
            tiles.append(cv2.resize(vis, (360, int(vis.shape[0] * 360 / vis.shape[1]))))
            cv2.putText(tiles[-1], f"{label[:34]} {name}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

print("\n==== SUMMARY ====")
if plausible:
    for label, name, txt, conf in plausible:
        print(f"  {txt:12s} reader {conf:.2f}   {label}  [{name}]")
else:
    print("  None of the models found a plausible plate. Send me this output and the picture.")
if tiles:
    cols = 4
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(tiles[0] * 0)
    rows = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    out = path.with_name(path.stem + "_diag.jpg")
    cv2.imwrite(str(out), cv2.vconcat(rows))
    print("\nsaved", out)
