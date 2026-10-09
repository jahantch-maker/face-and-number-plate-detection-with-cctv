"""Turn a public Pakistani plate dataset that marks every LETTER with a box
(Roboflow "YOLO" export with classes 0-9, A-Z and a plate class) into plate
photos + plate text, the format the plate reader is trained on.

    python training/convert_char_boxes.py <unzipped export folder> <output folder>

Writes <output>/images/*.jpg and <output>/labels.csv (columns image_path,
plate_text). Plates whose characters do not form a Pakistani plate (letters
then digits) are skipped, so foreign and synthetic plates drop out.
Two-line plates (bikes) are read top row first; the small year is dropped,
exactly like the app does: LEA-17-5989 -> LEA5989.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import cv2
import yaml

from common import ROOT  # noqa: F401  (puts the app on the import path)
from gatevision.plates import correct_pk_plate, drop_year

PLATE_NAMES = {"licenseplate", "license_plate", "plate", "number-plate", "numberplate", "number_plate"}


def class_names(root: Path) -> list[str]:
    for p in [root / "data.yaml", *root.glob("*/data.yaml")]:
        if p.exists():
            names = yaml.safe_load(p.read_text(encoding="utf-8"))["names"]
            return list(names.values()) if isinstance(names, dict) else list(names)
    raise SystemExit(f"no data.yaml in {root}")


def read_boxes(label_file: Path, w: int, h: int, names: list[str]):
    out = []
    for line in label_file.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if len(parts) > 5:                                  # polygon: take its bounding box
            xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
            cx, cy, bw, bh = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        else:
            cx, cy, bw, bh = map(float, parts[1:5])
        out.append((names[cls], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def plate_text(chars) -> str:
    """Characters inside one plate -> text, top row first for two-line plates."""
    if not chars:
        return ""
    hs = [c[4] - c[2] for c in chars]
    med = statistics.median(hs)
    chars = sorted(chars, key=lambda c: (c[2] + c[4]) / 2)
    rows, row = [], [chars[0]]
    for c in chars[1:]:
        if (c[2] + c[4]) / 2 - (row[-1][2] + row[-1][4]) / 2 > 0.6 * med:
            rows.append(row)
            row = [c]
        else:
            row.append(c)
    rows.append(row)
    return "".join(c[0] for r in rows for c in sorted(r, key=lambda c: c[1])).upper()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    names = class_names(src)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for img_path in sorted(p for p in src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        label_file = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not label_file.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_boxes(label_file, w, h, names)
        plates = [b for b in boxes if b[0].lower() in PLATE_NAMES]
        chars = [b for b in boxes if len(b[0]) == 1 and b[0].isalnum()]
        for i, (_, x1, y1, x2, y2) in enumerate(plates):
            inside = [c for c in chars if x1 <= (c[1] + c[3]) / 2 <= x2 and y1 <= (c[2] + c[4]) / 2 <= y2]
            text, valid = correct_pk_plate(plate_text(inside))
            text = drop_year(text)
            if not valid or len(text) < 3:
                skipped += 1
                continue
            px, py = (x2 - x1) * 0.04, (y2 - y1) * 0.04
            crop = img[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))]
            if crop.shape[0] < 12 or crop.shape[1] < 24:
                skipped += 1
                continue
            name = f"{img_path.stem}_{i}.jpg"
            cv2.imwrite(str(dst / "images" / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"image_path": f"images/{name}", "plate_text": text})
    with (dst / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["image_path", "plate_text"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} plates written to {dst}, {skipped} skipped (not a Pakistani pattern or too small).")


if __name__ == "__main__":
    main()
