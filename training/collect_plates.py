"""Copy the plate photos Gate Vision has saved into the training folder.

    collect_plates.bat               (double-click; same as below)
    python training/collect_plates.py [--days 30]

Gate Vision deletes its photos after 30 days, so run this every week or two:
it copies new plate photos to data/training/plates/ (kept for good) and adds
them to labels.csv with what Gate Vision read at the time. Photos already
collected are skipped, labels already typed are never touched.
"""
from __future__ import annotations

import argparse
import time

import cv2

from common import data_dir, plates_dir, read_labels, write_labels

SAVED_PAD = 0.15     # the pipeline saves the plate box plus 15% on every side


def unpad(img, pad=SAVED_PAD):
    """Cut the saved photo back to roughly the plate box the reader sees,
    keeping a little margin (the detector's boxes are not exact either)."""
    h, w = img.shape[:2]
    keep = pad * 0.5                         # keep about half of the extra border
    fx = (pad - keep) / (1 + 2 * pad)
    x0, y0 = int(w * fx), int(h * fx)
    out = img[y0:h - y0, x0:w - x0]
    return out if out.shape[0] >= 8 and out.shape[1] >= 16 else img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=60, help="only photos from the last N days (default 60)")
    args = ap.parse_args()

    from gatevision.db import Database
    ddir = data_dir()
    db_path = ddir / "gatevision.db"
    if not db_path.exists():
        print("No Gate Vision database at", db_path)
        return
    out = plates_dir()
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.csv"
    rows = read_labels(labels_path)
    have = {r["file"] for r in rows}

    db = Database(db_path)
    since = time.time() - args.days * 86400
    events = db._conn().execute(
        "SELECT id, ts_text, camera_id, vehicle_type, plate_norm, plate_path FROM events "
        "WHERE kind = 'vehicle' AND plate_path IS NOT NULL AND ts >= ? ORDER BY ts", (since,)).fetchall()

    added = missing = 0
    for e in events:
        name = f"{e['id']}.jpg"
        if name in have:
            continue
        img = cv2.imread(str(ddir / e["plate_path"]))
        if img is None:
            missing += 1
            continue
        cv2.imwrite(str(images / name), unpad(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"file": name, "guess": e["plate_norm"] or "", "label": "",
                     "vehicle": e["vehicle_type"] or "", "camera": e["camera_id"], "time": e["ts_text"]})
        have.add(name)
        added += 1
    write_labels(labels_path, rows)

    todo = sum(1 for r in rows if not r["label"])
    bikes = sum(1 for r in rows if r["vehicle"] in ("motorcycle", "bicycle"))
    print(f"Added {added} new plate photos ({missing} photos were already deleted).")
    print(f"Training folder now has {len(rows)} photos, {bikes} of them bikes; {todo} still need a label.")
    print("Folder:", out)
    print("Next: run label_plates.bat and type the correct plate for each photo.")


if __name__ == "__main__":
    main()
