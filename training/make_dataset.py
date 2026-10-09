"""Combine our own labelled gate photos with converted public datasets into
the train / check lists the plate reader is trained on.

    python training/make_dataset.py --own data/training/plates/labels.csv \
        --public converted/pk_chars/labels.csv [--public more/labels.csv ...] --out dataset

Rules:
- the same plate number never appears in both lists, so the check score is
  honest (the reader cannot just remember a car it has already seen);
- the check list is our own gate photos (when there are at least 40),
  because that is what we care about;
- our own photos are repeated in training (bikes and rickshaws more often),
  so a few hundred of them still count against thousands of public photos.

Writes <out>/train.csv, <out>/val.csv and <out>/val_bikes.csv (image_path, plate_text).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

from common import has_text

BIKES = ("motorcycle", "bicycle")


def own_rows(path: Path):
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = (r.get("label") or "").upper()
            if has_text(label):
                out.append({"img": str((path.parent / "images" / r["file"]).resolve()), "text": label,
                            "bike": r.get("vehicle") in BIKES, "own": True})
    return out


def public_rows(path: Path):
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append({"img": str((path.parent / r["image_path"]).resolve()), "text": r["plate_text"].upper(),
                        "bike": False, "own": False})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own", type=Path, help="labels.csv from label_plates")
    ap.add_argument("--public", type=Path, action="append", default=[], help="labels.csv from convert_char_boxes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-share", type=float, default=0.2)
    ap.add_argument("--own-repeat", type=int, default=4, help="times each own photo is used in training")
    ap.add_argument("--bike-repeat", type=int, default=8, help="times each own bike photo is used")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    own = own_rows(args.own) if args.own and args.own.exists() else []
    pub = [r for p in args.public for r in public_rows(p)]
    if not own and not pub:
        raise SystemExit("nothing to train on")
    rnd = random.Random(args.seed)

    def split(rows):
        """Split by plate number, so one plate is only in one list."""
        plates = sorted({r["text"] for r in rows})
        rnd.shuffle(plates)
        val_plates = set(plates[:int(len(plates) * args.val_share)])
        return [r for r in rows if r["text"] not in val_plates], [r for r in rows if r["text"] in val_plates]

    own_train, own_val = split(own)
    if len(own_val) >= 40:
        pub_val_plates = {r["text"] for r in own_val}
        train = own_train + [r for r in pub if r["text"] not in pub_val_plates]
        val = own_val
    else:
        pub_train, pub_val = split(pub)
        held = {r["text"] for r in own_val + pub_val}
        train = own_train + [r for r in pub_train if r["text"] not in held]
        val = own_val + pub_val

    # repeat our own photos so they are not drowned out by the public ones
    boosted = []
    for r in train:
        n = (args.bike_repeat if r["bike"] else args.own_repeat) if r["own"] else 1
        boosted.extend([r] * n)
    rnd.shuffle(boosted)

    args.out.mkdir(parents=True, exist_ok=True)
    val_bikes = [r for r in val if r["bike"]]
    for name, rows in (("train.csv", boosted), ("val.csv", val), ("val_bikes.csv", val_bikes)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "plate_text"])
            for r in rows:
                if re.fullmatch(r"[A-Z0-9]{1,9}", r["text"]):    # the trainer wants paths relative to the CSV
                    w.writerow([os.path.relpath(r["img"], args.out.resolve()).replace(os.sep, "/"), r["text"]])
    print(f"train: {len(boosted)} rows ({len(own_train)} own photos, {sum(r['bike'] for r in own_train)} bikes, "
          f"{len(train) - len(own_train)} public)")
    print(f"check: {len(val)} photos ({len(own_val)} own, {len(val_bikes)} of them bikes)")


if __name__ == "__main__":
    main()
