"""Shared bits for the training tools: where the training photos live and the
labels file format.

    data/training/plates/images/<event id>.jpg   plate photos (kept for good:
                                                 the normal 30-day clean-up
                                                 does not touch this folder)
    data/training/plates/labels.csv              one row per photo

labels.csv columns:
    file     image name inside images/
    guess    what Gate Vision read at the time (may be wrong or empty)
    label    the CORRECT plate, typed by a person: letters + number, no dash,
             no year (LEA-17-5989 -> LEA5989). Empty = not checked yet.
             "-" = unreadable (skipped in training and in the score).
    vehicle  car / motorcycle / truck ...
    camera   camera id
    time     when it was seen
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIELDS = ["file", "guess", "label", "vehicle", "camera", "time"]
UNREADABLE = "-"


def data_dir() -> Path:
    from gatevision.config import load_config
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        d = Path(load_config(cfg_path)["storage"]["data_dir"])
    else:
        d = Path("data")
    return d if d.is_absolute() else ROOT / d


def plates_dir() -> Path:
    return data_dir() / "training" / "plates"


def clean_label(text: str) -> str:
    """What people type -> the stored label. Keeps "-" (unreadable) as is."""
    t = (text or "").strip().upper()
    if t in (UNREADABLE, "X", "?"):
        return UNREADABLE
    return re.sub(r"[^A-Z0-9]", "", t)


def read_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (r.get(k) or "") for k in FIELDS} for r in csv.DictReader(f)]


def write_labels(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)          # never leave a half-written labels file
