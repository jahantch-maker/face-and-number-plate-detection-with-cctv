"""Writes event images to disk and the event row to the database."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2

from .db import Database


def _shrink(img, max_w):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


class EventStore:
    def __init__(self, data_dir: str | Path, db: Database, save_full_frame: bool = True):
        self.data_dir = Path(data_dir)
        self.db = db
        self.save_full_frame = save_full_frame
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, img, folder: Path, name: str, max_w: int, quality: int = 88) -> str | None:
        if img is None or img.size == 0:
            return None
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        ok = cv2.imwrite(str(path), _shrink(img, max_w), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return path.relative_to(self.data_dir).as_posix() if ok else None

    def save_event(self, ev: dict, full=None, crop=None, plate=None, face=None) -> int:
        """``ev`` holds the metadata; images are numpy BGR arrays (or None)."""
        ts = float(ev.get("ts") or time.time())
        ev["ts"] = ts
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        folder = self.data_dir / "images" / day / ev["camera_id"]
        stem = time.strftime("%H%M%S", time.localtime(ts)) + "_" + uuid.uuid4().hex[:6]
        if self.save_full_frame:
            ev["full_path"] = self._write(full, folder, f"{stem}_full.jpg", 1280, 82)
        ev["crop_path"] = self._write(crop, folder, f"{stem}_crop.jpg", 640)
        ev["plate_path"] = self._write(plate, folder, f"{stem}_plate.jpg", 480, 92)
        ev["face_path"] = self._write(face, folder, f"{stem}_face.jpg", 320, 92)
        return self.db.insert_event(ev)
