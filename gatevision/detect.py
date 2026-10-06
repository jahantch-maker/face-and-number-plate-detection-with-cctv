"""Vehicle / person detection with tracking (Ultralytics YOLO + ByteTrack).

The detector is hidden behind a tiny interface so the rest of the system (and
the tests) never depend on a specific model.
"""
from __future__ import annotations

from dataclasses import dataclass

# COCO class ids
PERSON = 0
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Det:
    track_id: int
    label: str          # car / motorcycle / bus / truck / bicycle / person
    conf: float
    box: tuple          # (x1, y1, x2, y2) in pixels


class YoloTracker:
    """One instance per camera: ByteTrack keeps per-camera state inside it."""

    def __init__(self, weights="yolo11s.pt", device=None, imgsz=960, conf=0.35):
        from ultralytics import YOLO  # imported lazily: heavy

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def track(self, frame, role: str) -> list[Det]:
        classes = [PERSON] if role == "face" else list(VEHICLE_CLASSES)
        res = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=classes,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        out: list[Det] = []
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.int().cpu().tolist()
        cls = boxes.cls.int().cpu().tolist()
        conf = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        for tid, c, p, b in zip(ids, cls, conf, xyxy):
            label = "person" if c == PERSON else VEHICLE_CLASSES.get(c, "vehicle")
            out.append(Det(int(tid), label, float(p), tuple(float(v) for v in b)))
        return out
