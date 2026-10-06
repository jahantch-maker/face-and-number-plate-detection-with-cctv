"""Face detection: OpenCV YuNet (preferred) with a Haar-cascade fallback."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2

log = logging.getLogger("gatevision.faces")


class FaceDetector:
    def __init__(self, model_path: str | None = None, score: float = 0.7):
        self.yunet = None
        self.haar = None
        self.score = score
        if model_path and Path(model_path).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score, 0.3, 5000)
            except Exception as exc:  # pragma: no cover
                log.warning("YuNet failed to load (%s); using Haar fallback", exc)
        if self.yunet is None:
            log.warning("YuNet model not found - using the weaker Haar fallback. "
                        "Run scripts/download_models.py for better faces.")
            self.haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def detect(self, bgr):
        """Return list of (x, y, w, h, score) in image pixels."""
        h, w = bgr.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in faces]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        return [(float(x), float(y), float(fw), float(fh), 0.6) for (x, y, fw, fh) in found]
