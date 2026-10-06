"""Threaded RTSP reader that always holds the newest frame and auto-reconnects."""
from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger("gatevision.stream")


class RTSPStream:
    def __init__(self, url: str, name: str = "cam"):
        self.url = url
        self.name = name
        self._lock = threading.Lock()
        self._frame = None
        self._idx = 0
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{name}", daemon=True)
        self.connected = False
        self.frames_total = 0

    def start(self) -> "RTSPStream":
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def last_frame_time(self) -> float:
        return self._ts

    def read(self, last_idx: int = -1):
        """Return (idx, frame, timestamp) if a newer frame exists, else None."""
        with self._lock:
            if self._frame is None or self._idx == last_idx:
                return None
            return self._idx, self._frame, self._ts

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.connected = False
                log.warning("%s: cannot open stream, retry in %.0fs", self.name, backoff)
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30)
                continue
            log.info("%s: stream connected", self.name)
            self.connected = True
            backoff = 1.0
            fails = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 25:
                        log.warning("%s: stream lost, reconnecting", self.name)
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                with self._lock:
                    self._frame = frame
                    self._idx += 1
                    self._ts = time.time()
                self.frames_total += 1
            cap.release()
            self.connected = False
            self._stop.wait(1.0)
