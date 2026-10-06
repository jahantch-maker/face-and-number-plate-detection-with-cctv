"""Threaded RTSP reader that always holds the newest frame and auto-reconnects."""
from __future__ import annotations

import logging
import threading
import time

import cv2

log = logging.getLogger("gatevision.stream")

# NVRs lock an account after a few wrong logins (Dahua: 5 tries -> locked for
# 30 min). All cameras share one account, so failures are counted together:
# 3 failed connection attempts within 2 minutes => every camera pauses 10 min.
_FAIL_WINDOW, _FAIL_LIMIT, _PAUSE = 120.0, 3, 600.0
_fail_lock = threading.Lock()
_fail_times: list = []
_pause_until = 0.0
_connected: set = set()    # names of cameras that are currently streaming


def pause_remaining(now: float | None = None) -> float:
    """Seconds every camera must still wait (0 when no pause is active)."""
    now = time.time() if now is None else now
    return max(0.0, _pause_until - now)


def register_failure(now: float | None = None, name: str = "?") -> float:
    """Record a failed connection; return how many seconds to wait before retrying.

    If any OTHER camera is streaming, the login works and this camera is just
    switched off / unreachable - nothing to protect, retry normally.
    """
    global _pause_until
    now = time.time() if now is None else now
    with _fail_lock:
        if _connected - {name}:
            return 30.0
        _fail_times[:] = [t for t in _fail_times if now - t < _FAIL_WINDOW] + [now]
        if len(_fail_times) >= _FAIL_LIMIT:
            _fail_times.clear()
            _pause_until = now + _PAUSE
        return max(15.0, _pause_until - now)


def register_success(name: str = "?"):
    global _pause_until
    with _fail_lock:
        _connected.add(name)
        _fail_times.clear()
        _pause_until = 0.0


def register_disconnect(name: str = "?"):
    with _fail_lock:
        _connected.discard(name)


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
        while not self._stop.is_set():
            hold = pause_remaining()
            if hold > 0:                       # another camera triggered the safety pause
                self._stop.wait(min(hold, 30.0))
                continue
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.connected = False
                wait = register_failure(name=self.name)
                if wait >= _PAUSE:
                    log.error("%s: repeated connection failures - pausing ALL cameras for %d min so the NVR "
                              "does not lock the account. Check the NVR password / permissions, then restart.",
                              self.name, _PAUSE // 60)
                else:
                    log.warning("%s: cannot open stream, retry in %.0fs", self.name, wait)
                cap.release()
                self._stop.wait(wait)
                continue
            log.info("%s: stream connected", self.name)
            self.connected = True
            register_success(self.name)
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
            register_disconnect(self.name)
            self._stop.wait(1.0)
