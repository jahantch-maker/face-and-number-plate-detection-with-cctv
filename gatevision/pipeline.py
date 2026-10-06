"""Per-camera processing: track objects, wait for them to stop, read plate /
face, vote over many frames, and write ONE clean event per vehicle / person."""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, deque

import cv2

from .colors import clothing_colors, vehicle_color
from .plates import pretty, vote

log = logging.getLogger("gatevision.pipeline")


def _sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if w > 200:
        g = cv2.resize(g, (200, max(1, int(h * 200 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _clip_crop(frame, box, pad=0.0):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * pad, x2 + bw * pad
    y1, y2 = y1 - bh * pad, y2 + bh * pad
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


class TrackState:
    def __init__(self, track_id: int, ts: float):
        self.id = track_id
        self.first_seen = ts
        self.last_seen = ts
        self.frames = 0
        self.labels: Counter = Counter()
        self.hist: deque = deque(maxlen=80)   # (ts, cx, cy, diag)
        self.was_stopped = False
        self.best_score = -1.0
        self.best_crop = None
        self.best_full = None
        self.reads: list = []
        self.plate_img = None
        self.plate_img_conf = 0.0
        self.face_img = None
        self.face_score = 0.0
        self.ocr_attempts = 0
        self.face_attempts = 0
        self.last_try = 0.0

    def update(self, det, ts: float):
        x1, y1, x2, y2 = det.box
        self.last_seen = ts
        self.frames += 1
        self.labels[det.label] += 1
        self.hist.append(((ts), (x1 + x2) / 2, (y1 + y2) / 2, math.hypot(x2 - x1, y2 - y1)))

    def is_stopped(self, ratio: float, window: float = 1.0, min_span: float = 0.4) -> bool:
        """True when the box centre barely moved over the last ``window`` s."""
        if len(self.hist) < 3:
            return False
        t_now, cx, cy, _ = self.hist[-1]
        recent = [h for h in self.hist if t_now - h[0] <= window]
        if len(recent) < 3 or recent[-1][0] - recent[0][0] < min_span:
            return False
        diag = sum(h[3] for h in recent) / len(recent)
        moved = max(math.hypot(h[1] - cx, h[2] - cy) for h in recent)
        stopped = moved < ratio * diag
        if stopped:
            self.was_stopped = True
        return stopped


class CameraWorker:
    def __init__(self, cam: dict, cfg: dict, detector, store, plates=None, faces=None):
        self.cam = cam
        self.cfg = cfg
        self.t = cfg["tracking"]
        self.detector = detector
        self.store = store
        self.plates = plates
        self.faces = faces
        self.tracks: dict[int, TrackState] = {}
        self.last_plate: dict[str, float] = {}   # plate_norm -> ts of last event
        self.last_event_ts: float | None = None
        self.events_saved = 0

    # --------------------------------------------------------------- step
    def step(self, frame, ts: float):
        H, W = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.cam["roi"]
        roi = (rx1 * W, ry1 * H, rx2 * W, ry2 * H)
        role = self.cam["role"]

        for d in self.detector.track(frame, role):
            x1, y1, x2, y2 = d.box
            foot_x, foot_y = (x1 + x2) / 2, y2
            if not (roi[0] <= foot_x <= roi[2] and roi[1] <= foot_y <= roi[3]):
                continue
            st = self.tracks.get(d.track_id)
            if st is None:
                st = self.tracks[d.track_id] = TrackState(d.track_id, ts)
            st.update(d, ts)
            self._maybe_update_best(st, frame, d)
            eligible = st.is_stopped(self.t["stop_motion_ratio"]) or not self.t["require_stop"]
            if eligible and ts - st.last_try >= self.t["ocr_interval"]:
                if role == "plate" and self.plates and st.ocr_attempts < self.t["max_ocr_attempts"]:
                    st.last_try = ts
                    self._read_plate(st, frame, d)
                elif role == "face" and self.faces and st.face_attempts < self.t["max_face_attempts"]:
                    st.last_try = ts
                    self._find_face(st, frame, d)

        for tid, st in list(self.tracks.items()):
            gone = ts - st.last_seen > self.t["lost_seconds"]
            too_long = ts - st.first_seen > self.t["max_dwell_seconds"]
            if gone or too_long:
                self._finalize(st)
                del self.tracks[tid]

    def flush(self):
        for st in list(self.tracks.values()):
            self._finalize(st)
        self.tracks.clear()

    # ------------------------------------------------------------ helpers
    def _maybe_update_best(self, st: TrackState, frame, det):
        crop = _clip_crop(frame, det.box)
        if crop is None:
            return
        area = crop.shape[0] * crop.shape[1]
        score = _sharpness(crop) * math.sqrt(area) * det.conf
        if score > st.best_score * 1.1:
            st.best_score = score
            st.best_crop = crop.copy()
            st.best_full = frame          # frames are never modified in place

    def _read_plate(self, st: TrackState, frame, det):
        st.ocr_attempts += 1
        crop = _clip_crop(frame, det.box, pad=0.08)
        if crop is None:
            return
        for r in self.plates.read(crop):
            if r.conf < self.t["min_plate_conf"]:
                continue
            st.reads.append(r)
            if r.conf > st.plate_img_conf and r.box:
                bx1, by1, bx2, by2 = r.box
                pc = _clip_crop(crop, (bx1, by1, bx2, by2), pad=0.15)
                if pc is not None:
                    st.plate_img, st.plate_img_conf = pc.copy(), r.conf

    def _find_face(self, st: TrackState, frame, det):
        st.face_attempts += 1
        crop = _clip_crop(frame, det.box)
        if crop is None:
            return
        upper = crop[: max(8, int(crop.shape[0] * 0.55))]
        for (fx, fy, fw, fh, sc) in self.faces.detect(upper):
            fc = _clip_crop(upper, (fx, fy, fx + fw, fy + fh), pad=0.25)
            if fc is None:
                continue
            score = sc * fw * fh * (1.0 + _sharpness(fc) / 100.0)
            if score > st.face_score:
                st.face_score, st.face_img = score, fc.copy()

    def _finalize(self, st: TrackState):
        dur = st.last_seen - st.first_seen
        if dur < self.t["min_track_seconds"] or st.frames < 3 or st.best_crop is None:
            return
        if self.t["require_stop"] and not st.was_stopped:
            return
        label = st.labels.most_common(1)[0][0]
        cam = self.cam
        ev = {
            "ts": st.first_seen + dur / 2,
            "camera_id": cam["id"], "camera_name": cam["name"],
            "gate": cam.get("gate"), "direction": cam["direction"],
            "track_id": st.id, "extra": {"duration_s": round(dur, 1), "frames": st.frames},
        }
        if cam["role"] == "plate":
            plate, conf, best = vote(st.reads, self.t["min_plate_conf"])
            norm = plate or ""
            if norm and self.last_plate.get(norm, -1e9) > ev["ts"] - self.t["dedup_seconds"]:
                self.last_plate[norm] = ev["ts"]
                return                      # same vehicle seen again moments ago
            if norm:
                self.last_plate[norm] = ev["ts"]
            ev.update(kind="vehicle", vehicle_type=label,
                      color=vehicle_color(st.best_crop),
                      plate_text=pretty(plate) if plate else None, plate_conf=conf)
            ev["extra"]["plate_reads"] = len(st.reads)
            ev["extra"]["plate_raw"] = best.raw if best else None
            self.store.save_event(ev, full=st.best_full, crop=st.best_crop, plate=st.plate_img)
        else:
            upper, lower = clothing_colors(st.best_crop)
            ev.update(kind="person", upper_color=upper, lower_color=lower)
            ev["extra"]["has_face"] = st.face_img is not None
            self.store.save_event(ev, full=st.best_full, crop=st.best_crop, face=st.face_img)
        self.events_saved += 1
        self.last_event_ts = ev["ts"]
        log.info("%s: saved %s event (%s)", cam["id"], ev["kind"], ev.get("plate_text") or label)

    # ---------------------------------------------------------- main loop
    def run(self, stream, stop_event, status_db=None):
        interval = 1.0 / max(self.t["process_fps"], 0.5)
        last_idx, last_proc, last_status = -1, 0.0, 0.0
        processed, window_start = 0, time.time()
        fps = 0.0
        while not stop_event.is_set():
            item = stream.read(last_idx)
            now = time.time()
            if item is not None:
                last_idx, frame, ts = item
                if now - last_proc >= interval:
                    last_proc = now
                    try:
                        self.step(frame, ts)
                    except Exception:
                        log.exception("%s: processing error", self.cam["id"])
                    processed += 1
            else:
                stop_event.wait(0.02)
            if status_db and now - last_status >= 10:
                span = max(now - window_start, 1e-6)
                fps, processed, window_start = processed / span, 0, now
                last_status = now
                status_db.update_camera_status(
                    self.cam["id"], self.cam["name"], self.cam["role"], self.cam["direction"],
                    stream.last_frame_time or None, self.last_event_ts, fps)
        self.flush()
