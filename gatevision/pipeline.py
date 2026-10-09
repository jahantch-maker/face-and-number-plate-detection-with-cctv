"""Per-camera processing: track objects, wait for them to stop, read plate /
face, vote over many frames, and write ONE clean event per vehicle / person."""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, deque

import cv2

from .colors import clothing_colors, vehicle_color
from .db import _fuzzy_match, canon_plate, edit_distance_le1, norm_plate
from .enhance import enhance_face, enhance_plate
from .plates import pretty, vote

log = logging.getLogger("gatevision.pipeline")


def _sharpness(bgr) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if w > 200:
        g = cv2.resize(g, (200, max(1, int(h * 200 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _signature(crop):
    """Tiny colour fingerprint of a person / vehicle, to tell two different ones apart."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=1.0, norm_type=cv2.NORM_L1)
    return h


def _brightness_ok(bgr) -> float:
    """1.0 for a well-lit picture, down to 0.3 for a very dark or washed-out one."""
    mean = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())
    return max(0.3, 1.0 - abs(mean - 120.0) / 160.0)


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
        self.first_pos = None                 # (cx, cy, diag) where it first appeared
        self.last_pos = None
        self.was_stopped = False
        self.best_score = -1.0
        self.best_crop = None
        self.best_full = None
        self.reads: list = []
        self.plate_img = None
        self.plate_img_conf = 0.0
        self.plate_full = None                # the picture in which the plate was clearest ...
        self.plate_crop = None                # ... and the vehicle in it: the photos saved for a vehicle
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
        pos = ((x1 + x2) / 2, (y1 + y2) / 2, math.hypot(x2 - x1, y2 - y1))
        self.hist.append((ts, *pos))
        if self.first_pos is None:
            self.first_pos = pos
        self.last_pos = pos

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
        self.last_plate: dict[str, tuple] = {}   # plate_norm -> (ts of last event, event id, conf)
        self.last_event_ts: float | None = None
        self.events_saved = 0
        self.recent: list = []                    # (last_seen, last_pos) of recently recorded tracks
        self.stats: Counter = Counter()           # why things were saved / skipped (shown on the Cameras page)
        self.started = time.time()
        # Background plate reading (switched on by run()). Reading one plate runs
        # the reader many times and is slow on a CPU; doing it here would stop
        # this camera from watching while vehicles drive past. Instead the newest
        # sharpest crop of each vehicle waits in a slot and a helper thread reads it.
        self._ocr_thread = None
        self._ocr_lock = threading.Lock()
        self._ocr_wake = threading.Event()
        self._ocr_pending: dict = {}             # track id -> (crop, frame, box, sharpness)
        self._ocr_busy: set = set()              # track ids being read right now
        self._ocr_done: deque = deque()          # (track id, [(reads, image), ...]) ready to apply

    # --------------------------------------------------------------- step
    def step(self, frame, ts: float):
        H, W = frame.shape[:2]
        rx1, ry1, rx2, ry2 = self.cam["roi"]
        roi = (rx1 * W, ry1 * H, rx2 * W, ry2 * H)
        role = self.cam["role"]
        self._apply_ocr_results()

        cam = self.cam
        min_h = cam.get("min_height", self.t["min_person_height"])
        min_w = cam.get("min_width", self.t["min_vehicle_width"])
        person_conf = cam.get("person_conf", self.t["person_conf"])

        for d in self.detector.track(frame, role):
            x1, y1, x2, y2 = d.box
            foot_x, foot_y = (x1 + x2) / 2, y2
            if not (roi[0] <= foot_x <= roi[2] and roi[1] <= foot_y <= roi[3]):
                continue
            # only things CLOSE to the camera: big enough in the picture
            if role == "face" and ((y2 - y1) / H < min_h or d.conf < person_conf):
                self.stats["too far / unsure (frames)"] += 1
                continue
            if role == "plate":
                is_bike = d.label in ("motorcycle", "bicycle")
                need = cam.get("min_bike_width", self.t["min_bike_width"]) if is_bike else min_w
                if (x2 - x1) / W < need:
                    self.stats["too far (frames)"] += 1
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
                    if self._ocr_thread is not None:
                        self._queue_plate(st, frame, d)
                    else:
                        self._read_plate(st, frame, d)
                elif role == "face" and self.faces and st.face_attempts < self.t["max_face_attempts"]:
                    st.last_try = ts
                    self._find_face(st, frame, d)

        for tid, st in list(self.tracks.items()):
            gone = ts - st.last_seen > self.t["lost_seconds"]
            too_long = ts - st.first_seen > self.t["max_dwell_seconds"]
            if gone and self._ocr_waiting(tid) and ts - st.last_seen < self.t["lost_seconds"] + 5:
                continue                      # its last plate pictures are still being read
            if gone or too_long:
                self._finalize(st)
                del self.tracks[tid]

    def flush(self):
        self._apply_ocr_results()
        for st in list(self.tracks.values()):
            self._finalize(st)
        self.tracks.clear()

    # ------------------------------------------------------------ helpers
    def _maybe_update_best(self, st: TrackState, frame, det):
        crop = _clip_crop(frame, det.box)
        if crop is None:
            return
        area = crop.shape[0] * crop.shape[1]
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = det.box
        edge = 0.02
        inside = 1.0 if (x1 > edge * W and y1 > edge * H and x2 < (1 - edge) * W and y2 < (1 - edge) * H) else 0.6
        # clearer, bigger, fully in the picture, well lit, and a sure detection
        score = (1.0 + min(_sharpness(crop), 400.0) / 100.0) * math.sqrt(area) * det.conf * inside * _brightness_ok(crop)
        if score > st.best_score * 1.1:
            st.best_score = score
            st.best_crop = crop.copy()
            st.best_full = frame          # frames are never modified in place

    def _read_plate(self, st: TrackState, frame, det):
        st.ocr_attempts += 1
        crop = _clip_crop(frame, det.box, pad=0.08)
        if crop is None:
            return
        whole = not any(r.text for r in st.reads) and st.ocr_attempts % 3 == 0
        for reads, image in self._plate_reads(crop, frame, det.box, whole):
            self._take_reads(st, reads, image, scene=(frame, crop))

    def _plate_reads(self, crop, frame, box, whole: bool):
        """Read one vehicle crop. ``whole``: also look at the whole picture, for
        when the crop has given no readable text so far. Returns [(reads, image)]."""
        read = getattr(self.plates, "read_robust", self.plates.read)
        out = [(read(crop), crop)]
        if whole and not any(r.text for r in out[0][0]):
            x1, y1, x2, y2 = box
            gx, gy = (x2 - x1) * 0.15, (y2 - y1) * 0.15
            near = [r for r in read(frame)
                    if r.box and x1 - gx <= (r.box[0] + r.box[2]) / 2 <= x2 + gx
                    and y1 - gy <= (r.box[1] + r.box[3]) / 2 <= y2 + gy]
            out.append((near, frame))
        return out

    # ------------------------------------------------- background plate reading
    def start_ocr(self):
        if self._ocr_thread is None and self.plates is not None and self.cam["role"] == "plate":
            self._ocr_thread = threading.Thread(target=self._ocr_loop, name=f"ocr-{self.cam['id']}", daemon=True)
            self._ocr_thread.start()

    def _queue_plate(self, st: TrackState, frame, det):
        crop = _clip_crop(frame, det.box, pad=0.08)
        if crop is None:
            return
        sharp = _sharpness(crop) * math.sqrt(crop.shape[0] * crop.shape[1])
        with self._ocr_lock:
            old = self._ocr_pending.get(st.id)
            if old is None:
                st.ocr_attempts += 1
            elif old[3] >= sharp:
                return                        # the waiting picture of this vehicle is clearer
            self._ocr_pending[st.id] = (crop, frame, det.box, sharp)
        self._ocr_wake.set()

    def _ocr_waiting(self, track_id) -> bool:
        if self._ocr_thread is None:
            return False
        with self._ocr_lock:
            return track_id in self._ocr_pending or track_id in self._ocr_busy

    def _ocr_loop(self):
        while True:
            self._ocr_wake.wait(0.5)
            with self._ocr_lock:
                if not self._ocr_pending:
                    self._ocr_wake.clear()
                    continue
                tid = next(iter(self._ocr_pending))       # oldest waiting vehicle first
                crop, frame, box, _ = self._ocr_pending.pop(tid)
                self._ocr_busy.add(tid)
                st = self.tracks.get(tid)
                whole = st is not None and not any(r.text for r in st.reads) and st.ocr_attempts % 3 == 0
            try:
                result = self._plate_reads(crop, frame, box, whole)
            except Exception:
                log.exception("%s: plate reading error", self.cam["id"])
                result = []
            with self._ocr_lock:
                self._ocr_done.append((tid, result, (frame, crop)))
                self._ocr_busy.discard(tid)

    def _apply_ocr_results(self):
        while self._ocr_done:
            tid, result, scene = self._ocr_done.popleft()
            st = self.tracks.get(tid)
            if st is not None:
                for reads, image in result:
                    self._take_reads(st, reads, image, scene=scene)

    def _take_reads(self, st: TrackState, reads, image, scene=None):
        """Keep readable texts for voting, and the best plate PHOTO even if the
        text could not be read (so a human can still read it). ``scene`` is the
        (full picture, vehicle crop) the reads came from: the vehicle photos are
        taken from the moment its plate was clearest, when it faced the camera,
        not from when it was biggest (often already driving past the camera)."""
        for r in reads:
            if r.det_conf and r.det_conf < 0.35:
                continue                      # the finder is not convinced this is a plate (grass, sticker ...)
            if r.box:
                w, h = r.box[2] - r.box[0], r.box[3] - r.box[1]
                if h <= 0 or not (0.9 <= w / h <= 6.5):
                    continue                  # plates are rectangles; leaves and bushes are not
            if r.text and not r.valid and r.conf < 0.7:
                continue                      # text that does not look like a plate and is not a sure read
            if r.text and r.conf >= self.t["min_plate_conf"]:
                st.reads.append(r)
            if r.box:
                pc = _clip_crop(image, r.box, pad=0.15)
                if pc is not None:
                    # best plate photo: sure it is a plate, sure of the text, big and sharp
                    sure = (0.4 + r.det_conf) * (0.5 + r.conf)
                    quality = sure * math.sqrt(pc.shape[0] * pc.shape[1]) * (1.0 + min(_sharpness(pc), 300.0) / 100.0)
                    if quality > st.plate_img_conf:
                        st.plate_img, st.plate_img_conf = pc.copy(), quality
                        if scene is not None:
                            st.plate_full, st.plate_crop = scene

    def _find_face(self, st: TrackState, frame, det):
        st.face_attempts += 1
        crop = _clip_crop(frame, det.box)
        if crop is None:
            return
        upper = crop[: max(8, int(crop.shape[0] * 0.55))]
        min_px = self.cam.get("min_face_px", self.t["min_face_px"])
        for (fx, fy, fw, fh, sc) in self.faces.detect(upper):
            if fw < min_px:
                continue
            fc = _clip_crop(upper, (fx, fy, fx + fw, fy + fh), pad=0.25)
            if fc is None:
                continue
            score = sc * fw * fh * (1.0 + min(_sharpness(fc), 400.0) / 100.0) * _brightness_ok(fc)
            if score > st.face_score:
                st.face_score, st.face_img = score, fc.copy()

    def _same_plate_recently(self, norm: str, conf: float, ts: float) -> bool:
        """True if this plate, or one differing by a single character (DN6555 /
        FDN6555, a letter lost at the picture edge), was recorded by this camera
        within ``dedup_seconds``. The earlier record then gets the better text:
        the longer one, or the surer one when both are the same length."""
        window = self.t["dedup_seconds"]
        self.last_plate = {k: v for k, v in self.last_plate.items() if ts - v[0] < max(window, 60)}
        canon = canon_plate(norm)
        for old, (old_ts, event_id, old_conf) in list(self.last_plate.items()):
            if ts - old_ts > window or not edit_distance_le1(canon, canon_plate(old)):
                continue
            better = len(norm) > len(old) or (len(norm) == len(old) and conf > old_conf)
            if better and norm != old and event_id is not None:
                self.store.db.update_event_plate(event_id, pretty(norm), conf)
                del self.last_plate[old]
                self.last_plate[norm] = (ts, event_id, conf)
            else:
                self.last_plate[old] = (ts, event_id, old_conf)
            return True
        return False

    def _is_retrack(self, st: TrackState, sig=None, plate: str = "") -> bool:
        """True if ``st`` is the SAME object as a track recorded moments ago.

        That happens when the tracker loses and re-finds someone who is standing
        still, or when a long stay is split. It is the same object only when the
        new track appears right where the old one ended, within a few seconds,
        LOOKS alike (colour, size) and does not have a different plate. A second
        vehicle or person right behind the first one fails these tests, so it
        is recorded too.
        """
        window = self.t["retrack_seconds"]
        self.recent = [r for r in self.recent if st.first_seen - r[0] < 60]
        for last_seen, pos, old_sig, old_plate in self.recent:
            gap = st.first_seen - last_seen
            if not (0 <= gap <= window and st.first_pos and pos):
                continue
            size = (st.first_pos[2] + pos[2]) / 2
            dist = math.hypot(st.first_pos[0] - pos[0], st.first_pos[1] - pos[1])
            if dist >= 0.30 * size:
                continue
            if plate and old_plate and not _fuzzy_match(norm_plate(plate), norm_plate(old_plate)):
                continue                                    # two different plates = two vehicles
            if sig is not None and old_sig is not None and \
                    cv2.compareHist(sig, old_sig, cv2.HISTCMP_CORREL) < 0.5:
                continue                                    # different colours = a different one
            ratio = st.first_pos[2] / max(pos[2], 1e-6)
            if not (0.6 <= ratio <= 1.7):
                continue                                    # clearly a different size
            return True
        return False

    def _finalize(self, st: TrackState):
        dur = st.last_seen - st.first_seen
        # A quick pass is still a real vehicle / person when a plate or face was
        # found on it: at the 4-5 pictures a second a CPU manages, a bike crossing
        # the speed breaker can be close enough for only one or two of them.
        # Without such proof, short tracks are flickers (shadows, trees) and dropped.
        proof = bool(st.reads) or st.plate_img is not None or st.face_img is not None
        if st.best_crop is None or (not proof and (dur < self.t["min_track_seconds"] or st.frames < 3)):
            self.stats["too short a visit"] += 1
            return
        if self.t["require_stop"] and not st.was_stopped:
            self.stats["did not stop"] += 1
            return
        sig = _signature(st.best_crop)
        plate, share, best = vote(st.reads, self.t["min_plate_conf"]) if self.cam["role"] == "plate" else (None, 0.0, None)
        if self._is_retrack(st, sig, plate or ""):
            self.stats["same one seen again"] += 1
            self.recent.append((st.last_seen, st.last_pos, sig, plate or ""))   # chain: later re-tracks match this one too
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
            conf = best.conf * share if best else 0.0          # how sure we are of the final text
            agree = sum(1 for r in st.reads if r.text == plate)
            if plate and (conf < self.t["min_show_conf"] or (agree < 2 and conf < self.t["min_single_read_conf"])):
                # one weak read, or readers that disagree: a made-up number would
                # only mislead the search. Keep the guess for the detail page.
                ev["extra"]["plate_guess"] = pretty(plate)
                self.stats["plate too unsure"] += 1
                plate, conf = None, 0.0
            norm = plate or ""
            if norm and self._same_plate_recently(norm, conf, ev["ts"]):
                self.stats["same plate again"] += 1
                return                      # same vehicle seen again moments ago
            ev.update(kind="vehicle", vehicle_type=label,
                      color=vehicle_color(st.best_crop),
                      plate_text=pretty(plate) if plate else None, plate_conf=conf)
            ev["extra"]["plate_reads"] = len(st.reads)
            ev["extra"]["plate_raw"] = best.raw if best else None
            plate_photo = enhance_plate(st.plate_img) if (st.plate_img is not None and self.cfg.get("enhance", {}).get("plates", True)) else st.plate_img
            full, crop = (st.plate_full, st.plate_crop) if st.plate_crop is not None else (st.best_full, st.best_crop)
            event_id = self.store.save_event(ev, full=full, crop=crop, plate=plate_photo)
            if norm:
                self.last_plate[norm] = (ev["ts"], event_id, conf)
        else:
            if self.cam.get("require_face", self.t["require_face"]) and st.face_img is None:
                self.stats["no face found"] += 1
                return                      # no face found: tree, shadow or someone far/turned away
            upper, lower = clothing_colors(st.best_crop)
            ev.update(kind="person", upper_color=upper, lower_color=lower)
            ev["extra"]["has_face"] = st.face_img is not None
            face_photo = enhance_face(st.face_img) if (st.face_img is not None and self.cfg.get("enhance", {}).get("faces", True)) else st.face_img
            self.store.save_event(ev, full=st.best_full, crop=st.best_crop, face=face_photo)
        self.recent.append((st.last_seen, st.last_pos, sig, plate or ""))
        self.events_saved += 1
        self.stats["SAVED"] += 1
        self.last_event_ts = ev["ts"]
        log.info("%s: saved %s event (%s)", cam["id"], ev["kind"], ev.get("plate_text") or label)

    # ---------------------------------------------------------- main loop
    def run(self, stream, stop_event, status_db=None):
        self.start_ocr()
        pf = self.t["process_fps"]
        interval = 0.0 if pf <= 0 else 1.0 / max(pf, 0.5)      # 0 = as fast as the PC can
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
                    stream.last_frame_time or None, self.last_event_ts, fps,
                    skips=json.dumps({"since": self.started, "counts": dict(self.stats)}))
        self.flush()
