import tempfile
import unittest
from pathlib import Path

import numpy as np

from gatevision.config import DEFAULTS, _merge
from gatevision.db import Database
from gatevision.detect import Det
from gatevision.pipeline import CameraWorker
from gatevision.plates import PlateRead
from gatevision.storage import EventStore


class FakeDetector:
    """Feeds a scripted list of boxes: one list per frame."""
    def __init__(self, script):
        self.script, self.i = script, 0

    def track(self, frame, role):
        dets = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return dets


class FakePlates:
    def __init__(self, texts):
        self.texts, self.n = texts, 0

    def read(self, crop):
        t = self.texts[self.n % len(self.texts)]
        self.n += 1
        return [PlateRead(t, t, 0.8, True, (10, 10, 60, 30))]


class FakeFaces:
    def detect(self, bgr):
        h, w = bgr.shape[:2]
        return [(w * 0.3, h * 0.05, w * 0.4, h * 0.3, 0.9)]


def frame():
    f = np.full((720, 1280, 3), 200, np.uint8)
    f[::7] = 60                       # texture so sharpness > 0
    return f


def box(x, y=300, w=300, h=200):
    return (x, y, x + w, y + h)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = Database(self.dir / "t.db")
        self.store = EventStore(self.dir, self.db)
        self.cfg = _merge(DEFAULTS, {})

    def tearDown(self):
        self.tmp.cleanup()

    def cam(self, role="plate"):
        return {"id": "c1", "name": "Cam", "gate": "main", "direction": "IN",
                "role": role, "roi": [0, 0, 1, 1]}

    def run_script(self, worker, n_frames, fps=6, t0=1000.0, tail=4.0):
        f = frame()
        for i in range(n_frames):
            worker.step(f, t0 + i / fps)
        worker.step(f, t0 + n_frames / fps + tail)   # long gap => track is finished

    def test_vehicle_that_stops_gives_one_event_with_voted_plate(self):
        approach = [[Det(1, "car", 0.9, box(100 + 60 * i))] for i in range(6)]   # moving
        stopped = [[Det(1, "car", 0.9, box(460))] for _ in range(24)]            # stopped 4 s
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(approach + stopped), self.store,
                              plates=FakePlates(["ASY3549", "ASY3549", "A5Y3549", "ASY3649", "ASY3549"]))
        self.run_script(worker, 30)
        rows, total = self.db.search({})
        self.assertEqual(total, 1)
        r = rows[0]
        self.assertEqual(r["plate_text"], "ASY-3549")
        self.assertEqual((r["kind"], r["direction"], r["vehicle_type"]), ("vehicle", "IN", "car"))
        self.assertTrue((self.dir / r["crop_path"]).exists())
        self.assertTrue((self.dir / r["full_path"]).exists())

    def test_vehicle_driving_past_without_stopping_is_ignored(self):
        moving = [[Det(2, "car", 0.9, box(50 + 70 * i))] for i in range(14)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(moving), self.store,
                              plates=FakePlates(["ABC123"]))
        self.run_script(worker, 14)
        self.assertEqual(self.db.search({})[1], 0)

    def test_same_plate_again_within_dedup_window_is_one_event(self):
        a = [[Det(1, "car", 0.9, box(400))] for _ in range(18)]
        gap = [[] for _ in range(14)]                       # ~2.3 s: track 1 finalised
        b = [[Det(5, "car", 0.9, box(400))] for _ in range(18)]   # new tracker id, same car
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(a + gap + b), self.store,
                              plates=FakePlates(["ASY3549"]))
        f = frame()
        for i in range(18 + 14 + 18):
            worker.step(f, 1000 + i / 6)
        worker.step(f, 1000 + 60 / 6 + 5)
        self.assertEqual(self.db.search({})[1], 1)

    def test_roi_excludes_road_behind_barrier(self):
        cam = self.cam()
        cam["roi"] = [0, 0.6, 1, 1]          # only the lower 40 % of the picture counts
        far = [[Det(1, "car", 0.9, box(400, y=100, h=150))] for _ in range(20)]
        worker = CameraWorker(cam, self.cfg, FakeDetector(far), self.store, plates=FakePlates(["ABC123"]))
        self.run_script(worker, 20)
        self.assertEqual(self.db.search({})[1], 0)

    def test_person_event_has_face_and_clothing_colours(self):
        f = np.zeros((720, 1280, 3), np.uint8)
        f[:] = (200, 200, 200)
        f[360:480, 400:520] = (30, 30, 200)       # red shirt region of the person box
        f[500:620, 400:520] = (150, 60, 20)       # blue trousers
        f[::5] += 3
        person = [[Det(7, "person", 0.9, (380, 300, 540, 640))] for _ in range(20)]
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(person), self.store,
                              faces=FakeFaces())
        for i in range(20):
            worker.step(f, 2000 + i / 6)
        worker.step(f, 2020)
        rows, total = self.db.search({"kind": "person"})
        self.assertEqual(total, 1)
        r = rows[0]
        self.assertTrue(r["face_path"] and (self.dir / r["face_path"]).exists())
        self.assertEqual((r["upper_color"], r["lower_color"]), ("red", "blue"))

    def test_flush_saves_tracks_still_open_at_shutdown(self):
        a = [[Det(1, "car", 0.9, box(400))] for _ in range(18)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(a), self.store, plates=FakePlates(["ASY3549"]))
        f = frame()
        for i in range(18):
            worker.step(f, 1000 + i / 6)
        worker.flush()
        self.assertEqual(self.db.search({})[1], 1)


if __name__ == "__main__":
    unittest.main()
