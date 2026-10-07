import tempfile
import unittest
from pathlib import Path

import cv2
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


def box(x, y=300, w=500, h=200):
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

    def test_vehicle_driving_past_without_stopping_is_captured_by_default(self):
        moving = [[Det(2, "car", 0.9, box(50 + 70 * i))] for i in range(14)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(moving), self.store,
                              plates=FakePlates(["ABC123"]))
        self.run_script(worker, 14)
        rows, total = self.db.search({})
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["plate_text"], "ABC-123")

    def test_vehicle_driving_past_is_ignored_when_stop_is_required(self):
        cfg = _merge(DEFAULTS, {"tracking": {"require_stop": True}})
        moving = [[Det(2, "car", 0.9, box(50 + 70 * i))] for i in range(14)]
        worker = CameraWorker(self.cam(), cfg, FakeDetector(moving), self.store,
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
        far = [[Det(1, "car", 0.9, box(400, y=100, h=150))] for _ in range(20)]   # big enough, but outside the ROI
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

    def test_far_away_vehicle_is_ignored(self):
        small = [[Det(1, "car", 0.9, box(500, w=200, h=120))] for _ in range(20)]       # 200/1280 = 16% wide
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(small), self.store, plates=FakePlates(["ABC123"]))
        self.run_script(worker, 20)
        self.assertEqual(self.db.search({})[1], 0)

    def test_far_tree_mistaken_for_person_is_ignored(self):
        tree = [[Det(3, "person", 0.9, (900, 300, 940, 380))] for _ in range(20)]       # 80 px tall = 11%
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(tree), self.store, faces=FakeFaces())
        f = frame()
        for i in range(20):
            worker.step(f, 2000 + i / 6)
        worker.step(f, 2020)
        self.assertEqual(self.db.search({})[1], 0)

    def test_low_confidence_person_is_ignored(self):
        shaky = [[Det(4, "person", 0.40, (380, 300, 540, 640))] for _ in range(20)]
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(shaky), self.store, faces=FakeFaces())
        f = frame()
        for i in range(20):
            worker.step(f, 2000 + i / 6)
        worker.step(f, 2020)
        self.assertEqual(self.db.search({})[1], 0)

    def test_near_person_without_a_face_is_not_saved_when_face_required(self):
        class NoFaces:
            def detect(self, bgr):
                return []
        near = [[Det(5, "person", 0.9, (380, 300, 540, 640))] for _ in range(20)]
        cam_strict = self.cam("face")
        cam_strict["require_face"] = True
        worker = CameraWorker(cam_strict, self.cfg, FakeDetector(near), self.store, faces=NoFaces())
        f = frame()
        for i in range(20):
            worker.step(f, 2000 + i / 6)
        worker.step(f, 2020)
        self.assertEqual(self.db.search({})[1], 0)
        self.assertEqual(worker.stats["no face found"], 1)        # ... and the reason is counted
        # ... but a camera can opt out of the rule
        cam = self.cam("face")
        cam["require_face"] = False
        worker = CameraWorker(cam, self.cfg, FakeDetector(near), self.store, faces=NoFaces())
        for i in range(20):
            worker.step(f, 3000 + i / 6)
        worker.step(f, 3020)
        self.assertEqual(self.db.search({})[1], 1)

    def test_tiny_face_is_ignored(self):
        class TinyFace:
            def detect(self, bgr):
                return [(10.0, 10.0, 30.0, 30.0, 0.95)]      # 30 px wide < 50 px
        near = [[Det(6, "person", 0.9, (380, 300, 540, 640))] for _ in range(20)]
        cam_strict = self.cam("face")
        cam_strict["require_face"] = True
        worker = CameraWorker(cam_strict, self.cfg, FakeDetector(near), self.store, faces=TinyFace())
        f = frame()
        for i in range(20):
            worker.step(f, 2000 + i / 6)
        worker.step(f, 2020)
        self.assertEqual(self.db.search({})[1], 0)

    def test_person_without_face_is_saved_by_default(self):
        class NoFaces:
            def detect(self, bgr):
                return []
        near = [[Det(7, "person", 0.9, (380, 300, 540, 640))] for _ in range(20)]
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(near), self.store, faces=NoFaces())
        f = frame()
        for i in range(20):
            worker.step(f, 2000 + i / 6)
        worker.step(f, 2020)
        self.assertEqual(self.db.search({})[1], 1)
        self.assertEqual(worker.stats["SAVED"], 1)

    def test_far_people_are_counted_not_saved(self):
        far = [[Det(8, "person", 0.9, (380, 300, 400, 340))] for _ in range(10)]     # tiny in the picture
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(far), self.store, faces=None)
        f = frame()
        for i in range(10):
            worker.step(f, 2000 + i / 6)
        self.assertEqual(self.db.search({})[1], 0)
        self.assertEqual(worker.stats["too far / unsure (frames)"], 10)

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


class DuplicateTests(PipelineTests):
    """One record (max 3 photos) per person/vehicle, however often the tracker re-finds them."""

    def _run(self, worker, script, t0=1000.0, fps=10):
        f = frame()
        for i in range(len(script)):
            worker.step(f, t0 + i / fps)
        worker.step(f, t0 + len(script) / fps + 6)

    def person_box(self, x=380):
        return (x, 300, x + 160, 640)

    def test_each_record_has_at_most_three_photos(self):
        stopped = [[Det(1, "car", 0.9, box(400))] for _ in range(30)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(stopped), self.store, plates=FakePlates(["ASY3549"]))
        self._run(worker, stopped)
        row = self.db.search({})[0][0]
        photos = [row[k] for k in ("full_path", "crop_path", "plate_path", "face_path") if row[k]]
        self.assertLessEqual(len(photos), 3)

    def test_person_whose_track_id_switches_is_recorded_once(self):
        a = [[Det(1, "person", 0.9, self.person_box())] for _ in range(25)]
        gap = [[] for _ in range(3)]                       # tracker loses them for 0.3 s ...
        b = [[Det(2, "person", 0.9, self.person_box(385))] for _ in range(25)]   # ... and re-finds at the same spot
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(a + gap + b), self.store, faces=FakeFaces())
        self._run(worker, a + gap + b)
        self.assertEqual(self.db.search({"kind": "person"})[1], 1)

    def test_long_stay_split_into_two_tracks_is_recorded_once(self):
        cfg = _merge(self.cfg, {"tracking": {"max_dwell_seconds": 3}})
        stay = [[Det(1, "car", 0.9, box(400))] for _ in range(80)]       # 8 s > 3 s dwell limit
        worker = CameraWorker(self.cam(), cfg, FakeDetector(stay), self.store, plates=FakePlates(["ABC123", "XYZ999"]))
        self._run(worker, stay)
        self.assertEqual(self.db.search({})[1], 1)

    def test_next_vehicle_arriving_from_the_side_is_still_recorded(self):
        # car A stops at x=400 and leaves; car B enters from the left edge a moment later and stops at the same spot
        class SwitchablePlates:
            text = "AAA111"

            def read(self, crop):
                return [PlateRead(self.text, self.text, 0.9, True, (5, 5, 40, 20))]

        a = [[Det(1, "car", 0.9, box(400))] for _ in range(25)]
        b = [[Det(2, "car", 0.9, box(-100 + 50 * i))] for i in range(10)] + [[Det(2, "car", 0.9, box(400))] for _ in range(25)]
        plates = SwitchablePlates()
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(a + b), self.store, plates=plates)
        f = frame()
        for i in range(len(a)):
            worker.step(f, 1000 + i / 10)
        plates.text = "BBB222"
        for i in range(len(b)):
            worker.step(f, 1000 + (len(a) + i) / 10)
        worker.step(f, 1000 + (len(a) + len(b)) / 10 + 6)
        rows = self.db.search({})[0]
        self.assertEqual(sorted(r["plate_text"] for r in rows), ["AAA-111", "BBB-222"])

    def test_same_spot_different_plate_arriving_instantly_is_not_lost(self):
        # worst case: B appears exactly where A stood, right after A vanished, with a different plate
        class SwitchablePlates:
            text = "AAA111"

            def read(self, crop):
                return [PlateRead(self.text, self.text, 0.9, True, (5, 5, 40, 20))]

        a = [[Det(1, "car", 0.9, box(400))] for _ in range(25)]
        gap = [[] for _ in range(3)]
        b = [[Det(2, "car", 0.9, box(400))] for _ in range(25)]
        plates = SwitchablePlates()
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(a + gap + b), self.store, plates=plates)
        f = frame()
        for i in range(len(a) + len(gap)):
            worker.step(f, 1000 + i / 10)
        plates.text = "BBB222"
        for i in range(len(b)):
            worker.step(f, 1000 + (len(a) + len(gap) + i) / 10)
        worker.step(f, 1000 + (len(a) + len(gap) + len(b)) / 10 + 6)
        # two different plates are two different vehicles, even at the same spot right after each other
        self.assertEqual(self.db.search({})[1], 2)

    def test_different_looking_person_right_behind_the_first_is_also_recorded(self):
        red = np.full((720, 1280, 3), (40, 40, 200), np.uint8)
        red[::7] = (20, 20, 120)
        blue = np.full((720, 1280, 3), (200, 60, 40), np.uint8)
        blue[::7] = (120, 30, 20)
        a = [[Det(1, "person", 0.9, (380, 100, 520, 700))] for _ in range(15)]
        b = [[Det(2, "person", 0.9, (380, 100, 520, 700))] for _ in range(15)]
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(a + [[]] * 3 + b), self.store, faces=FakeFaces())
        t = 3000.0
        for i in range(len(a)):
            worker.step(red, t + i / 10)
        for i in range(3):
            worker.step(red, t + (len(a) + i) / 10)
        for i in range(len(b)):
            worker.step(blue, t + (len(a) + 3 + i) / 10)
        worker.step(blue, t + 40)
        self.assertEqual(self.db.search({})[1], 2)

    def test_narrow_motorcycle_is_accepted_but_a_narrow_car_is_not(self):
        bike = [[Det(1, "motorcycle", 0.9, (500, 300, 640, 640))] for _ in range(20)]     # 140 px = 11% of width
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(bike), self.store, plates=FakePlates(["ABC123"]))
        self.run_script(worker, 20)
        self.assertEqual(self.db.search({})[1], 1)
        car = [[Det(2, "car", 0.9, (500, 300, 640, 640))] for _ in range(20)]
        self.db2 = Database(self.dir / "t2.db")
        worker2 = CameraWorker(self.cam(), self.cfg, FakeDetector(car), EventStore(self.dir, self.db2),
                               plates=FakePlates(["XYZ789"]))
        self.run_script(worker2, 20)
        self.assertEqual(self.db2.search({})[1], 0)
        self.assertGreater(worker2.stats["too far (frames)"], 0)

    def test_saved_face_photo_is_enhanced(self):
        class DarkFaces:
            def detect(self, bgr):
                h, w = bgr.shape[:2]
                return [(w * 0.3, h * 0.05, w * 0.4, h * 0.3, 0.9)]
        dark = np.full((720, 1280, 3), 30, np.uint8)
        dark[::7] = 10
        near = [[Det(9, "person", 0.9, (380, 100, 540, 700))] for _ in range(20)]
        worker = CameraWorker(self.cam("face"), self.cfg, FakeDetector(near), self.store, faces=DarkFaces())
        for i in range(20):
            worker.step(dark, 4000 + i / 10)
        worker.step(dark, 4020)
        row = self.db.search({})[0][0]
        face = cv2.imread(str(self.dir / row["face_path"]))
        self.assertGreater(face.mean(), 60)                 # brightened from ~30
        self.assertGreaterEqual(face.shape[1], 200)         # enlarged


class PlateRobustnessTests(PipelineTests):
    def _run_stop(self, worker, n=30, t0=1000.0):
        f = frame()
        for i in range(n):
            worker.step(f, t0 + i / 10)
        worker.step(f, t0 + n / 10 + 6)

    def _stopped(self, n=30):
        return [[Det(1, "motorcycle", 0.9, box(400, w=500, h=300))] for _ in range(n)]

    def test_plate_photo_is_kept_even_when_text_cannot_be_read(self):
        class DetectsOnly:
            def read(self, img):
                return [PlateRead("", "", 0.0, False, (10, 10, 90, 40), det_conf=0.9)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(self._stopped()), self.store, plates=DetectsOnly())
        self._run_stop(worker)
        row = self.db.search({})[0][0]
        self.assertIsNone(row["plate_text"])
        self.assertTrue(row["plate_path"] and (self.dir / row["plate_path"]).exists())

    def test_uncertain_read_is_saved_with_low_confidence(self):
        class Unsure:
            def read(self, img):
                return [PlateRead("FDX4944", "FDX4944", 0.25, True, (10, 10, 90, 40), det_conf=0.8)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(self._stopped()), self.store, plates=Unsure())
        self._run_stop(worker)
        row = self.db.search({})[0][0]
        self.assertEqual(row["plate_text"], "FDX-4944")
        self.assertLess(row["plate_conf"], 0.5)          # the web page shows it as "FDX-4944?"

    def test_whole_picture_is_tried_when_the_vehicle_crop_gives_nothing(self):
        class FullFrameOnly:
            def read(self, img):
                if img.shape[1] == 1280:                      # only the full frame
                    return [PlateRead("FDX4944", "FDX4944", 0.9, True, (560, 450, 640, 480), det_conf=0.9)]
                return []
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(self._stopped()), self.store, plates=FullFrameOnly())
        self._run_stop(worker)
        row = self.db.search({})[0][0]
        self.assertEqual(row["plate_text"], "FDX-4944")
        self.assertTrue((self.dir / row["plate_path"]).exists())

    def test_colour_cast_variant_neutralises_a_red_scene(self):
        from gatevision.plates import variants
        red = np.zeros((60, 100, 3), np.uint8)
        red[:] = (40, 40, 200)                                # strongly red picture
        red[20:40, 30:70] = (120, 120, 255)
        names = {n: v for n, v in variants(red)}
        self.assertEqual(set(names), {"original", "balanced", "gray"})
        m = names["balanced"].reshape(-1, 3).mean(axis=0)
        self.assertLess(m.max() - m.min(), 25)                # channels now roughly equal


class PlateFinderSafetyTests(PipelineTests):
    def test_box_the_finder_doubts_is_ignored(self):
        class Grass:
            def read(self, img):
                return [PlateRead("AB6425", "AB6425", 0.27, True, (0, 110, 40, 140), det_conf=0.27)]
        stopped = [[Det(1, "motorcycle", 0.9, box(400, w=500, h=300))] for _ in range(30)]
        worker = CameraWorker(self.cam(), self.cfg, FakeDetector(stopped), self.store, plates=Grass())
        f = frame()
        for i in range(30):
            worker.step(f, 1000 + i / 10)
        worker.step(f, 1000 + 9)
        row = self.db.search({})[0][0]
        self.assertIsNone(row["plate_text"])              # "plate not read" is better than a wrong plate
        self.assertIsNone(row["plate_path"])

    def test_reader_falls_back_to_the_small_model_when_the_big_one_is_missing(self):
        import sys
        import types
        calls = []

        class FakeALPR:
            def __init__(self, detector_model, ocr_model, **kw):
                calls.append(detector_model)
                if "608" in detector_model:
                    raise ValueError("unknown model")

        fake = types.ModuleType("fast_alpr")
        fake.ALPR = FakeALPR
        sys.modules["fast_alpr"] = fake
        try:
            from gatevision.plates import FALLBACK_DETECTORS, PlateReader
            reader = PlateReader("yolo-v9-s-608-license-plate-end2end", "cct-xs-v1-global-model")
            self.assertEqual(reader.detector_model, FALLBACK_DETECTORS[0])
            self.assertEqual(calls[0], "yolo-v9-s-608-license-plate-end2end")
        finally:
            del sys.modules["fast_alpr"]
