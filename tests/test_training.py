import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import common  # noqa: E402
from collect_plates import unpad  # noqa: E402
from convert_char_boxes import plate_text  # noqa: E402


def box(ch, x, y, w=10, h=20):
    return (ch, x, y, x + w, y + h)


class LabelTests(unittest.TestCase):
    def test_clean_label(self):
        self.assertEqual(common.clean_label(" lea-5989 "), "LEA5989")
        self.assertEqual(common.clean_label("x"), common.UNREADABLE)
        self.assertEqual(common.clean_label("-"), common.UNREADABLE)
        self.assertEqual(common.clean_label("!"), common.NOT_PLATE)

    def test_only_typed_plates_count_as_text(self):
        self.assertTrue(common.has_text("LEA5989"))
        for label in ("", common.UNREADABLE, common.NOT_PLATE):
            self.assertFalse(common.has_text(label))

    def test_labels_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "labels.csv"
            rows = [{"file": "1.jpg", "guess": "ABC1", "label": "ABC12", "vehicle": "motorcycle", "camera": "in1",
                     "time": "2026-10-09 08:00:00"}]
            common.write_labels(p, rows)
            self.assertEqual(common.read_labels(p), rows)


class UnpadTests(unittest.TestCase):
    def test_unpad_keeps_the_plate_and_a_small_margin(self):
        plate_w, plate_h = 200, 60
        img = np.zeros((int(plate_h * 1.3), int(plate_w * 1.3), 3), np.uint8)
        out = unpad(img)
        self.assertGreater(out.shape[1], plate_w)
        self.assertLess(out.shape[1], img.shape[1])


class CharBoxTests(unittest.TestCase):
    def test_one_line_left_to_right(self):
        chars = [box("3", 50, 0), box("L", 0, 1), box("E", 12, 0), box("A", 24, 2), box("5", 40, 1)]
        self.assertEqual(plate_text(chars), "LEA53")

    def test_two_line_bike_plate_top_row_first(self):
        top = [box("M", 10, 0), box("N", 22, 1), box("C", 34, 0), box("1", 50, 2, 6, 10), box("7", 57, 2, 6, 10)]
        bottom = [box("4", 8, 30), box("2", 20, 31), box("1", 32, 30)]
        self.assertEqual(plate_text(bottom + top), "MNC17421")


class DatasetTests(unittest.TestCase):
    def test_same_plate_never_in_both_lists(self):
        import make_dataset
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "pub" / "images").mkdir(parents=True)
            with (d / "pub" / "labels.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_path", "plate_text"])
                for i in range(300):
                    w.writerow([f"images/{i}.jpg", f"ABC{i % 60}"])
            argv = ["make_dataset.py", "--public", str(d / "pub" / "labels.csv"), "--out", str(d / "ds")]
            with mock.patch.object(sys, "argv", argv):
                make_dataset.main()

            def plates(name):
                with (d / "ds" / name).open() as f:
                    return {r["plate_text"] for r in csv.DictReader(f)}
            self.assertTrue(plates("val.csv"))
            self.assertFalse(plates("train.csv") & plates("val.csv"))
            with (d / "ds" / "train.csv").open() as f:
                first = next(csv.DictReader(f))["image_path"]
            self.assertTrue((d / "ds" / first).resolve().parent == (d / "pub" / "images").resolve())


class CustomReaderTests(unittest.TestCase):
    def test_trained_reader_used_only_when_model_and_settings_exist(self):
        from gatevision.plates import _custom_ocr_files
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "pk_plate_ocr.onnx"
            self.assertIsNone(_custom_ocr_files(str(model)))
            model.write_bytes(b"x")
            self.assertIsNone(_custom_ocr_files(str(model)))      # settings file missing
            (Path(d) / "pk_plate_ocr_config.yaml").write_text("x")
            self.assertEqual(_custom_ocr_files(str(model))[0], model)
        self.assertIsNone(_custom_ocr_files(None))


if __name__ == "__main__":
    unittest.main()
