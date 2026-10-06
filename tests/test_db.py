import tempfile
import time
import unittest
from pathlib import Path

from gatevision.db import Database, canon_plate, edit_distance_le1


def ev(db, plate, ts, **kw):
    base = dict(ts=ts, camera_id="c1", camera_name="Cam 1", direction="IN", kind="vehicle",
                plate_text=plate, color="white", vehicle_type="car")
    base.update(kw)
    return db.insert_event(base)


class DbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = Database(self.dir / "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_edit_distance(self):
        self.assertTrue(edit_distance_le1("ASY3549", "ASY3549"))
        self.assertTrue(edit_distance_le1("ASY3549", "ASY354"))
        self.assertTrue(edit_distance_le1("ASY3549", "ASY3649"))
        self.assertFalse(edit_distance_le1("ASY3549", "ASY3999"))
        self.assertEqual(canon_plate("A5Y-3549"), canon_plate("ASY3549"))

    def test_search_partial_filters_and_paging(self):
        now = time.time()
        ev(self.db, "ASY-3549", now - 60, color="white")
        ev(self.db, "ARG-808", now - 30, color="black", direction="OUT")
        ev(self.db, None, now - 10, kind="person", upper_color="red", lower_color="blue")
        rows, total = self.db.search({"plate": "3549"})
        self.assertEqual([r["plate_text"] for r in rows], ["ASY-3549"])
        self.assertEqual(self.db.search({"plate": "asy 3549"})[1], 1)
        self.assertEqual(self.db.search({"color": "black"})[1], 1)
        self.assertEqual(self.db.search({"direction": "OUT"})[1], 1)
        self.assertEqual(self.db.search({"kind": "person", "upper_color": "red"})[1], 1)
        self.assertEqual(self.db.search({"ts_from": now - 40})[1], 2)
        rows, total = self.db.search({}, limit=2, offset=0)
        self.assertEqual((len(rows), total), (2, 3))
        self.assertEqual(rows[0]["kind"], "person")          # newest first

    def test_fuzzy_search_tolerates_ocr_errors(self):
        now = time.time()
        ev(self.db, "ASY-3S49", now)       # OCR read S for 5
        ev(self.db, "XYZ-111", now)
        rows, total = self.db.search({"plate": "ASY3549", "fuzzy": True})
        self.assertEqual(total, 1)
        self.assertEqual(self.db.search({"plate": "ASY3549", "color": "white", "fuzzy": True})[1], 1)
        self.assertEqual(self.db.search({"plate": "ASY3549", "color": "red", "fuzzy": True})[1], 0)

    def test_retention_removes_rows_and_files(self):
        now = time.time()
        old_img = self.dir / "images" / "old" / "c1"
        old_img.mkdir(parents=True)
        (old_img / "a.jpg").write_bytes(b"x")
        ev(self.db, "OLD-1", now - 40 * 86400, crop_path="images/old/c1/a.jpg")
        ev(self.db, "NEW-1", now - 86400)
        removed = self.db.delete_older_than(30, self.dir, now=now)
        self.assertEqual(removed, 1)
        self.assertEqual(self.db.search({})[1], 1)
        self.assertFalse((old_img / "a.jpg").exists())
        self.assertFalse((self.dir / "images" / "old").exists())   # empty folders pruned

    def test_users(self):
        self.db.add_user("bob", "hash", "manager")
        self.assertEqual(self.db.get_user("bob")["role"], "manager")
        self.assertTrue(self.db.delete_user("bob"))
        self.assertIsNone(self.db.get_user("bob"))


if __name__ == "__main__":
    unittest.main()
