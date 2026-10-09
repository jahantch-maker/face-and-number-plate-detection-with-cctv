import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from werkzeug.security import generate_password_hash

from gatevision.config import DEFAULTS, _merge
from gatevision.db import Database
from gatevision.storage import EventStore
from gatevision.web.app import create_app


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.cfg = _merge(DEFAULTS, {"storage": {"data_dir": str(d)},
                                     "cameras": [{"id": "c1", "name": "RG Barrier IN 1", "role": "plate",
                                                  "direction": "IN", "url": "x", "gate": "main", "roi": [0, 0, 1, 1]}]})
        self.db = Database(d / "gatevision.db")
        self.store = EventStore(d, self.db)
        img = np.full((120, 200, 3), 230, np.uint8)
        now = time.time()
        self.id1 = self.store.save_event(dict(ts=now - 30, camera_id="c1", camera_name="RG Barrier IN 1", direction="IN",
                                              kind="vehicle", plate_text="ASY-3549", plate_conf=0.9,
                                              vehicle_type="car", color="white"), full=img, crop=img, plate=img)
        self.store.save_event(dict(ts=now - 10, camera_id="c1", camera_name="RG Barrier IN 1", direction="IN",
                                   kind="person", upper_color="red", lower_color="blue"), crop=img, face=img)
        for name, role in (("boss", "manager"), ("guard1", "guard")):
            self.db.add_user(name, generate_password_hash("secret-pass-1"), role)
        self.app = create_app(self.cfg, self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def login(self, client, user, pw="secret-pass-1"):
        client.get("/login")
        with client.session_transaction() as s:
            token = s.setdefault("csrf", "tok")
        return client.post("/login", data={"username": user, "password": pw, "csrf": token})

    def test_requires_login(self):
        c = self.app.test_client()
        self.assertEqual(c.get("/").status_code, 302)
        self.assertEqual(c.get("/api/events").status_code, 401)
        self.assertEqual(c.get(f"/media/images/x.jpg").status_code, 302)

    def test_wrong_password_and_lockout(self):
        c = self.app.test_client()
        for _ in range(5):
            r = self.login(c, "boss", "nope")
            self.assertIn(b"Wrong username", r.data)
        self.assertIn(b"Too many attempts", self.login(c, "boss").data)

    def test_manager_search_filters_and_pages(self):
        c = self.app.test_client()
        self.assertEqual(self.login(c, "boss").status_code, 302)
        html = c.get("/").get_data(as_text=True)
        self.assertIn("ASY-3549", html)
        self.assertIn("2</strong> results", html)
        html = c.get("/?submitted=1&plate=3549&kind=&ts_from=&ts_to=").get_data(as_text=True)
        self.assertIn("1</strong> result", html)
        self.assertIn("ASY-3549", html)
        html = c.get("/?submitted=1&kind=person&upper_color=red&ts_from=&ts_to=").get_data(as_text=True)
        self.assertIn("1</strong> result", html)
        self.assertNotIn("ASY-3549", html)
        html = c.get("/?submitted=1&plate=A5Y3549&fuzzy=1&kind=&ts_from=&ts_to=").get_data(as_text=True)
        self.assertIn("ASY-3549", html)

    def test_event_detail_and_media(self):
        c = self.app.test_client()
        self.login(c, "boss")
        r = c.get(f"/event/{self.id1}")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"ASY-3549", r.data)
        row = self.db.get_event(self.id1)
        img = c.get("/media/" + row["crop_path"])
        self.assertEqual(img.status_code, 200)
        self.assertEqual(img.mimetype, "image/jpeg")
        self.assertEqual(c.get("/media/../gatevision.db").status_code, 404)
        self.assertEqual(c.get("/event/9999").status_code, 404)

    def test_event_detail_shows_unsure_plate_and_guess(self):
        img = np.full((120, 200, 3), 230, np.uint8)
        unsure = self.store.save_event(dict(ts=time.time() - 5, camera_id="c1", camera_name="RG Barrier IN 1",
                                            direction="IN", kind="vehicle", plate_text="FDX-4944", plate_conf=0.3),
                                       crop=img, plate=img)
        guessed = self.store.save_event(dict(ts=time.time() - 4, camera_id="c1", camera_name="RG Barrier IN 1",
                                             direction="IN", kind="vehicle", extra={"plate_guess": "UCC-433"}),
                                        crop=img, plate=img)
        c = self.app.test_client()
        self.login(c, "boss")
        html = c.get(f"/event/{unsure}").get_data(as_text=True)
        self.assertIn("FDX-4944?", html)
        self.assertIn("Uncertain", html)
        html = c.get(f"/event/{guessed}").get_data(as_text=True)
        self.assertIn("plate not read", html)
        self.assertIn("UCC-433?", html)

    def test_guard_sees_live_only(self):
        c = self.app.test_client()
        self.login(c, "guard1")
        self.assertEqual(c.get("/").status_code, 302)
        self.assertTrue(c.get("/").headers["Location"].endswith("/live"))
        self.assertEqual(c.get("/live").status_code, 200)
        self.assertEqual(c.get(f"/event/{self.id1}").status_code, 302)
        data = c.get("/api/events?after_id=0").get_json()
        self.assertEqual(len(data["events"]), 2)
        newest = data["events"][0]["id"]
        self.assertEqual(c.get(f"/api/events?after_id={newest}").get_json()["events"], [])

    def test_status_page(self):
        self.db.update_camera_status("c1", "RG Barrier IN 1", "plate", "IN", time.time(), None, 5.5)
        c = self.app.test_client()
        self.login(c, "boss")
        html = c.get("/status").get_data(as_text=True)
        self.assertIn("online", html)
        self.assertIn("5.5", html)

    def test_status_page_shows_why_records_were_skipped(self):
        import json
        skips = json.dumps({"since": time.time(), "counts": {"no face found": 7, "SAVED": 2}})
        self.db.update_camera_status("c1", "RG Barrier IN 1", "plate", "IN", time.time(), None, 5.5, skips=skips)
        c = self.app.test_client()
        self.login(c, "boss")
        html = c.get("/status").get_data(as_text=True)
        self.assertIn("SAVED: <b>2</b>", html)
        self.assertIn("no face found: <b>7</b>", html)

    def test_logout_needs_csrf(self):
        c = self.app.test_client()
        self.login(c, "boss")
        self.assertEqual(c.post("/logout", data={"csrf": "wrong"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
