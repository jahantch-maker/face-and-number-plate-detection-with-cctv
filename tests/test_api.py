import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from werkzeug.security import generate_password_hash

from gatevision.config import DEFAULTS, _merge
from gatevision.db import Database
from gatevision.storage import EventStore
from gatevision.web.app import create_app


class MobileApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        cam = {"id": "c1", "name": "Gate IN", "role": "plate", "direction": "IN",
               "url": "x", "gate": "main", "roi": [0, 0, 1, 1]}
        self.cfg = _merge(DEFAULTS, {"storage": {"data_dir": str(d)}, "cameras": [cam]})
        self.db = Database(d / "gatevision.db")
        store = EventStore(d, self.db)
        img = np.full((60, 100, 3), 200, np.uint8)
        now = time.time()
        self.id1 = store.save_event(dict(ts=now - 30, camera_id="c1", camera_name="Gate IN", direction="IN",
                                         kind="vehicle", plate_text="ASY-3549", plate_conf=0.9,
                                         vehicle_type="car", color="white"), full=img, crop=img, plate=img)
        self.id2 = store.save_event(dict(ts=now - 10, camera_id="c1", camera_name="Gate IN", direction="IN",
                                         kind="person", upper_color="red", lower_color="blue"),
                                    crop=img, face=img)
        self.db.add_user("boss", generate_password_hash("secret-pass-1"), "manager")
        self.db.add_user("guard1", generate_password_hash("secret-pass-1"), "guard")
        self.app = create_app(self.cfg, self.db)
        self.c = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def token(self, user="boss", pw="secret-pass-1"):
        r = self.c.post("/api/v1/login", json={"username": user, "password": pw})
        return r, (r.get_json() or {}).get("token")

    def auth(self, tok):
        return {"Authorization": f"Bearer {tok}"}

    def test_login_ok_and_bad(self):
        r, tok = self.token()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["role"], "manager")
        self.assertTrue(tok)
        self.assertEqual(self.token("boss", "wrong")[0].status_code, 401)

    def test_lockout_after_five_failures(self):
        for _ in range(5):
            self.token("boss", "wrong")
        self.assertEqual(self.token()[0].status_code, 429)

    def test_requires_token(self):
        self.assertEqual(self.c.get("/api/v1/events").status_code, 401)
        self.assertEqual(self.c.get("/api/v1/events", headers=self.auth("junk")).status_code, 401)
        self.assertEqual(self.c.get("/media/images/x.jpg").status_code, 302)

    def test_live_and_after_id(self):
        _, tok = self.token()
        ev = self.c.get("/api/v1/events", headers=self.auth(tok)).get_json()["events"]
        self.assertEqual([e["id"] for e in ev], [self.id2, self.id1])
        new = self.c.get(f"/api/v1/events?after_id={self.id1}", headers=self.auth(tok)).get_json()["events"]
        self.assertEqual([e["id"] for e in new], [self.id2])
        self.assertTrue(ev[1]["plate_path"].startswith("/media/"))

    def test_media_with_token(self):
        _, tok = self.token()
        ev = self.c.get("/api/v1/events", headers=self.auth(tok)).get_json()["events"]
        url = [e for e in ev if e["kind"] == "vehicle"][0]["plate_path"]
        self.assertEqual(self.c.get(url, headers=self.auth(tok)).status_code, 200)

    def test_search_filters(self):
        _, tok = self.token()
        h = self.auth(tok)
        r = self.c.get("/api/v1/search?plate=3549", headers=h).get_json()
        self.assertEqual(r["total"], 1)
        r = self.c.get("/api/v1/search?plate=A5Y3549&fuzzy=1", headers=h).get_json()
        self.assertEqual(r["total"], 1)
        r = self.c.get("/api/v1/search?kind=person&upper_color=red", headers=h).get_json()
        self.assertEqual(r["total"], 1)
        r = self.c.get(f"/api/v1/search?ts_from={time.time() - 20}", headers=h).get_json()
        self.assertEqual(r["total"], 1)
        r = self.c.get("/api/v1/search?page_size=1&page=2", headers=h).get_json()
        self.assertEqual((r["total"], r["pages"], len(r["events"])), (2, 2, 1))

    def test_meta_status_detail(self):
        _, tok = self.token()
        h = self.auth(tok)
        m = self.c.get("/api/v1/meta", headers=h).get_json()
        self.assertEqual(m["cameras"][0]["id"], "c1")
        self.assertIn("white", m["colors"])
        self.assertEqual(self.c.get("/api/v1/status", headers=h).status_code, 200)
        self.assertEqual(self.c.get(f"/api/v1/events/{self.id1}", headers=h).get_json()["event"]["plate_text"], "ASY-3549")
        self.assertEqual(self.c.get("/api/v1/events/9999", headers=h).status_code, 404)

    def test_guard_can_watch_live_but_not_search(self):
        _, tok = self.token("guard1")
        h = self.auth(tok)
        self.assertEqual(self.c.get("/api/v1/events", headers=h).status_code, 200)
        self.assertEqual(self.c.get("/api/v1/search", headers=h).status_code, 403)

    def test_deleted_user_token_stops_working(self):
        _, tok = self.token()
        self.db.delete_user("boss")
        self.assertEqual(self.c.get("/api/v1/events", headers=self.auth(tok)).status_code, 401)


if __name__ == "__main__":
    unittest.main()
