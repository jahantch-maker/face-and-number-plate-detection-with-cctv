import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlparse, unquote

import yaml

from gatevision.config import load_config
from gatevision.db import Database
from setup_wizard import build_config, is_ipv4, rtsp_url

ROOT = Path(__file__).resolve().parent.parent


class WizardTests(unittest.TestCase):
    def test_url_encoding_of_special_characters(self):
        url = rtsp_url("viewer", "p@ss:w/rd#1", "192.168.1.50", 554, 3)
        p = urlparse(url)
        self.assertEqual(p.hostname, "192.168.1.50")
        self.assertEqual(p.port, 554)
        self.assertEqual(unquote(p.password), "p@ss:w/rd#1")
        self.assertIn("channel=3&subtype=0", url)

    def test_ip_validation(self):
        self.assertTrue(is_ipv4("192.168.1.108"))
        self.assertFalse(is_ipv4("192.168.1"))
        self.assertFalse(is_ipv4("300.1.1.1"))
        self.assertFalse(is_ipv4("abc"))

    def test_build_config_roundtrips_through_loader_and_skips_missing_cameras(self):
        ch = {"in_plate": {"channel": 1, "name": "RG Barrier IN 1"}, "in_face": {"channel": 2, "name": "In face"},
              "out_plate": {"channel": 3, "name": "RG Barrier OUT 1"}, "out_face": {"channel": 0, "name": "x"}}
        cfg = build_config("10.0.0.5", 554, "viewer", "pw", ch, "D:/gv", cpu_mode=True)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            loaded = load_config(p)
        self.assertEqual([c["id"] for c in loaded["cameras"]], ["in_plate", "in_face", "out_plate"])
        self.assertEqual(loaded["cameras"][0]["role"], "plate")
        self.assertEqual(loaded["cameras"][2]["direction"], "OUT")
        self.assertEqual(loaded["models"]["detector"], "yolo11n.pt")
        self.assertEqual(loaded["tracking"]["process_fps"], 6)
        self.assertEqual(loaded["tracking"]["lost_seconds"], 2.0)     # defaults still merged

    def test_interactive_run_end_to_end(self):
        """Pipe a full set of answers into the wizard and check what it produced."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            data = d / "gvdata"
            answers = "\n".join([
                "192.168.1.108", "", "viewer", "Nvr#pass@1",        # step 1 (port default)
                "1", "", "2", "", "3", "", "0",                      # channels; names default; OUT face skipped
                str(data),                                           # data folder
                "AdminPass123", "AdminPass123",                      # admin
                "y", "ManagerPass1", "ManagerPass1",                 # manager
                "n",                                                 # no guard
            ]) + "\n"
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            r = subprocess.run([sys.executable, str(ROOT / "setup_wizard.py"), "--no-probe"], input=answers,
                               text=True, capture_output=True, cwd=d, env=env, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("Setup finished", r.stdout)
            cfg = load_config(d / "config.yaml")
            self.assertEqual(len(cfg["cameras"]), 3)
            self.assertEqual(cfg["cameras"][0]["name"], "RG Barrier IN 1")
            self.assertIn("Nvr%23pass%401", cfg["cameras"][0]["url"])
            db = Database(data / "gatevision.db")
            self.assertEqual({u["username"]: u["role"] for u in db.list_users()},
                             {"admin": "admin", "manager": "manager"})


if __name__ == "__main__":
    unittest.main()
