import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import update


def make_zip(path: Path, files: dict):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr("repo-main/" + name, data)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "install"
        self.root.mkdir()
        update.ROOT = self.root
        update.VERSION_FILE = self.root / ".version"
        update.MANIFEST_FILE = self.root / ".manifest.json"
        update.BACKUP_DIR = self.root / "_backup"
        update.stop_services = lambda: None
        update.install_deps = lambda: True

    def run_update(self, files):
        z = self.tmp / "u.zip"
        make_zip(z, files)
        return update.do_update(str(z), False, False)

    def test_preserved_files_survive_and_code_is_replaced(self):
        (self.root / "config.yaml").write_text("secret: 1")
        (self.root / "data").mkdir()
        (self.root / "data" / "gatevision.db").write_text("db")
        (self.root / "yolo11s.pt").write_text("weights")
        (self.root / "engine.log").write_text("log")
        files = {"run_engine.py": "v1", "requirements.txt": "a", "install_deps.py": "x",
                 "config.yaml": "OVERWRITE", "data/x.db": "OVERWRITE", "a.pt": "OVERWRITE"}
        self.assertEqual(self.run_update(files), 0)
        self.assertEqual((self.root / "run_engine.py").read_text(), "v1")
        self.assertEqual((self.root / "config.yaml").read_text(), "secret: 1")
        self.assertEqual((self.root / "data" / "gatevision.db").read_text(), "db")
        self.assertEqual((self.root / "yolo11s.pt").read_text(), "weights")
        self.assertFalse((self.root / "a.pt").exists())

    def test_obsolete_files_removed_only_if_we_installed_them(self):
        base = {"run_engine.py": "v1", "old.py": "old"}
        self.run_update(base)
        (self.root / "my_notes.txt").write_text("mine")
        self.run_update({"run_engine.py": "v2"})
        self.assertFalse((self.root / "old.py").exists())
        self.assertTrue((self.root / "my_notes.txt").exists())
        self.assertEqual((self.root / "run_engine.py").read_text(), "v2")

    def test_rollback_restores_previous_version(self):
        self.run_update({"run_engine.py": "v1", "old.py": "old"})
        self.run_update({"run_engine.py": "v2", "new.py": "new"})
        self.assertEqual(update.do_rollback(), 0)
        self.assertEqual((self.root / "run_engine.py").read_text(), "v1")
        self.assertTrue((self.root / "old.py").exists())
        self.assertFalse((self.root / "new.py").exists())

    def test_phone_app_source_not_installed_on_server(self):
        self.run_update({"run_engine.py": "v1", "android/app/x.kt": "k", ".github/workflows/a.yml": "y"})
        self.assertFalse((self.root / "android").exists())
        self.assertFalse((self.root / ".github").exists())

    def test_rejects_wrong_zip(self):
        self.assertEqual(self.run_update({"hello.txt": "x"}), 1)

    def test_unsafe_zip_path_rejected(self):
        z = self.tmp / "bad.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("../evil.py", "x")
        self.assertEqual(update.do_update(str(z), False, False), 1)


if __name__ == "__main__":
    unittest.main()
