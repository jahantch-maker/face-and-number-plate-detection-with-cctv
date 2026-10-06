"""Delete ALL saved records and photos (use after testing).

    python clear_events.py --yes
Logins and settings are kept.
"""
import sys
import time
from pathlib import Path

from gatevision.config import load_config
from gatevision.db import Database

if "--yes" not in sys.argv:
    raise SystemExit("This deletes every saved record and photo. Run again with --yes to confirm.")
cfg = load_config("config.yaml")
data = Path(cfg["storage"]["data_dir"])
n = Database(data / "gatevision.db").delete_older_than(0, data, now=time.time() + 60)
print(f"Deleted {n} records and their photos.")
