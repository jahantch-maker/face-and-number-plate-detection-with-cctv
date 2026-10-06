"""Change how many pictures per second the engine analyses per camera.

    python set_fps.py          -> shows the current value
    python set_fps.py 15       -> analyse up to 15 per second per camera
    python set_fps.py 0        -> as fast as the PC can (uses the most CPU/GPU)

Restart the engine afterwards. The Cameras page shows the speed really reached.
"""
import sys

import yaml

PATH = "config.yaml"
with open(PATH, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}
tr = cfg.setdefault("tracking", {})
print("current process_fps:", tr.get("process_fps", "default (15)"))
if len(sys.argv) > 1:
    tr["process_fps"] = float(sys.argv[1]) if "." in sys.argv[1] else int(sys.argv[1])
    with open(PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    print("new process_fps:    ", tr["process_fps"], " - now close and restart start_all.bat")
