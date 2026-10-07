"""Change the NVR channel of one camera in config.yaml - no Notepad needed.

    python set_channel.py                 -> lists every camera and its channel
    python set_channel.py in_plate 8      -> camera "in_plate" now uses NVR channel 8

A backup of the old file is saved as config.yaml.before-channel first.
Close the Gate Vision windows before running, and start them again afterwards.
"""
import re
import shutil
import sys
from pathlib import Path

import yaml

PATH = Path("config.yaml")
if not PATH.exists():
    sys.exit("config.yaml not found - run this from the C:\\gate-vision folder.")
print("Reading:", PATH.resolve(), "\n")
cfg = yaml.safe_load(PATH.read_text(encoding="utf-8")) or {}
cams = cfg.get("cameras", [])


def show():
    for c in cams:
        m = re.search(r"channel=(\d+)", c.get("url", ""))
        print(f"  {c.get('id'):10s} {c.get('direction')}/{c.get('role'):5s}  {c.get('name'):18s} channel {m.group(1) if m else '?'}")


print("Cameras now:")
show()
if len(sys.argv) == 3:
    cam_id, new = sys.argv[1], sys.argv[2]
    hit = [c for c in cams if c.get("id") == cam_id]
    if not hit or not new.isdigit():
        sys.exit(f"\nNo camera called '{cam_id}', or the channel is not a number.")
    shutil.copy(PATH, "config.yaml.before-channel")
    hit[0]["url"] = re.sub(r"channel=\d+", f"channel={new}", hit[0]["url"])
    PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nChanged {cam_id} to channel {new}. Cameras now:")
    show()
    print("\nNow start start_all.bat again.")
