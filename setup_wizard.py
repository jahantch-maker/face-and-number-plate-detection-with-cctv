"""Question-and-answer setup: writes config.yaml, creates web logins, tests cameras.

Run by install.bat. Can be run again any time:  .venv\\Scripts\\python setup_wizard.py
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import yaml

CAMERA_SLOTS = [
    # key, gate side, role, default channel, default name
    ("in_plate", "IN", "plate", 1, "RG Barrier IN 1"),
    ("in_face", "IN", "face", 2, "RG Barrier IN 2"),
    ("out_plate", "OUT", "plate", 3, "RG Barrier OUT 1"),
    ("out_face", "OUT", "face", 4, "RG Barrier OUT 2"),
]


def rtsp_url(user: str, password: str, host: str, port: int, channel: int, subtype: int = 0) -> str:
    """Dahua RTSP address. User and password are URL-encoded (handles @ # : / etc.)."""
    return (f"rtsp://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}"
            f"/cam/realmonitor?channel={channel}&subtype={subtype}")


def build_config(host, port, user, password, channels: dict, data_dir: str, cpu_mode: bool) -> dict:
    cams = []
    for key, direction, role, _default, default_name in CAMERA_SLOTS:
        if key not in channels or not channels[key]["channel"]:
            continue
        c = channels[key]
        cams.append({
            "id": key,
            "name": c.get("name") or default_name,
            "gate": "main",
            "direction": direction,
            "role": role,
            "url": rtsp_url(user, password, host, port, c["channel"]),
        })
    cfg = {
        "storage": {"data_dir": data_dir, "retention_days": 30, "save_full_frame": True},
        "web": {"port": 8080},
        "cameras": cams,
    }
    if cpu_mode:   # lighter settings so a PC without an NVIDIA card keeps up
        cfg["models"] = {"detector": "yolo11n.pt", "detector_imgsz": 640}
        cfg["tracking"] = {"process_fps": 6}
    return cfg


# ----------------------------------------------------------------- prompts
def ask(prompt: str, default: str | None = None, validator=None) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        val = input(f"{prompt}{suffix}: ").strip()
        if not val and default is not None:
            val = default
        if val and (validator is None or validator(val)):
            return val
        print("  -> not valid, please try again.")


def ask_secret(prompt: str, min_len: int = 1, confirm: bool = False) -> str:
    while True:
        pw = getpass.getpass(f"{prompt} (you will not see the letters while typing): ")
        if len(pw) < min_len:
            print(f"  -> use at least {min_len} characters.")
            continue
        if confirm and getpass.getpass("  Type it again: ") != pw:
            print("  -> the two entries do not match.")
            continue
        return pw


def is_ipv4(s: str) -> bool:
    return bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", s)) and all(int(p) <= 255 for p in s.split("."))


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "this-PC-ip"


def default_data_dir() -> str:
    return "D:/gatevision_data" if os.name == "nt" and Path("D:/").exists() else "C:/gatevision_data" \
        if os.name == "nt" else str(Path.cwd() / "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-probe", action="store_true", help="skip the live camera test")
    args = ap.parse_args()

    print("=" * 62)
    print(" Gate Vision - setup")
    print(" Answer the questions. Press Enter to accept the [suggested] value.")
    print("=" * 62)

    print("\nSTEP 1 of 4 - Your NVR (the Dahua recorder)")
    host = ask("NVR IP address (the one you open in the browser)", validator=is_ipv4)
    port = int(ask("RTSP port (almost always 554)", "554", str.isdigit))
    user = ask("NVR username created for this system (e.g. viewer)", "viewer")
    password = ask_secret("NVR password")

    print("\nSTEP 2 of 4 - Which NVR channel is each camera?")
    print("  (Channel = the camera's number in the NVR web page: 1, 2, 3 ...)")
    print("  Type 0 if a camera does not exist.")
    channels = {}
    for key, direction, role, default_ch, default_name in CAMERA_SLOTS:
        what = "PLATE camera (low, sees number plates)" if role == "plate" else "FACE camera (sees faces)"
        ch = int(ask(f"{direction} gate - {what}: channel", str(default_ch), str.isdigit))
        name = default_name
        if ch:
            name = ask("   Name to show on the web page", default_name)
        channels[key] = {"channel": ch, "name": name}

    print("\nSTEP 3 of 4 - Where to keep the photos and database")
    print("  Choose a big drive. 30 days of records need only a few GB.")
    data_dir = ask("Folder", default_data_dir())

    cpu_mode = Path("gpu_mode.txt").exists() and Path("gpu_mode.txt").read_text().strip() == "cpu"
    cfg = build_config(host, port, user, password, channels, data_dir, cpu_mode)
    if not cfg["cameras"]:
        raise SystemExit("No cameras were configured. Run the setup again.")
    cfgp = Path(args.config)
    if cfgp.exists():
        shutil.copy(cfgp, cfgp.with_suffix(".yaml.bak"))
        print(f"  (old settings saved as {cfgp.with_suffix('.yaml.bak').name})")
    cfgp.write_text("# Written by setup_wizard.py - safe to edit with Notepad.\n"
                    + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nSettings saved to {cfgp.resolve()}")

    print("\nSTEP 4 of 4 - Web page logins")
    from werkzeug.security import generate_password_hash
    from gatevision.config import load_config
    from gatevision.db import Database
    full = load_config(cfgp)
    db = Database(Path(full["storage"]["data_dir"]) / "gatevision.db")
    print("  admin   = full access (you).  manager = search and playback info.  guard = live list only.")
    db.add_user("admin", generate_password_hash(ask_secret("Password for 'admin'", 8, True)), "admin")
    if ask("Create a 'manager' login for the management team? (y/n)", "y").lower().startswith("y"):
        db.add_user("manager", generate_password_hash(ask_secret("Password for 'manager'", 8, True)), "manager")
    if ask("Create a 'guard' login for the gate guards? (y/n)", "y").lower().startswith("y"):
        db.add_user("guard", generate_password_hash(ask_secret("Password for 'guard'", 8, True)), "guard")

    if not args.no_probe:
        print("\nTesting every camera (about 10 seconds each) ...\n")
        subprocess.call([sys.executable, "probe_rtsp.py", "--config", str(cfgp)])

    print("\n" + "=" * 62)
    print(" Setup finished.")
    print(" 1. Double-click  start_all.bat  to run the system.")
    print(f" 2. Open  http://localhost:8080  (on other devices: http://{local_ip()}:8080)")
    print(" 3. After you see it working, run  install_autostart.bat  as Administrator")
    print("    so it starts by itself after a power cut.")
    print("=" * 62)


if __name__ == "__main__":
    main()
