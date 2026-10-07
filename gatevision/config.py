"""Configuration loading with sensible defaults."""
from __future__ import annotations

import copy
import secrets
from pathlib import Path

import yaml

DEFAULTS = {
    "storage": {
        "data_dir": "data",
        "retention_days": 30,
        "save_full_frame": True,
    },
    "models": {
        "detector": "yolo11s.pt",      # downloaded automatically on first run
        "detector_device": None,        # None = auto (GPU if available); "cpu"; "0"
        "detector_imgsz": 960,
        "detector_conf": 0.35,
        "plate_detector": "yolo-v9-t-640-license-plate-end2end",   # best on tilted / red-lit plates in our gate test
        "plate_ocr": "cct-xs-v1-global-model",
        "face_model": "models/face_detection_yunet_2023mar.onnx",
    },
    "tracking": {
        "process_fps": 15,         # frames per second analysed per camera, 0 = as fast as the PC can (the Cameras page shows what is really achieved)
        "lost_seconds": 2.0,       # track is finished when unseen this long
        "min_track_seconds": 0.4,  # ignore flickers shorter than this
        "max_dwell_seconds": 120,  # force-save very long stays
        "stop_motion_ratio": 0.08, # moved < 8% of its size in 1 s => "stopped"
        "require_stop": False,     # capture people/vehicles even while moving (True = only those that stop)
        "ocr_interval": 0.15,      # seconds between plate reads of one vehicle
        "max_ocr_attempts": 12,
        "max_face_attempts": 20,
        "dedup_seconds": 15,       # same plate on same camera within this = one
        "min_plate_conf": 0.20,      # reads below 0.5 are shown with a "?" so a human checks the photo
        # "Near the camera only" rules. Sizes are fractions of the picture, so
        # they work for any camera resolution. Override per camera with
        # min_height / min_width / require_face / person_conf in config.yaml.
        "min_person_height": 0.40,   # a person must be at least 40% of the picture height
        "min_vehicle_width": 0.30,   # a vehicle must be at least 30% of the picture width
        "person_conf": 0.55,         # people detections below this certainty are ignored
        "min_face_px": 50,           # ignore faces narrower than this many pixels
        "require_face": False,       # True = a face camera saves a person only if a face was found (misses people who look away)
        "retrack_seconds": 4,        # same person/vehicle re-detected within this long at the same spot = ONE record
    },
    "web": {
        "host": "0.0.0.0",
        "port": 8080,
        "secret_key": None,
        "page_size": 48,
    },
    "cameras": [],
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy config.example.yaml to config.yaml and edit it."
        )
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, raw)
    base = path.resolve().parent
    data_dir = Path(cfg["storage"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = base / data_dir
    cfg["storage"]["data_dir"] = str(data_dir)
    face_model = Path(cfg["models"]["face_model"])
    if not face_model.is_absolute():
        cfg["models"]["face_model"] = str(base / face_model)
    seen = set()
    for cam in cfg["cameras"]:
        for key in ("id", "name", "url", "direction", "role"):
            if key not in cam:
                raise ValueError(f"camera entry missing '{key}': {cam}")
        if cam["id"] in seen:
            raise ValueError(f"duplicate camera id {cam['id']}")
        seen.add(cam["id"])
        cam["direction"] = str(cam["direction"]).upper()
        cam["role"] = str(cam["role"]).lower()
        if cam["direction"] not in ("IN", "OUT"):
            raise ValueError(f"camera {cam['id']}: direction must be IN or OUT")
        if cam["role"] not in ("plate", "face"):
            raise ValueError(f"camera {cam['id']}: role must be plate or face")
        cam.setdefault("gate", "main")
        cam.setdefault("roi", [0.0, 0.0, 1.0, 1.0])
    return cfg


def web_secret(cfg: dict) -> str:
    """Return the configured session secret, or a persistent generated one."""
    if cfg["web"].get("secret_key"):
        return cfg["web"]["secret_key"]
    keyfile = Path(cfg["storage"]["data_dir"]) / "secret.key"
    keyfile.parent.mkdir(parents=True, exist_ok=True)
    if not keyfile.exists():
        keyfile.write_text(secrets.token_hex(32))
    return keyfile.read_text().strip()
