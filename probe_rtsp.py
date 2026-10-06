"""Check that a camera's RTSP stream works and save one snapshot.

    python probe_rtsp.py "rtsp://user:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
    python probe_rtsp.py --config config.yaml          # test every camera in the config
"""
from __future__ import annotations

import argparse
import re
import time

import gatevision  # noqa: F401  (sets RTSP-over-TCP before cv2 loads)
import cv2


def mask(url: str) -> str:
    return re.sub(r"//([^:/]+):[^@]*@", r"//\1:****@", url)


def probe(name: str, url: str, out: str) -> bool:
    print(f"[{name}] connecting to {mask(url)} ...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[{name}] FAILED to open. Check IP, password (special characters must be "
              "URL-encoded, e.g. @ -> %40), port 554 and that RTSP is enabled on the camera.")
        return False
    frames, t0, frame = 0, time.time(), None
    while time.time() - t0 < 5:
        ok, f = cap.read()
        if ok:
            frames, frame = frames + 1, f
    cap.release()
    if frame is None:
        print(f"[{name}] opened, but no frames arrived.")
        return False
    h, w = frame.shape[:2]
    cv2.imwrite(out, frame)
    print(f"[{name}] OK  {w}x{h}  ~{frames / 5:.1f} fps  snapshot -> {out}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?")
    ap.add_argument("--config")
    args = ap.parse_args()
    if args.config:
        from gatevision.config import load_config
        results = [probe(c["id"], c["url"], f"probe_{c['id']}.jpg") for c in load_config(args.config)["cameras"]]
        print(f"\n{sum(results)}/{len(results)} cameras OK")
    elif args.url:
        probe("camera", args.url, "probe_snapshot.jpg")
    else:
        ap.error("give an RTSP url or --config")


if __name__ == "__main__":
    main()
