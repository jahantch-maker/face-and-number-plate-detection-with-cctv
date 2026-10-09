"""Start the recognition engine: reads all cameras, saves plate / face events.

    python run_engine.py                       # all cameras in config.yaml
    python run_engine.py --camera in_plate     # pilot: one camera only
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

from gatevision.config import load_config
from gatevision.db import Database
from gatevision.storage import EventStore

log = logging.getLogger("gatevision.engine")


def cleanup_loop(db, data_dir, days, stop):
    while not stop.is_set():
        try:
            n = db.delete_older_than(days, data_dir)
            if n:
                log.info("retention: removed %d events older than %s days", n, days)
        except Exception:
            log.exception("retention cleanup failed")
        stop.wait(3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--camera", action="append", help="camera id to run (repeatable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    data_dir = Path(cfg["storage"]["data_dir"])
    db = Database(data_dir / "gatevision.db")
    store = EventStore(data_dir, db, cfg["storage"]["save_full_frame"])

    cams = [c for c in cfg["cameras"] if not args.camera or c["id"] in args.camera]
    if not cams:
        raise SystemExit("No cameras selected. Check config.yaml / --camera.")

    # heavy imports only now, so config errors show up instantly
    from gatevision.detect import YoloTracker
    from gatevision.faces import FaceDetector
    from gatevision.pipeline import CameraWorker
    from gatevision.plates import PlateReader
    from gatevision.stream import RTSPStream

    m = cfg["models"]
    plates = None
    if any(c["role"] == "plate" for c in cams):
        try:
            import onnxruntime as ort
            gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            gpu = False
        log.info("loading plate models (GPU=%s)", gpu)
        plates = PlateReader(m["plate_detector"], m["plate_ocr"], use_gpu=gpu, ocr_custom=m.get("plate_ocr_custom"))

    stop = threading.Event()
    threads, streams = [], []
    for cam in cams:
        log.info("starting %s (%s %s)", cam["id"], cam["direction"], cam["role"])
        stream = RTSPStream(cam["url"], cam["id"]).start()
        detector = YoloTracker(m["detector"], m["detector_device"], m["detector_imgsz"], m["detector_conf"])
        faces = FaceDetector(m["face_model"]) if cam["role"] == "face" else None
        worker = CameraWorker(cam, cfg, detector, store, plates=plates, faces=faces)
        t = threading.Thread(target=worker.run, args=(stream, stop, db), name=cam["id"], daemon=True)
        t.start()
        threads.append(t)
        streams.append(stream)

    threading.Thread(
        target=cleanup_loop,
        args=(db, data_dir, cfg["storage"]["retention_days"], stop),
        daemon=True,
    ).start()

    log.info("engine running - press Ctrl+C to stop")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("stopping ...")
        stop.set()
        for t in threads:
            t.join(timeout=10)
        for s in streams:
            s.stop()


if __name__ == "__main__":
    main()
