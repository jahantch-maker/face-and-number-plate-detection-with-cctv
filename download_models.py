"""One-time download of the small face-detection model (YuNet, MIT licence).

The YOLO and plate models download themselves the first time the engine runs.
"""
import gatevision  # noqa: F401  (enables Windows certificate store for HTTPS)
from pathlib import Path
from urllib.request import urlretrieve

URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
       "face_detection_yunet_2023mar.onnx")
dest = Path("models/face_detection_yunet_2023mar.onnx")
dest.parent.mkdir(exist_ok=True)
if dest.exists():
    print("already present:", dest)
else:
    print("downloading", URL)
    urlretrieve(URL, dest)
    print("saved", dest, f"({dest.stat().st_size // 1024} KB)")
