"""Try the recognition on still images (no camera needed) - great for tuning.

    python process_image.py gate1.png gate2.png            # plate mode
    python process_image.py face1.png --role face

Writes <name>_result.jpg next to each image with boxes + text drawn on it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from gatevision.colors import clothing_colors, vehicle_color
from gatevision.config import DEFAULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--role", choices=["plate", "face"], default="plate")
    args = ap.parse_args()
    m = DEFAULTS["models"]

    from ultralytics import YOLO
    from gatevision.detect import PERSON, VEHICLE_CLASSES
    model = YOLO(m["detector"])
    classes = [PERSON] if args.role == "face" else list(VEHICLE_CLASSES)

    if args.role == "plate":
        from gatevision.plates import PlateReader, pretty
        plates = PlateReader(m["plate_detector"], m["plate_ocr"], ocr_custom=m.get("plate_ocr_custom"))
    else:
        from gatevision.faces import FaceDetector
        faces = FaceDetector("models/face_detection_yunet_2023mar.onnx")

    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        res = model.predict(img, classes=classes, conf=0.25, imgsz=m["detector_imgsz"], verbose=False)[0]
        print(f"\n{path}: {len(res.boxes)} object(s)")
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            label = res.names[int(b.cls[0])]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            note = label
            if args.role == "plate":
                reads = plates.read(crop)
                color = vehicle_color(crop)
                txt = ", ".join(f"{pretty(r.text)} ({r.conf:.2f}{'' if r.valid else ', odd format'})" for r in reads if r.text) \
                    or ("plate found but not readable" if reads else "no plate found")
                note = f"{label} {color} | {txt}"
            else:
                up, lo = clothing_colors(crop)
                n = len(faces.detect(crop[: int(crop.shape[0] * 0.55)]))
                note = f"{label} top={up} bottom={lo} faces={n}"
            print("  ", note)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, note, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        out = str(Path(path).with_name(Path(path).stem + "_result.jpg"))
        cv2.imwrite(out, img)
        print("   saved", out)


if __name__ == "__main__":
    main()
