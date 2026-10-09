"""Make made-up Pakistani bike-style plates (two lines: letters + small year on
top, the big number below) so the reader learns that layout even before we
have many real bike photos. No public dataset of Pakistani bike plates exists.

    python training/synth_plates.py <output folder> [--count 6000]

Writes <output>/images/*.jpg and <output>/labels.csv (image_path, plate_text).
The text label follows the app: letters + number, year left out.
Made-up photos only help; real labelled gate photos matter far more.
"""
from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
CITY = ["LAHORE", "PUNJAB", "SINDH", "ISLAMABAD", "KARACHI", "ICT", ""]


def font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def fitted(d, text, size, max_w):
    """The biggest font up to ``size`` at which ``text`` fits in ``max_w`` pixels."""
    while size > 12 and d.textlength(text, font=font(size)) > max_w:
        size -= 2
    return font(size)


def plate_text(rnd):
    letters = "".join(rnd.choice(string.ascii_uppercase) for _ in range(rnd.choice((2, 3, 3, 3, 4))))
    number = str(rnd.randint(1, 9999)) if rnd.random() < 0.8 else f"{rnd.randint(1, 999):03d}"
    year = f"{rnd.randint(5, 25):02d}"
    return letters, year, number


def draw(rnd, letters, year, number, two_line):
    w, h = (300, 200) if two_line else (440, 120)
    bg = rnd.choice([(245, 245, 245), (235, 238, 230), (250, 245, 225)])
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, w - 4, h - 4], outline=(20, 20, 20), width=rnd.randint(2, 5))
    if rnd.random() < 0.7:                                           # green emblem strip on the left
        d.rectangle([6, 6, int(w * 0.14), h - 7], fill=(20, 110 + rnd.randint(0, 40), 50))
    ink = (rnd.randint(0, 40),) * 3
    x0 = int(w * 0.17)
    room = w - x0 - 12
    if two_line:
        small = font(rnd.randint(30, 38))
        top = fitted(d, letters, rnd.randint(62, 74), room - 60)
        d.text((x0, 18), letters, font=top, fill=ink)
        d.text((x0 + d.textlength(letters, font=top) + 10, 34), year, font=small, fill=ink)
        d.text((x0 + 4, 100), number, font=fitted(d, number, rnd.randint(70, 84), room), fill=ink)
    else:
        text = f"{letters}-{year}-{number}" if rnd.random() < 0.5 else f"{letters} {number}"
        d.text((x0, 18), text, font=fitted(d, text, rnd.randint(66, 78), room), fill=ink)
        city = rnd.choice(CITY)
        if city:
            d.text((x0 + 40, h - 30), city, font=font(20), fill=ink)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def degrade(rnd, img):
    """Look like a CCTV crop: perspective, blur, low resolution, light, noise."""
    h, w = img.shape[:2]
    j = 0.12
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rnd.uniform(-j, j) * w + x, rnd.uniform(-j, j) * h + y] for x, y in src])
    m = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    scale = rnd.uniform(0.25, 0.6)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if rnd.random() < 0.6:
        k = rnd.choice((3, 3, 5))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k                                  # sideways motion blur
        img = cv2.filter2D(img, -1, kernel)
    img = img.astype(np.float32) * rnd.uniform(0.4, 1.3) + rnd.uniform(-40, 30)
    if rnd.random() < 0.3:                                           # orange night light
        img *= np.array([rnd.uniform(0.4, 0.8), rnd.uniform(0.7, 0.95), 1.0], np.float32)
    img += np.random.default_rng(rnd.randint(0, 10**9)).normal(0, rnd.uniform(2, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rnd.randint(35, 90)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--two-line-share", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.count):
        letters, year, number = plate_text(rnd)
        img = degrade(rnd, draw(rnd, letters, year, number, rnd.random() < args.two_line_share))
        name = f"synth_{i:06d}.jpg"
        cv2.imwrite(str(args.out / "images" / name), img)
        rows.append({"image_path": f"images/{name}", "plate_text": letters + number})
    with (args.out / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "plate_text"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} made-up plates written to {args.out}")


if __name__ == "__main__":
    main()
