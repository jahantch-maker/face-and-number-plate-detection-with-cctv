"""Zip the labelled plate photos (labels.csv + images) into plates.zip in the
Gate Vision folder, ready to upload to Google Drive / GateVisionTraining.

    pack_plates.bat
"""
from __future__ import annotations

import shutil

from common import ROOT, has_text, plates_dir, read_labels


def main():
    folder = plates_dir()
    rows = read_labels(folder / "labels.csv")
    labelled = [r for r in rows if has_text(r["label"])]
    if not labelled:
        print("No labelled photos yet: run collect_plates.bat and label_plates.bat first.")
        return
    zip_path = shutil.make_archive(str(ROOT / "plates"), "zip", folder)
    print(f"Made {zip_path} with {len(labelled)} labelled photos "
          f"({sum(r['vehicle'] in ('motorcycle', 'bicycle') for r in labelled)} bikes).")
    print("Upload it to Google Drive, folder GateVisionTraining, then open train_plate_reader.ipynb in Colab.")


if __name__ == "__main__":
    main()
