# Teaching the plate reader Pakistani plates

Gate Vision reads car plates well but gets many motorbike and loader-rickshaw
plates wrong. These tools teach the plate reader on **our own gate photos**,
helped by public Pakistani plate photos and made-up two-line bike plates.
Training runs free on Google Colab; nothing is paid for.

## Why our own photos

We searched for a public dataset of Pakistani **bike** plates and did not
find one (October 2026). What exists is mostly car plates, see the table at the
bottom. Our own camera photos are the best teacher anyway: same cameras, same
angle, same light, same dirt. A few hundred labelled bike plates should make
the biggest difference.

## Steps

All `.bat` files are in `G:\gate-vision\training\`. Double-click them.

1. **collect_plates.bat** copies the plate photos Gate Vision has saved into
   `data\training\plates\`. Gate Vision deletes photos after 30 days, so run
   it every week or two. It never changes labels you already typed.
2. **label_plates.bat** opens a labelling page. On the server open
   `http://localhost:8090`; from the phone use the same address you open
   Gate Vision with, but ending in `:8090` instead of `:8080`. Log in with a Gate Vision admin or manager
   account. For each photo type the correct plate:
   - letters and the big number, no dash, no small year: `LEA-17-5989` is `LEA5989`
   - press **Can't read it** when even you cannot read it
   - bikes and rickshaws come first, because they matter most

   Aim for **at least 300 photos, as many bikes as possible**. More is better.
3. **evaluate_ocr.bat** scores the reader on your labelled photos. Write the
   number down: this is the true accuracy today.
4. **pack_plates.bat** makes `G:\gate-vision\plates.zip`. Upload it to Google
   Drive into a folder called **GateVisionTraining**.
5. Open [train_plate_reader.ipynb in Colab](https://colab.research.google.com/github/jahantch-maker/face-and-number-plate-detection-with-cctv/blob/main/training/train_plate_reader.ipynb),
   choose **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**.
   It prints the score before and after training.
6. If the after score is higher, download `pk_plate_ocr.onnx` and
   `pk_plate_ocr_config.yaml` from the Drive folder and copy both into
   `G:\gate-vision\models\`. Restart Gate Vision (close its windows, run
   `start_all.bat`). It uses the trained reader automatically when those two
   files are there. To go back, delete them and restart.

`update.bat` never touches `models\` or `data\`, so the trained reader and
your labels survive updates.

## Optional: public Pakistani plate photos

These help a little with cars and fonts. To use one, sign in to Roboflow
(free), open the page, choose **Download dataset > YOLOv8**, download the zip
and put it in `GateVisionTraining\public\` on Google Drive. The notebook
converts it.

| Dataset | Photos | Labels | Licence | Bikes? | Use |
| --- | --- | --- | --- | --- | --- |
| [Pakistan License Plate Detection](https://universe.roboflow.com/license-plate-yf8bv/pakistan-license-plate-detection) (Roboflow) | about 6,000 | every letter boxed, so plate text can be rebuilt | MIT | not stated, mostly cars | **best public one**, works with the notebook |
| [Pk Number Plates](https://universe.roboflow.com/burhan-khan/pk-number-plates) (Roboflow) | about 1,700 | plate box only, no text | CC BY 4.0 | not stated | only for a plate *finder*, not the reader |
| [Pakistani Number Plates](https://universe.roboflow.com/malik-kashif-saeed-aswwf/pakistani-number-plates) (Roboflow) | about 340 | plate box only | CC BY 4.0 | not stated | plate finder only |
| [pakistan_number_plates](https://github.com/usamabhatti1998/pakistan_number_plates) (GitHub + Kaggle) | about 12,000 (1,750 real photos, rest copies) | every letter boxed | none stated | no | mixed with foreign and fake plates; skip unless the licence is cleared |
| PLPD (Sensors 2021 paper) | 6,000 | box + text | not released | not stated | not downloadable |

## For developers

| File | What it does |
| --- | --- |
| `collect_plates.py` | copies saved plate photos (un-padded to the plate box) into `data/training/plates/` + `labels.csv` |
| `label_plates.py` | Flask labelling page, Basic auth against the Gate Vision users table |
| `evaluate_ocr.py` | exact-match accuracy of a reader (hub name or `.onnx`) with the app's PK clean-up |
| `convert_char_boxes.py` | Roboflow per-character YOLO export to plate crops + text (two-row aware) |
| `synth_plates.py` | made-up two-line bike plates with CCTV-style blur, angle and light |
| `make_dataset.py` | merges sources, splits by plate number (no plate in both lists), repeats own and bike photos |
| `fix_onnx.py` | rewrites the `Erfc` op newer TensorFlow emits (onnxruntime cannot load it) and checks against Keras |
| `train_plate_reader.ipynb` | Colab: fine-tunes `cct-xs-v1-global` (the reader in use) with `fast-plate-ocr train --weights-path` |

The app loads `models/pk_plate_ocr.onnx` + `models/pk_plate_ocr_config.yaml`
when both exist (`models.plate_ocr_custom` in the config), and falls back to
the built-in reader if they fail to load.
