# Gate Vision

Local number-plate and face logging for a CCTV gate. It reads the live video of
your Dahua cameras over RTSP, and for every vehicle and person that stops at
the barrier it saves one record: plate text, plate photo, vehicle type and
colour, face photo, clothing colours, time, camera and IN/OUT. A password
protected web page lets management filter those records in seconds instead of
scrubbing through hours of video.

* 100 % open-source models, runs offline on your PC. **No subscription.**
* Your NVRs keep recording exactly as before. This is an add-on.
* Records and photos older than 30 days (configurable) are deleted automatically.

```
Cameras --RTSP--> run_engine.py --> database + photos (D:\gatevision_data)
                                         |
                    browser (phone/PC) <-- run_web.py (login, search, live list)
```

## Quick start on the gate PC (Windows)

1. Install **Python 3.12** from python.org (tick *Add python.exe to PATH*).
2. In the NVR web page create a user `viewer` (live view only). Set each camera's main-stream frame rate to 10 fps.
3. Unzip this folder (e.g. `C:\gate-vision`) and double-click **`install.bat`**. It installs everything
   (GPU version automatically if an NVIDIA card is present), downloads the AI models, then asks you
   a few questions (NVR address, channel numbers, folder, passwords) and tests every camera.
4. Double-click **`start_all.bat`**, then open http://localhost:8080.
5. When happy, right-click **`install_autostart.bat`** and *Run as administrator* so it starts after power cuts.

## Android app (phone)

A native Android app (**Gate Vision**) shows the same live detections and search as the web page, and refreshes
itself every 2 seconds while it is open. Source: `android/`. Guard accounts see only the live list; admin / manager
accounts get Live, Search (all the web filters) and Cameras.

1. **Server side:** run `update.bat` once so the server has the phone API (`/api/v1/...`).
2. **Reach the server from anywhere:** install **Tailscale** (free) on the server and on the phone, sign in to the
   same account, and note the server's Tailscale address (`100.x.y.z`).
3. **Get the app:** open this repo's **Releases** page on the phone, tap the `android-latest` release and install
   `GateVision.apk` (allow "install unknown apps" for your browser once).
4. Open the app, enter the server address (`100.x.y.z` is enough), your web user name and password.

Every change to `android/` on GitHub is built automatically (Actions tab -> *Build Android app*) and replaces the
`android-latest` release; installing the new APK updates the app in place and keeps you signed in.

## Updating to the latest version (one click)

Every improvement is published on GitHub. To get it on the gate PC / server, double-click **`update.bat`**.
It checks GitHub, stops Gate Vision, downloads the new version, replaces the program files, installs any
new packages, restarts everything and opens the browser. Internet access is needed; Git is **not**.

* Your `config.yaml`, `data\`, `models\`, `.venv\`, logs and `*.pt` models are never touched.
* The old version is saved in `_backup\` (last 5 kept). Something wrong? Run `update.bat --rollback`.
* `update.bat --force` re-applies the latest version; `update.bat --deps` re-runs the package installer;
  `update.bat --zip file.zip` updates from a zip you downloaded by hand (no internet on the server).
* First time only: copy `update.bat` and `update.py` into the install folder once. After that the updater updates itself
  (except `update.bat`, which is tiny and rarely changes).

The detailed manual steps are below for reference.

## 1. Recommended PC

Windows 11 Pro, Intel i5 12th gen+ / Ryzen 5, **NVIDIA RTX 3060 12 GB** (or 3050 6 GB / 4060),
16-32 GB RAM, 512 GB SSD + 2 TB HDD, gigabit LAN to the camera switch, and a UPS.

## 2. Camera settings that matter more than any software

On each camera's web page (use a **separate read-only user** for this PC):

* Main stream: keep full resolution, set frame rate to **8-12 fps** (saves PC load, plenty for a stopped vehicle).
* **Plate cameras:** fixed, fast shutter (about 1/500 s or faster) so wheels/plates are not blurred, and
  check night footage for headlight glare washing out the plate.
* Make sure the clock/time zone on the cameras, NVR and this PC agree - the web page shows the PC's
  time for each record, and you will use it to find the matching NVR video.

## 3. Install on Windows

1. Install **Python 3.11 or 3.12** (tick "Add to PATH") and the latest **NVIDIA driver**.
2. Copy this folder to e.g. `C:\gate-vision`, open Command Prompt there:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install PyTorch **with CUDA** using the selector at https://pytorch.org (Windows / Pip / CUDA), then:
   ```
   pip install -r requirements.txt
   python download_models.py
   ```
   (No NVIDIA GPU? Change `fast-alpr[onnx-gpu]` to `fast-alpr[onnx]` - it works for 1-2 cameras, slowly.)
4. `copy config.example.yaml config.yaml`, then edit it: camera IPs/passwords and `data_dir`.

## 4. First run - in this order

1. **Check the streams** (this is the RTSP test):
   `python probe_rtsp.py --config config.yaml`
   Each camera should print `OK`, its resolution and fps, and save `probe_<id>.jpg`.
2. **Try recognition on pictures, no camera needed:**
   `python process_image.py probe_in_plate.jpg other_screenshot.png`
   It writes `*_result.jpg` showing the boxes, plate text and colours. Send me these if plates are misread.
3. **Pilot with one camera:** `python run_engine.py --camera in_plate`, drive a car through, watch the log.
4. **Create logins and start the web page:**
   ```
   python manage_users.py add admin   --role admin
   python manage_users.py add manager1 --role manager
   python manage_users.py add guard1   --role guard
   python run_web.py
   ```
   Open `http://localhost:8080` (other devices: `http://<PC-IP>:8080`).
   *manager / admin*: search, detail, camera status. *guard*: only the Live list.
5. **All cameras:** `python run_engine.py`.

## 5. Run automatically after power cuts

Run Command Prompt **as Administrator** (adjust the folder):
```
schtasks /create /tn "GateVision Engine" /tr "C:\gate-vision\run_engine.bat" /sc onstart /ru SYSTEM /rl HIGHEST
schtasks /create /tn "GateVision Web"    /tr "C:\gate-vision\run_web.bat"    /sc onstart /ru SYSTEM /rl HIGHEST
```
Also enable "restore after power failure" in the PC's BIOS.

## 6. Access from outside the society

Do **not** forward a router port to this PC. Install **Tailscale** (free) on the PC and on your phone, then open
`http://<tailscale-ip-of-PC>:8080`. The login page protects it on top of that.

## 7. Tuning (config.yaml, section `tracking`)

| Problem | Setting |
| --- | --- |
| Far-away people/cars/trees are logged | raise `min_person_height` (0.40 -> 0.55) or `min_vehicle_width` (0.30 -> 0.40); per camera: `min_height`, `min_width` |
| Saved face / plate photos look too dull | they are enhanced automatically (set `enhance: {faces: false}` in config.yaml to turn off) |
| Real people near the camera are missed | lower `min_person_height`, or set `require_face: false` on that camera |
| Cars on the public road behind the barrier are logged | give that camera a `roi` (box on the lower part of the picture) |
| Vehicles are skipped because they barely pause | raise `stop_motion_ratio` (0.08 -> 0.15) |
| Plates misread | `max_ocr_attempts` up, check shutter/glare; send me sample `*_result.jpg` files |
| Same vehicle logged twice | raise `dedup_seconds` |
| PC overloaded | lower camera fps, `process_fps` to 4, `detector_imgsz` to 640 |

## 8. Honest limits

* Plate reading is never 100 %. Search has a **"Similar plates"** option that tolerates one wrong character
  (0/O, 1/I, 8/B, 5/S ...). The plate **photo** is always saved, so a human can read it.
* Pakistani plates are not special-cased by the OCR model. Plate text is corrected by pattern
  (letters then digits). Once we have a few hundred real gate photos we can fine-tune the plate model.
* Colours come from simple image analysis and are approximate (night, strong sun, glare).
* Every record keeps its time, so you can always jump to the NVR video for proof.

## 9. Later phases

Face search by uploading a photo, watch-list alerts, linking a vehicle record with the faces seen at the same
moment, and fine-tuning the plate model on your own gate images.

## Tests

`python -m unittest discover -s tests -t .`  (simulated cameras; no GPU or models needed)
