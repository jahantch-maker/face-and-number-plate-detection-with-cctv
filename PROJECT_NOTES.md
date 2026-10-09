# Gate Vision - project hand-over notes

Written 2026-10-09 so a new chat (or developer) can continue without the old conversation.
No passwords or camera credentials are in this file or in the repo; they live only in the server's `config.yaml`.

## What it is
Local, subscription-free system for a housing-society gate (Dahua NVRs/cameras). Reads the cameras' RTSP streams,
and for every person and vehicle near a camera saves one record: plate text + plate photo, vehicle type and colour,
face photo, clothing colours, time, camera, IN/OUT. Management searches/filters them in a web page and a phone app.
Owner: Jahanzaib (Faisalabad). Brand name of the product: **Gate Vision**.

## Where things run
- **Production:** the owner's Windows Server (2016/2019/2022) in `G:\gate-vision`. No NVIDIA GPU (CPU mode). Reached by
  Remote Desktop. Runs 24/7 via start-up scheduled tasks (`run_engine.bat`, `run_web.bat`). Data is in `G:\gate-vision\data`
  (relative `data_dir: data`).
- It started on a laptop (`C:\gate-vision`), then moved to the server. The server copy is the real one.
- Cameras: 4 streams from one NVR (`in_plate`, `in_face`, `out_plate`, `out_face`); 32 cameras / 8 TB on the NVR in total.
- Retention: 30 days of events/photos (configurable). About 200 people and 100 vehicles a day.
- Remote access: **Tailscale** on the server and the owner's phone (same Google account). Do not port-forward the router.
  On the server enable "Run unattended" and "Disable key expiry" for the server in the Tailscale console.

## Code map
- `run_engine.py` - reads cameras, detects, tracks, saves events. `gatevision/pipeline.py`, `detect.py` (YOLO11 + ByteTrack, needs `lap`),
  `plates.py` (fast-alpr: plate detector + OCR), `faces.py` (YuNet), `colors.py`, `stream.py` (RTSP with back-off), `db.py` (SQLite), `storage.py`.
- `run_web.py` + `gatevision/web/` - Flask + waitress on port 8080: login, search, live list, event detail, camera status.
- Roles: `admin`, `manager` (everything), `guard` (live list only). Users via `manage_users.py`.
- **Mobile API** (`/api/v1/...` in `gatevision/web/app.py`): token login (`POST /api/v1/login`, 90-day signed Bearer token,
  same lockout as the web login), `meta`, `events` (live, `after_id`), `search`, `events/<id>`, `status`. Images via `/media/...` with the same token.
- `config.example.yaml` -> `config.yaml` (never committed). New settings have defaults in `gatevision/config.py`.
- Tests: `python -m unittest discover -s tests -t .` (cameras are simulated; no GPU needed). Includes `test_api.py`, `test_update.py`.

## Install / update
- First install: `install.bat` (+ `setup_wizard.py`), then `start_all.bat`.
- **Updating the server: double-click `update.bat`** (run as Administrator). `update.py` downloads the latest `main` zip from GitHub,
  stops the services, backs up the old version into `_backup\`, replaces program files, installs new packages when
  `requirements.txt`/`install_deps.py` changed, restarts and opens the browser. It never touches `config.yaml`, `data\`, `models\`,
  `.venv\`, `*.pt`, logs, `gpu_mode.txt`, and it skips `android\` and `.github\`. `update.bat --rollback` undoes the last update.
  Other flags: `--force`, `--deps`, `--zip file.zip`.

## Android app (`android/`)
- Kotlin + Jetpack Compose, package `com.gatevision.app`, minSdk 26. Screens: sign-in, Live (polls every 2 s while open), Search
  (plate with "similar plates", who, direction, time range/day, camera, vehicle type, colours), Detail (zoomable photos, "find this plate"),
  Cameras. Guards see only Live. Plain `http://` is allowed because traffic runs over Tailscale.
- Build: GitHub Actions workflow `.github/workflows/android.yml` builds `app-release.apk` on every change under `android/` and
  publishes it as the **`android-latest` release** (`GateVision.apk`). The APK is signed with `android/gatevision-release.jks`
  (committed on purpose, private-use app) so new builds install as updates.
- **STATUS: the app has never been compiled.** The cloud workspace used to write it has no Android SDK and cannot reach
  Maven/Gradle/Google repos or the GitHub Actions API, so the first Actions run may fail. If it does, paste the red error lines from the
  Actions log and fix them (expect a round or two). Then test on a real phone: login over Tailscale, live list, search, detail.

## Decisions and constraints
- No monthly fees or subscriptions anywhere; open-source models only. No cloud AI.
- Capture people/vehicles that are **near** the camera (size rules in `tracking:` config), also while moving, at most 3 photos each.
- Phase 1 = capture + filters. Later: face search by uploading a photo, watch-list alerts, linking vehicles with faces, push notifications in the app,
  live video in the app (heavy over mobile data; not planned for v1).
- Plate OCR is not Pakistani-specific yet; the plate photo is always saved so a human can read it.

## Next steps
1. Get the Android build green and install the APK; fix whatever the first build/test on a phone reveals.
2. **Train/fine-tune the plate model on Pakistani plates** (owner planned this after ~2 days of real gate photos in `data\images`).
   No NVIDIA GPU on the server, so train in the cloud workspace or on a rented GPU and ship the model file; `benchmark_plates.py` and
   `diagnose_plate.py` help measure accuracy. Needs labelled images: either the gate's own crops (label by hand) or a public dataset.
3. Optional ideas: push alerts for watch-listed plates, nightly auto-update (deliberately not enabled), in-app APK update check.

## Gotchas for a future session
- The cloud workspace can't reach PyPI/npm/apt; it can reach GitHub (git push works). RAR files can be extracted with the system `libarchive`
  through `ctypes` (no `unrar` available).
- The repo is public: never commit `config.yaml`, camera passwords, or anything from `data\`.
- Windows batch files (`update.bat`, `install.bat`, `start_all.bat`) were written without being able to run Windows here - test changes on the server.
