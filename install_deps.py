"""Installs everything Gate Vision needs. Run by install.bat inside the .venv.

* Detects an NVIDIA graphics card and installs the matching GPU build of
  PyTorch (trying newest to oldest until one actually sees the GPU).
* Falls back to the normal CPU build if there is no GPU or no build works.
* Downloads all AI models now, so the first live run does not stall.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

PIP = [sys.executable, "-m", "pip"]
CUDA_TAGS = ("cu128", "cu126", "cu124", "cu118")


def run(cmd, quiet=False) -> bool:
    kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if quiet else {}
    return subprocess.call(cmd, **kw) == 0


def pip(*args) -> bool:
    return run(PIP + list(args))


def has_nvidia() -> bool:
    return bool(shutil.which("nvidia-smi")) and run(["nvidia-smi"], quiet=True)


def cuda_works() -> bool:
    code = "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    return run([sys.executable, "-c", code], quiet=True)


def main():
    if sys.version_info < (3, 9):
        raise SystemExit("Python is too old. Please install Python 3.12 from python.org.")
    if sys.version_info >= (3, 13):
        print("NOTE: Python %d.%d detected. If a package fails to install, "
              "install Python 3.12 instead and run install.bat again.\n" % sys.version_info[:2])

    print("== 1/4  Updating installer tools")
    pip("install", "--upgrade", "pip")

    print("\n== 2/4  PyTorch (AI engine)")
    gpu = False
    if has_nvidia():
        print("NVIDIA graphics card found - installing the GPU version.")
        for tag in CUDA_TAGS:
            print(f"  trying {tag} ...")
            if pip("install", "--upgrade", "--force-reinstall", "torch", "torchvision",
                   "--index-url", f"https://download.pytorch.org/whl/{tag}") and cuda_works():
                gpu = True
                print(f"  GPU build {tag} is working.")
                break
        if not gpu:
            print("  Could not get a working GPU build (old graphics driver?). "
                  "Update the NVIDIA driver later; continuing with the CPU version.")
    else:
        print("No NVIDIA graphics card found - using the CPU version (slower, fine for a trial).")

    print("\n== 3/4  Other packages")
    if not pip("install", "ultralytics", "fast-alpr[onnx]", "opencv-python", "numpy",
               "flask", "waitress", "pyyaml"):
        raise SystemExit("Package install failed. Check the internet connection and run install.bat again.")
    if not gpu and not cuda_works():
        pass  # ultralytics brought its own CPU torch

    print("\n== 4/4  Downloading AI models (one time, a few minutes)")
    run([sys.executable, "download_models.py"])
    warm = (
        "from gatevision.config import DEFAULTS as D;m=D['models'];"
        "from ultralytics import YOLO;YOLO(m['detector']);"
        "from gatevision.plates import PlateReader;PlateReader(m['plate_detector'],m['plate_ocr']);"
        "print('models ready')"
    )
    if not run([sys.executable, "-c", warm]):
        print("WARNING: could not pre-download all models. They will download on first run "
              "(internet needed then).")

    open("gpu_mode.txt", "w").write("gpu" if gpu else "cpu")
    print("\nInstall finished. Mode:", "GPU" if gpu else "CPU")


if __name__ == "__main__":
    main()
