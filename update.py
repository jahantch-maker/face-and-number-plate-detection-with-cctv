"""One-click updater: pulls the latest Gate Vision from GitHub into this folder.

    update.bat                 normal update (use this)
    update.bat --force         re-download and re-apply even if already latest
    update.bat --deps          also re-run the package installer
    update.bat --rollback      go back to the version before the last update
    update.bat --zip file.zip  update from a zip you downloaded by hand (offline)

Never touched by an update: config.yaml, data\\, models\\, .venv\\, *.pt model
files, logs, gpu_mode.txt. (Your recordings folder, if it is outside this
folder, is never touched either.)

Exit code: 0 = updated (update.bat then restarts the system), 2 = already
up to date, 1 = failed (nothing was changed or the old version was restored).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

try:  # use the Windows certificate store, like the rest of Gate Vision
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

OWNER = "jahantch-maker"
REPO = "face-and-number-plate-recognition-with-cctv"
BRANCH = "main"
API = f"https://api.github.com/repos/{OWNER}/{REPO}/commits/{BRANCH}"
ZIP_URL = f"https://codeload.github.com/{OWNER}/{REPO}/zip/refs/heads/{BRANCH}"

ROOT = Path(__file__).resolve().parent
VERSION_FILE = ROOT / ".version"
MANIFEST_FILE = ROOT / ".manifest.json"
BACKUP_DIR = ROOT / "_backup"
KEEP_BACKUPS = 5

# Top-level names an update must never overwrite or delete.
PRESERVE_NAMES = {
    "config.yaml", "data", "models", ".venv", "_backup", ".git",
    "gpu_mode.txt", "update.bat", ".version", ".manifest.json",
}
# Patterns that are preserved anywhere in the tree.
PRESERVE_PATTERNS = ("*.pt", "*.log", "probe_*.jpg", "*_result.jpg", "*.pyc")
SKIP_DIRS = {"__pycache__"}


def say(msg: str = "") -> None:
    print(" " + msg, flush=True)


def is_preserved(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    if parts[0] in PRESERVE_NAMES or any(p in SKIP_DIRS for p in parts):
        return True
    return any(fnmatch.fnmatch(parts[-1], pat) for pat in PRESERVE_PATTERNS)


def read_version() -> str:
    try:
        return VERSION_FILE.read_text().strip()
    except OSError:
        return ""


def latest_sha() -> str | None:
    try:
        req = urllib.request.Request(API, headers={"User-Agent": "gatevision-updater"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)["sha"]
    except Exception as exc:  # rate limit, offline, ...
        say(f"(could not check the latest version: {exc})")
        return None


def download(dest: Path) -> None:
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "gatevision-updater"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def extract_tree(zip_path: Path, out: Path) -> Path:
    """Extract a GitHub source zip; return the folder that holds the project."""
    out_real = out.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            target = (out / name).resolve()
            if target != out_real and out_real not in target.parents:
                raise RuntimeError(f"unsafe path in zip: {name}")
        zf.extractall(out)
    tops = [p for p in out.iterdir() if p.is_dir()]
    if len(tops) == 1 and not any(p.is_file() for p in out.iterdir()):
        return tops[0]
    return out


def tree_files(base: Path) -> list[str]:
    files = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            if not is_preserved(rel):
                files.append(rel)
    return files


def deps_hash(base: Path) -> str:
    h = hashlib.sha256()
    for name in ("requirements.txt", "install_deps.py"):
        p = base / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def stop_services() -> None:
    """Stop the engine and web server (and their auto-restart wrapper loops)."""
    if os.name != "nt":
        return
    ps = (
        "$me=%d; $par=(Get-CimInstance Win32_Process -Filter \"ProcessId=$me\").ParentProcessId;"
        "$all=Get-CimInstance Win32_Process;"
        # wrappers first, otherwise they restart the python process after 10 s
        "$all|?{$_.Name -eq 'cmd.exe' -and $_.CommandLine -match 'run_(engine|web)\\.bat' -and $_.ProcessId -ne $par}"
        "|%%{Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue};"
        "$root='%s';"
        "$all|?{$_.ExecutablePath -like ($root+'\\.venv\\*') -and $_.ProcessId -ne $me -and $_.ProcessId -ne $par "
        "-and $_.CommandLine -match 'run_(engine|web)\\.py'}"
        "|%%{Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}"
    ) % (os.getpid(), str(ROOT).replace("'", "''"))
    subprocess.call(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps])
    time.sleep(2)


def make_backup(old_manifest: list[str], new_files: list[str], tag: str) -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    path = BACKUP_DIR / f"before-{time.strftime('%Y%m%d-%H%M%S')}-{tag}.zip"
    n = 1
    while path.exists():   # two updates in the same second
        n += 1
        path = BACKUP_DIR / f"before-{time.strftime('%Y%m%d-%H%M%S')}-{n}-{tag}.zip"
    added = [f for f in new_files if not (ROOT / f).exists()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(set(old_manifest) | set(new_files)):
            src = ROOT / rel
            if src.is_file() and not is_preserved(rel):
                zf.write(src, rel)
        zf.writestr("_added.json", json.dumps(added))
    for old in sorted(BACKUP_DIR.glob("before-*.zip"), key=lambda f: f.stat().st_mtime_ns)[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return path


def apply_tree(src: Path, new_files: list[str], old_manifest: list[str]) -> tuple[int, int]:
    changed = 0
    for rel in new_files:
        s, d = src / rel, ROOT / rel
        if rel == "update.py":
            # we are running from this file; replace it last, at the very end
            continue
        if d.is_file() and d.read_bytes() == s.read_bytes():
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, d)
        changed += 1
    removed = 0
    for rel in old_manifest:
        if rel not in new_files and not is_preserved(rel):
            f = ROOT / rel
            if f.is_file():
                f.unlink()
                removed += 1
    # self-update last (Python has already loaded this file, so this is safe)
    if "update.py" in new_files:
        s, d = src / "update.py", ROOT / "update.py"
        if not d.is_file() or d.read_bytes() != s.read_bytes():
            shutil.copyfile(s, d)
            changed += 1
    return changed, removed


def venv_python() -> str:
    for cand in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return sys.executable


def install_deps() -> bool:
    say("Updating Python packages (only what is missing; can take a few minutes)...")
    return subprocess.call([venv_python(), "install_deps.py", "--update"], cwd=ROOT) == 0


def do_update(zip_file: str | None, force: bool, deps: bool) -> int:
    current = read_version()
    new_sha = None
    if not zip_file:
        say("Checking GitHub for a newer version...")
        new_sha = latest_sha()
        if new_sha and new_sha == current and not force:
            say(f"Already up to date (version {current[:7]}).")
            return 2
    tmp = Path(tempfile.mkdtemp(prefix="gv_update_"))
    try:
        zpath = Path(zip_file) if zip_file else tmp / "latest.zip"
        if not zip_file:
            say("Downloading the latest version...")
            try:
                download(zpath)
            except Exception as exc:
                say(f"Download failed: {exc}")
                say("Check the internet connection. Nothing was changed.")
                return 1
        try:
            src = extract_tree(zpath, tmp / "x")
        except (RuntimeError, zipfile.BadZipFile) as exc:
            say(f"Bad update file ({exc}). Nothing was changed.")
            return 1
        if not (src / "run_engine.py").exists():
            say("That download does not look like Gate Vision. Nothing was changed.")
            return 1
        new_files = tree_files(src)
        try:
            old_manifest = json.loads(MANIFEST_FILE.read_text())
        except (OSError, ValueError):
            old_manifest = []
        old_deps = ""
        if (ROOT / "requirements.txt").exists():
            old_deps = deps_hash(ROOT)

        say("Stopping Gate Vision for a moment...")
        stop_services()
        backup = make_backup(old_manifest, new_files, current[:7] or "old")
        say(f"Old version saved: {backup.name}")
        changed, removed = apply_tree(src, new_files, old_manifest)
        say(f"Files updated: {changed}, obsolete files removed: {removed}")

        MANIFEST_FILE.write_text(json.dumps(new_files))
        VERSION_FILE.write_text(new_sha or "manual")
        if deps or force or old_deps != deps_hash(ROOT):
            if not install_deps():
                say("WARNING: package install had problems. Run update.bat --deps once "
                    "the internet is stable, or run install.bat.")
        say(f"Updated to version {(new_sha or 'manual')[:7]}.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def do_rollback() -> int:
    backups = sorted(BACKUP_DIR.glob("before-*.zip"), key=lambda f: f.stat().st_mtime_ns)
    if not backups:
        say("No backup found - nothing to roll back to.")
        return 1
    path = backups[-1]
    say(f"Rolling back using {path.name} ...")
    stop_services()
    with zipfile.ZipFile(path) as zf:
        added = json.loads(zf.read("_added.json")) if "_added.json" in zf.namelist() else []
        restored = []
        for name in zf.namelist():
            if name == "_added.json" or is_preserved(name) or name == "update.py":
                continue
            target = (ROOT / name).resolve()
            if ROOT not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
            restored.append(name)
    for rel in added:
        f = ROOT / rel
        if f.is_file() and not is_preserved(rel):
            f.unlink()
    MANIFEST_FILE.write_text(json.dumps(restored))
    VERSION_FILE.write_text("rolled-back")
    path.rename(path.with_suffix(".used"))
    say("Rolled back to the previous version.")
    return 0


def main(argv: list[str]) -> int:
    zip_file = None
    if "--zip" in argv:
        i = argv.index("--zip")
        if i + 1 >= len(argv):
            say("--zip needs a file name")
            return 1
        zip_file = argv[i + 1]
    print()
    say("Gate Vision updater")
    say("===================")
    if "--rollback" in argv:
        return do_rollback()
    return do_update(zip_file, "--force" in argv, "--deps" in argv)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        say(f"Update failed: {exc}")
        sys.exit(1)
