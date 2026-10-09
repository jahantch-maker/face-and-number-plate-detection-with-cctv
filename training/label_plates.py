"""A small web page for typing the correct plate under each training photo.

    label_plates.bat                 (double-click; same as below)
    python training/label_plates.py [--port 8090]

Then open http://<server>:8090 (over Tailscale from the phone works) and log
in with a Gate Vision admin or manager account. Each photo shows what Gate
Vision read; fix it if it is wrong and press Enter. Bikes and rickshaws come
first, because those are what we most need to teach.
"""
from __future__ import annotations

import argparse
import io
import threading
import time
import zipfile
from functools import wraps

from flask import (Flask, Response, abort, redirect, render_template_string, request, send_file,
                   send_from_directory, url_for)

from common import NOT_PLATE, UNREADABLE, clean_label, data_dir, has_text, plates_dir, read_labels, write_labels

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Plate labelling</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#f4f5f7;color:#111}
 .box{max-width:640px;margin:auto}
 img{width:100%;image-rendering:auto;background:#000;border-radius:8px}
 input[type=text]{font-size:28px;width:100%;box-sizing:border-box;padding:10px;letter-spacing:2px;
   text-transform:uppercase;border:2px solid #888;border-radius:8px;margin:12px 0}
 button{font-size:18px;padding:12px 16px;border-radius:8px;border:0;margin:4px 4px 4px 0}
 .save{background:#1a7f37;color:#fff}.bad{background:#b42318;color:#fff}.skip{background:#ddd}
 .none{background:#6e40c9;color:#fff}
 .muted{color:#666;font-size:14px}
 a{color:#0550ae}
</style></head><body><div class="box">
<p class="muted">{{done}} labelled, {{todo}} to go ({{bikes_todo}} bikes / rickshaws).
 Unreadable so far: {{bad}}. Not a plate: {{not_plate}}.</p>
{% if row %}
 <img src="{{url_for('image', name=row.file)}}" alt="plate">
 <p class="muted">{{row.vehicle or 'vehicle'}} &middot; {{row.camera}} &middot; {{row.time}}
  &middot; Gate Vision read: <b>{{row.guess or 'nothing'}}</b></p>
 <form method="post" action="{{url_for('save')}}" autocomplete="off">
  <input type="hidden" name="file" value="{{row.file}}">
  <input type="text" name="label" value="{{row.label if row.label and row.label != '-' else row.guess}}" autofocus
    autocapitalize="characters" spellcheck="false">
  <div class="muted">Letters and the big number only, no dash and no small year: LEA-17-5989 is LEA5989.
   <b>Can't read it</b> = it is a plate but too blurry. <b>Not a plate</b> = feet, lamp, cargo, anything else.</div>
  <button class="save" name="action" value="save">Save (Enter)</button>
  <button class="bad" name="action" value="bad">Can't read it</button>
  <button class="none" name="action" value="none">Not a plate</button>
  <button class="skip" name="action" value="skip">Skip</button>
 </form>
{% else %}
 <h2>All photos are labelled.</h2>
 <p>Run collect_plates.bat again in a few days to get new photos.</p>
{% endif %}
<p class="muted"><a href="{{url_for('download')}}">Download all photos and labels (zip)</a></p>
{% if recent %}<p class="muted">Recently labelled (tap to fix):
 {% for r in recent %}<a href="{{url_for('index', file=r.file)}}">{{r.label}}</a> {% endfor %}</p>{% endif %}
</div></body></html>"""

BIKES = ("motorcycle", "bicycle")


def create_app():
    from werkzeug.security import check_password_hash
    from gatevision.db import Database

    app = Flask(__name__)
    folder = plates_dir()
    labels_path = folder / "labels.csv"
    db = Database(data_dir() / "gatevision.db")
    lock = threading.Lock()
    skipped: set[str] = set()

    def auth(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            cred = request.authorization
            u = db.get_user(cred.username) if cred and cred.username else None
            if not (u and u["role"] in ("admin", "manager") and check_password_hash(u["password_hash"], cred.password or "")):
                return Response("Log in with a Gate Vision admin or manager account.", 401,
                                {"WWW-Authenticate": 'Basic realm="Gate Vision labelling"'})
            return fn(*a, **kw)
        return wrapper

    @app.get("/")
    @auth
    def index():
        rows = read_labels(labels_path)
        todo = [r for r in rows if not r["label"]]
        want = request.args.get("file")
        row = next((r for r in rows if r["file"] == want), None) if want else None
        if row is None:
            fresh = [r for r in todo if r["file"] not in skipped] or todo
            # bikes and rickshaws first: those are the plates the reader gets wrong
            fresh.sort(key=lambda r: (r["vehicle"] not in BIKES, r["time"]))
            row = fresh[0] if fresh else None
        recent = [r for r in rows if has_text(r["label"])][-12:][::-1]
        return render_template_string(
            PAGE, row=row, recent=recent, todo=len(todo),
            done=sum(1 for r in rows if has_text(r["label"])),
            bad=sum(1 for r in rows if r["label"] == UNREADABLE),
            not_plate=sum(1 for r in rows if r["label"] == NOT_PLATE),
            bikes_todo=sum(1 for r in todo if r["vehicle"] in BIKES))

    @app.post("/save")
    @auth
    def save():
        name, action = request.form.get("file", ""), request.form.get("action", "save")
        if action == "skip":
            skipped.add(name)
            return redirect(url_for("index"))
        label = {"bad": UNREADABLE, "none": NOT_PLATE}.get(action) or clean_label(request.form.get("label", ""))
        if not label:
            label = UNREADABLE
        with lock:
            rows = read_labels(labels_path)
            for r in rows:
                if r["file"] == name:
                    r["label"] = label
                    break
            else:
                abort(404)
            write_labels(labels_path, rows)
        return redirect(url_for("index"))

    @app.get("/download")
    @auth
    def download():
        """Everything in one zip, e.g. to send for training: labels.csv, the
        plate photos (images/) and, while Gate Vision still has them, the
        vehicle photos they came from (vehicles/), to check the plate finder."""
        buf = io.BytesIO()
        ddir = data_dir()
        with lock, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(labels_path, "labels.csv")
            for p in sorted((folder / "images").glob("*.jpg")):
                z.write(p, f"images/{p.name}")
                row = db.get_event(int(p.stem)) if p.stem.isdigit() else None
                src = ddir / row["crop_path"] if row is not None and row["crop_path"] else None
                if src is not None and src.is_file():
                    z.write(src, f"vehicles/{p.name}")
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True,
                         download_name=f"gate_plates_{time.strftime('%Y%m%d')}.zip")

    @app.get("/img/<path:name>")
    @auth
    def image(name):
        return send_from_directory(folder / "images", name)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    if not (plates_dir() / "labels.csv").exists():
        print("No training photos yet: run collect_plates.bat first.")
        return
    from waitress import serve
    print(f"Labelling page: http://localhost:{args.port}  (or http://<server address>:{args.port} from the phone)")
    print("Close this window to stop it.")
    serve(create_app(), host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
