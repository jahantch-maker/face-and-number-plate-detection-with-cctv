"""Web app: login, search/filter gallery, live list, event detail, camera status."""
from __future__ import annotations

import functools
import secrets
import json
import time
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.security import check_password_hash

from ..colors import NAMES as COLOR_NAMES
from ..config import web_secret
from ..db import Database

VEHICLE_TYPES = ["car", "motorcycle", "bus", "truck", "bicycle"]
MAX_FAILS, LOCK_SECONDS = 5, 60


def _parse_local(text: str | None):
    """'2026-10-06T14:30' (browser datetime-local) -> epoch seconds, or None."""
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _to_local_input(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M", time.localtime(epoch))


def event_to_dict(r) -> dict:
    d = dict(r)
    d.pop("extra", None)
    return d


def create_app(cfg: dict, db: Database | None = None) -> Flask:
    data_dir = Path(cfg["storage"]["data_dir"])
    db = db or Database(data_dir / "gatevision.db")
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=web_secret(cfg),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=60 * 60 * 12,
    )
    cameras = cfg["cameras"]
    page_size = cfg["web"]["page_size"]
    fails: dict[str, list] = {}   # ip -> [count, locked_until]

    # ------------------------------------------------------------ helpers
    def login_required(roles=None):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(*a, **kw):
                if "user" not in session:
                    if request.path.startswith("/api/"):
                        return jsonify(error="login required"), 401
                    return redirect(url_for("login", next=request.path))
                if roles and session.get("role") not in roles:
                    if session.get("role") == "guard":
                        return redirect(url_for("live"))
                    abort(403)
                return fn(*a, **kw)
            return wrapper
        return deco

    def csrf_token() -> str:
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    def check_csrf():
        if not secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "x")):
            abort(400, "bad form token - reload the page")

    app.jinja_env.globals.update(csrf_token=csrf_token, COLOR_NAMES=COLOR_NAMES)
    app.jinja_env.filters["media"] = lambda p: url_for("media", rel=p) if p else ""

    # --------------------------------------------------------------- auth
    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        ip = request.remote_addr or "?"
        if request.method == "POST":
            check_csrf()
            count, locked = fails.get(ip, [0, 0.0])
            if locked > time.time():
                error = "Too many attempts. Wait a minute and try again."
            else:
                u = db.get_user(request.form.get("username", "").strip())
                if u and check_password_hash(u["password_hash"], request.form.get("password", "")):
                    fails.pop(ip, None)
                    session.clear()
                    session.permanent = True
                    session["user"], session["role"] = u["username"], u["role"]
                    nxt = request.args.get("next", "")
                    if not nxt.startswith("/") or nxt.startswith("//"):
                        nxt = url_for("live" if u["role"] == "guard" else "search")
                    return redirect(nxt)
                count += 1
                fails[ip] = [count, time.time() + LOCK_SECONDS if count >= MAX_FAILS else 0.0]
                if count >= MAX_FAILS:
                    fails[ip][0] = 0
                error = "Wrong username or password."
        return render_template("login.html", error=error)

    @app.post("/logout")
    def logout():
        check_csrf()
        session.clear()
        return redirect(url_for("login"))

    # -------------------------------------------------------------- pages
    @app.route("/")
    @login_required(roles=("admin", "manager"))
    def search():
        a = request.args
        submitted = "kind" in a or "ts_from" in a
        ts_from = _parse_local(a.get("ts_from")) if submitted else time.time() - 86400
        ts_to = _parse_local(a.get("ts_to")) if submitted else None
        f = {
            "plate": a.get("plate", "").strip(), "fuzzy": a.get("fuzzy") == "1",
            "kind": a.get("kind", ""), "direction": a.get("direction", ""),
            "camera_id": a.get("camera_id", ""), "vehicle_type": a.get("vehicle_type", ""),
            "color": a.get("color", ""), "upper_color": a.get("upper_color", ""),
            "lower_color": a.get("lower_color", ""), "ts_from": ts_from, "ts_to": ts_to,
        }
        # a plate query only makes sense for vehicles
        if f["plate"] and not f["kind"]:
            f["kind"] = "vehicle"
        page = max(1, int(a.get("page", 1) if a.get("page", "1").isdigit() else 1))
        rows, total = db.search(f, limit=page_size, offset=(page - 1) * page_size)
        form = dict(f)
        form["ts_from"] = _to_local_input(ts_from) if ts_from else ""
        form["ts_to"] = _to_local_input(ts_to) if ts_to else ""
        args = {k: v for k, v in request.args.items() if k != "page"}
        if not submitted:
            args = {}
        return render_template(
            "search.html", rows=rows, total=total, page=page, pages=max(1, -(-total // page_size)),
            f=form, cameras=cameras, vehicle_types=VEHICLE_TYPES, args=args, page_size=page_size)

    @app.route("/live")
    @login_required()
    def live():
        rows = db.latest_events(30)
        return render_template("live.html", rows=rows,
                               last_id=rows[0]["id"] if rows else 0)

    @app.route("/event/<int:event_id>")
    @login_required(roles=("admin", "manager"))
    def event(event_id):
        r = db.get_event(event_id)
        if r is None:
            abort(404)
        return render_template("event.html", e=r, window=(r["ts"] - 120, r["ts"] + 120),
                               local=_to_local_input)

    @app.route("/status")
    @login_required(roles=("admin", "manager"))
    def status():
        now = time.time()
        rows = []
        for s in db.camera_statuses():
            d = dict(s)
            d["online"] = bool(d["last_frame"]) and now - d["last_frame"] < 30 and now - d["updated"] < 45
            d["age"] = int(now - d["last_frame"]) if d["last_frame"] else None
            try:
                counts = json.loads(d.get("skips") or "{}").get("counts", {})
            except ValueError:
                counts = {}
            d["skip_counts"] = sorted(counts.items(), key=lambda kv: (kv[0] != "SAVED", -kv[1]))
            rows.append(d)
        return render_template("status.html", rows=rows)

    # -------------------------------------------------------------- media
    @app.route("/media/<path:rel>")
    @login_required()
    def media(rel):
        resp = send_from_directory(data_dir, rel, max_age=86400)
        return resp

    # ---------------------------------------------------------------- api
    @app.route("/api/events")
    @login_required()
    def api_events():
        after = request.args.get("after_id", "0")
        rows = db.latest_events(30, int(after) if after.isdigit() else 0)
        out = []
        for r in rows:
            d = event_to_dict(r)
            for k in ("full_path", "crop_path", "plate_path", "face_path"):
                d[k] = url_for("media", rel=d[k]) if d[k] else None
            out.append(d)
        return jsonify(events=out)

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "same-origin"
        return resp

    return app
