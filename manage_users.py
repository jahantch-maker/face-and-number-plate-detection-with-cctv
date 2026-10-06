"""Create and manage web logins.

    python manage_users.py add manager1 --role manager
    python manage_users.py add admin    --role admin
    python manage_users.py add guard1   --role guard
    python manage_users.py list
    python manage_users.py delete guard1

Roles: admin and manager see everything (search, playback info, camera status);
guard sees only the live list of the latest events.
"""
import argparse
import getpass
from pathlib import Path

from werkzeug.security import generate_password_hash

from gatevision.config import load_config
from gatevision.db import Database

ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["add", "list", "delete"])
ap.add_argument("username", nargs="?")
ap.add_argument("--role", choices=["admin", "manager", "guard"], default="manager")
ap.add_argument("--password")
ap.add_argument("--config", default="config.yaml")
a = ap.parse_args()

cfg = load_config(a.config)
db = Database(Path(cfg["storage"]["data_dir"]) / "gatevision.db")

if a.action == "list":
    for u in db.list_users():
        print(f"{u['username']:20s} {u['role']}")
elif not a.username:
    ap.error("username required")
elif a.action == "delete":
    print("deleted" if db.delete_user(a.username) else "no such user")
else:
    pw = a.password or getpass.getpass("Password: ")
    if len(pw) < 8:
        raise SystemExit("Use a password of at least 8 characters.")
    db.add_user(a.username, generate_password_hash(pw), a.role)
    print(f"user '{a.username}' saved with role {a.role}")
