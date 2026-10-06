"""Start the web app (search, live list, camera status).

    python run_web.py
Then open  http://<this-pc-ip>:8080  from any device on the network.
"""
import argparse
import logging

from gatevision.config import load_config
from gatevision.web.app import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
args = ap.parse_args()

logging.basicConfig(level=logging.INFO)
cfg = load_config(args.config)
app = create_app(cfg)
host, port = cfg["web"]["host"], cfg["web"]["port"]
try:
    from waitress import serve
    print(f"Serving on http://{host}:{port} (waitress)")
    serve(app, host=host, port=port, threads=8)
except ImportError:
    print("waitress not installed - using Flask's development server")
    app.run(host=host, port=port)
