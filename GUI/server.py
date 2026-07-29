from __future__ import annotations

import json
import mimetypes
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "projections.json"
MODEL_DIR = BASE_DIR / "models"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8081"))

DATA = json.loads(DATA_PATH.read_text(encoding="utf-8"))

REQUIRED_MODELS = [
    "saidi_high_event_classifier.cbm",
    "saidi_high_regressor.cbm",
    "saidi_normal_regressor.cbm",
    "saifi_log_catboost.cbm",
    "drought_anova_ridge.joblib",
    "compliance_tweedie.cbm",
]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format, *args):
        print(f"[local] {self.address_string()} - {format % args}")

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/healthz":
            return self.send_json({
                "status": "ok",
                "latest_observed_year": DATA["bootstrap"]["latest_year"],
                "model_files_present": all((MODEL_DIR / name).exists() for name in REQUIRED_MODELS),
                "mode": "precomputed frozen-model projections",
            })

        if parsed.path == "/api/bootstrap":
            return self.send_json(DATA["bootstrap"])

        if parsed.path == "/api/project":
            query = parse_qs(parsed.query)
            state = query.get("state", [""])[0].upper()
            year = query.get("year", [""])[0]
            if state not in DATA["states"]:
                return self.send_json({"detail": "State not found."}, 404)
            if year not in DATA["projections"][state]:
                return self.send_json({
                    "detail": (
                        f"Year must be from {DATA['bootstrap']['min_year']} "
                        f"to {DATA['bootstrap']['max_year']}."
                    )
                }, 422)

            projection = dict(DATA["projections"][state][year])
            projection["state"] = DATA["states"][state]
            projection["latest_observed_year"] = DATA["bootstrap"]["latest_year"]
            projection["rank"] = DATA["rankings"][year][state]
            projection["series"] = (
                DATA["historical"][state]
                + [
                    {
                        "year": current,
                        "electricity": DATA["projections"][state][str(current)]["electricity"]["score"],
                        "water": DATA["projections"][state][str(current)]["water"]["score"],
                        "electricity_low": DATA["projections"][state][str(current)]["electricity"]["range"][0],
                        "electricity_high": DATA["projections"][state][str(current)]["electricity"]["range"][1],
                        "water_low": DATA["projections"][state][str(current)]["water"]["range"][0],
                        "water_high": DATA["projections"][state][str(current)]["water"]["range"][1],
                        "type": "projected",
                    }
                    for current in range(DATA["bootstrap"]["min_year"], int(year) + 1)
                ]
            )
            return self.send_json(projection)

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()


def open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        print()
        print(f"Could not start localhost on port {PORT}.")
        print("Another program may already be using that port.")
        print("Close the other server, or run:")
        print("  set PORT=8082 && py server.py")
        print()
        raise SystemExit(1) from exc

    print()
    print("U.S. INFRASTRUCTURE STRESS MONITOR")
    print("---------------------------------")
    print(f"Website:  http://{HOST}:{PORT}")
    print(f"Health:   http://{HOST}:{PORT}/healthz")
    print(f"Example:  http://{HOST}:{PORT}/api/project?state=CA&year=2030")
    print()
    print("Press Ctrl+C to stop the server.")
    threading.Timer(0.7, open_browser).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
