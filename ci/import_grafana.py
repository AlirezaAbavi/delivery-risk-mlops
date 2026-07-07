#!/usr/bin/env python3
"""Import/overwrite the delivery Grafana dashboards (deploy-hook step).

POSTs every grafana/dashboards/*.json to the shared Grafana's /api/dashboards/db
with overwrite=true, into the delivery folder — the exact call used for the initial
manual import, now automatic whenever a dashboard file changes. Grafana basic auth
works for its API. Credentials come from ~/.deploy-secrets (chmod 600).

~/.deploy-secrets keys:
    GRAFANA_URL=http://localhost:3010
    GRAFANA_USER=delivery
    GRAFANA_PASS=...
    GRAFANA_FOLDER=delivery        # optional; defaults to "delivery"
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", os.path.expanduser("~/project")))
DASH_DIR = PROJECT_DIR / "grafana" / "dashboards"


def load_secrets(path: Path) -> dict:
    secrets: dict[str, str] = {}
    if not path.exists():
        sys.exit(f"import_grafana: secrets file {path} not found")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        secrets[k.strip()] = v.strip()
    return secrets


def main() -> None:
    s = load_secrets(Path(os.path.expanduser("~/.deploy-secrets")))
    base = s.get("GRAFANA_URL", "http://localhost:3010").rstrip("/")
    user, pw = s.get("GRAFANA_USER"), s.get("GRAFANA_PASS")
    folder_title = s.get("GRAFANA_FOLDER", "delivery")
    if not user or not pw:
        sys.exit("import_grafana: GRAFANA_USER/GRAFANA_PASS missing in ~/.deploy-secrets")

    files = sorted(DASH_DIR.glob("*.json"))
    if not files:
        print(f"import_grafana: no dashboards in {DASH_DIR}; nothing to do")
        return

    sess = requests.Session()
    sess.auth = (user, pw)
    sess.headers["Content-Type"] = "application/json"

    # Resolve the target folder id (0 = General if not found).
    folder_id = 0
    fr = sess.get(base + "/api/folders", timeout=20)
    fr.raise_for_status()
    for f in fr.json():
        if f.get("title", "").lower() == folder_title.lower():
            folder_id = f.get("id", 0)
            break

    failures = 0
    for path in files:
        dash = json.loads(path.read_text())
        dash.pop("id", None)  # force create/replace-by-uid, never update-by-id
        payload = {
            "dashboard": dash,
            "folderId": folder_id,
            "overwrite": True,
            "message": f"deploy-hook import {path.name}",
        }
        r = sess.post(base + "/api/dashboards/db", data=json.dumps(payload), timeout=25)
        if r.status_code == 200:
            print(f"import_grafana: {path.name} -> {r.json().get('url')}")
        else:
            failures += 1
            print(f"import_grafana: {path.name} FAILED {r.status_code} {r.text[:160]}",
                  file=sys.stderr)

    if failures:
        sys.exit(f"import_grafana: {failures} dashboard(s) failed")


if __name__ == "__main__":
    main()
