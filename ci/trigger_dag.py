#!/usr/bin/env python3
"""Trigger the course Airflow DAG delivery_capstone_workflow (deploy-hook step).

The course Airflow REST API rejects HTTP basic auth, so we do the Flask-AppBuilder
session login (fetch CSRF -> POST /login) and then POST a dag run. Credentials come
from ~/.deploy-secrets (KEY=value, chmod 600) — never from the repo. Exits non-zero
on any failure so the hook can report it.

~/.deploy-secrets keys:
    AIRFLOW_URL=http://localhost:33013
    AIRFLOW_USER=delivery
    AIRFLOW_PASS=...
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import requests

DAG_ID = "delivery_capstone_workflow"


def load_secrets(path: Path) -> dict:
    """Parse the KEY=VALUE ~/.deploy-secrets file (see import_grafana for the rationale)."""
    secrets: dict[str, str] = {}
    if not path.exists():
        sys.exit(f"trigger_dag: secrets file {path} not found")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        secrets[k.strip()] = v.strip()
    return secrets


def main() -> None:
    s = load_secrets(Path(os.path.expanduser("~/.deploy-secrets")))
    base = s.get("AIRFLOW_URL", "http://localhost:33013").rstrip("/")
    user, pw = s.get("AIRFLOW_USER"), s.get("AIRFLOW_PASS")
    if not user or not pw:
        sys.exit("trigger_dag: AIRFLOW_USER/AIRFLOW_PASS missing in ~/.deploy-secrets")

    # The course Airflow's REST API rejects plain HTTP basic auth, so we authenticate the
    # way the *web UI* does: log in through Flask-AppBuilder to obtain a session cookie.
    # A shared Session object carries that cookie onto the subsequent API call.
    sess = requests.Session()
    sess.headers["User-Agent"] = "delivery-deploy-hook"
    # FAB protects its login form with a CSRF token embedded in the page HTML. Fetch the
    # login page, scrape the token out of it, and post it back with the credentials — a
    # login POST without the matching token would be rejected.
    login = sess.get(base + "/login/", timeout=20)
    tok = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login.text)
    data = {"username": user, "password": pw}
    if tok:
        data["csrf_token"] = tok.group(1)
    sess.post(base + "/login/", data=data, timeout=20)

    # Now authenticated, POST a new DAG run. We supply our own dag_run_id (timestamped) so
    # it's unique per trigger and, crucially, so the hook can hand it to watch_dag.py to
    # reconcile this exact run's outcome later. 200/201 both mean "run created".
    run_id = f"deploy_hook__{int(time.time())}"
    r = sess.post(
        f"{base}/api/v1/dags/{DAG_ID}/dagRuns",
        json={"dag_run_id": run_id},
        timeout=25,
    )
    if r.status_code not in (200, 201):
        sys.exit(f"trigger_dag: POST dagRuns -> {r.status_code} {r.text[:200]}")
    print(f"trigger_dag: started {DAG_ID} run {run_id}")

    # Hand the run id back so the deploy hook can record it and the retrain
    # watcher can reconcile the run's real outcome later.
    rid_file = os.getenv("DAG_RUN_ID_FILE")
    if rid_file:
        Path(rid_file).write_text(run_id)


if __name__ == "__main__":
    main()
