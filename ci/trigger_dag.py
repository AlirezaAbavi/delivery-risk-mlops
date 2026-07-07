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

    sess = requests.Session()
    sess.headers["User-Agent"] = "delivery-deploy-hook"
    login = sess.get(base + "/login/", timeout=20)
    tok = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login.text)
    data = {"username": user, "password": pw}
    if tok:
        data["csrf_token"] = tok.group(1)
    sess.post(base + "/login/", data=data, timeout=20)

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
