#!/usr/bin/env python3
"""Reconcile the *outcome* of the Airflow retrain a deploy triggered.

The deploy hook triggers the retrain DAG fire-and-forget and records
``trigger: queued`` — it deliberately does NOT block for the (multi-minute)
run. This script closes that loop: it looks at the latest deploy record, and if
that deploy queued a DAG run whose terminal state isn't known yet, it polls
Airflow once and, when the run has finished (or aged out), appends the outcome
to ``DEPLOY_RETRAIN_PATH``. The API joins that back to the deploy so the UI can
show the retrain as running → success/failed.

It is driven by the existing deploy timer (called once per tick), NOT a detached
background process: a child of the oneshot deploy service would be killed with
the service's cgroup when it exits. Timer-driven reconciliation is idempotent
and survives restarts — each tick does at most one quick poll.

Reads ~/.deploy-secrets (AIRFLOW_URL/USER/PASS), same as trigger_dag.py. Runs
quiet on no-op; best-effort, exits 0 on its own errors so it never disturbs a
deploy tick.

Env:
    DEPLOY_RUNS_PATH        deploy run records (default ~/deploy-runs.jsonl)
    DEPLOY_RETRAIN_PATH     retrain outcome records (default ~/deploy-retrain.jsonl)
    RETRAIN_TIMEOUT_SECONDS give up waiting after this (default 1800 = 30 min)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DAG_ID = "delivery_capstone_workflow"
TERMINAL = {"success", "failed"}


def _runs_path() -> Path:
    return Path(os.path.expanduser(os.getenv("DEPLOY_RUNS_PATH", "~/deploy-runs.jsonl")))


def _retrain_path() -> Path:
    return Path(os.path.expanduser(os.getenv("DEPLOY_RETRAIN_PATH", "~/deploy-retrain.jsonl")))


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return out


def _load_secrets() -> dict:
    secrets: dict[str, str] = {}
    p = Path(os.path.expanduser("~/.deploy-secrets"))
    if not p.exists():
        return secrets
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            secrets[k.strip()] = v.strip()
    return secrets


def _airflow_run_state(run_id: str) -> str | None:
    """Return the Airflow state of the run, or None if it can't be fetched."""
    s = _load_secrets()
    base = s.get("AIRFLOW_URL", "http://localhost:33013").rstrip("/")
    user, pw = s.get("AIRFLOW_USER"), s.get("AIRFLOW_PASS")
    if not user or not pw:
        return None
    sess = requests.Session()
    sess.headers["User-Agent"] = "delivery-deploy-hook"
    login = sess.get(base + "/login/", timeout=20)
    tok = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', login.text)
    data = {"username": user, "password": pw}
    if tok:
        data["csrf_token"] = tok.group(1)
    sess.post(base + "/login/", data=data, timeout=20)
    r = sess.get(f"{base}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}", timeout=20)
    if r.status_code != 200:
        return None
    return r.json().get("state")


def _age_seconds(iso: str) -> float:
    try:
        started = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return (datetime.now(timezone.utc) - started).total_seconds()


def reconcile() -> None:
    runs = _read_jsonl(_runs_path())
    if not runs:
        return
    latest = runs[-1]
    run_id = latest.get("dag_run_id")
    if not run_id:
        return  # this deploy didn't queue a retrain — nothing to watch

    done = {r.get("dag_run_id") for r in _read_jsonl(_retrain_path())}
    if run_id in done:
        return  # already reconciled to a terminal outcome

    state = _airflow_run_state(run_id)
    timeout = float(os.getenv("RETRAIN_TIMEOUT_SECONDS", "1800"))
    outcome: str | None = None
    if state in TERMINAL:
        outcome = state
    elif _age_seconds(latest.get("finished_at", "")) > timeout:
        outcome = "timeout"
    # else: still running (or transiently unreachable) — try again next tick.

    if outcome is None:
        return

    record = {
        "dag_run_id": run_id,
        "dag_id": DAG_ID,
        "deploy_commit": latest.get("new_commit", ""),
        "state": outcome,
        "airflow_state": state,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with _retrain_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"watch_dag: retrain {run_id} -> {outcome}")


if __name__ == "__main__":
    try:
        reconcile()
    except Exception as exc:  # noqa: BLE001 — best-effort, never disturb a deploy
        print(f"watch_dag: error: {exc}", file=sys.stderr)
    sys.exit(0)
