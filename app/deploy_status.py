"""Read the CD-hook run records so the API can report deployment status.

``ci/deploy_hook.sh`` appends one JSON object per deploy attempt to
``DEPLOY_RUNS_PATH`` (default ``~/deploy-runs.jsonl``). This module reads that
file **on demand** — no background thread, no in-memory state, never writes —
so it always reflects the latest completed deploy and survives the very API
restart that a deploy triggers. A missing or malformed file degrades to an
``unknown`` status rather than raising, so deploy monitoring can never affect
prediction serving.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from . import config

log = logging.getLogger("api.deploy_status")

# statuses the hook writes when a deploy did not fully succeed
_FAILURE_STATUSES = {"tests_failed", "ff_failed", "fetch_failed", "error"}


def _read_records(limit: int) -> list[dict]:
    """Return up to ``limit`` most-recent run records, newest last.

    Reads the whole (small, deploy-frequency) file and keeps the tail. Any line
    that isn't valid JSON is skipped, so a partially written last line is safe.
    """
    path = config.DEPLOY_RUNS_PATH
    records: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    except OSError as exc:  # unreadable file — report unknown, never crash
        log.warning("deploy-runs unreadable: %s", exc)
        return []
    return records[-limit:] if limit > 0 else records


def is_success(record: Optional[dict]) -> bool:
    return bool(record) and record.get("status") == "success"


def snapshot(history_limit: Optional[int] = None) -> dict:
    """Full view for /deploy-status: latest run + recent history + counts."""
    limit = config.DEPLOY_HISTORY_LIMIT if history_limit is None else history_limit
    recent = _read_records(limit)
    if not recent:
        return {
            "status": "unknown",
            "message": "no deploys recorded yet",
            "latest": None,
            "recent": [],
            "total_recorded": 0,
        }
    latest = recent[-1]
    return {
        "status": latest.get("status", "unknown"),
        "latest": latest,
        "recent": list(reversed(recent)),  # newest first for display
        "total_recorded": len(recent),
    }


def latest() -> Optional[dict]:
    """The single most-recent run record, or None if there are none."""
    recent = _read_records(1)
    return recent[-1] if recent else None


# --- flowchart model -------------------------------------------------------
# The deploy hook runs a fixed pipeline: fetch a new commit -> fast-forward ->
# test gate -> then, per changed paths, restart the API / trigger Airflow /
# import Grafana. deploy_steps() maps a run record onto that pipeline so the UI
# can draw it as a DAG with each node coloured by outcome.
#
# state is one of: ok | failed | warn | skipped | pending

_GATE = [("New commit", "fetch"), ("Fast-forward", "ff"), ("Test gate", "tests")]
_ACTIONS = [("Restart API", "restart"), ("Trigger Airflow", "trigger"), ("Import Grafana", "import")]


def _gate_state(key: str, status: str) -> str:
    if key == "fetch":
        return "ok"  # a record exists only because we fetched and saw a new commit
    if key == "ff":
        return "failed" if status == "ff_failed" else "ok"
    if key == "tests":
        if status == "ff_failed":
            return "skipped"  # never reached the gate
        return "failed" if status == "tests_failed" else "ok"
    return "pending"


def _action_state(value: Optional[str], tests_passed: bool) -> str:
    if not tests_passed:
        return "skipped"  # actions only run after a green test gate
    v = (value or "no").lower()
    if v == "no":
        return "skipped"  # this action wasn't selected by the changed paths
    if "fail" in v:
        return "failed"
    if v == "ok":
        return "ok"
    return "warn"  # e.g. restarted-but-health-unconfirmed


def deploy_steps(record: Optional[dict]) -> dict:
    """Per-step pipeline states for the flowchart view, from a run record."""
    if not record:
        return {
            "gate": [{"name": n, "state": "pending", "detail": ""} for n, _ in _GATE],
            "actions": [{"name": n, "state": "pending", "detail": ""} for n, _ in _ACTIONS],
        }
    status = record.get("status", "")
    tests_passed = status not in ("ff_failed", "tests_failed")
    actions = record.get("actions", {})
    return {
        "gate": [
            {"name": n, "state": _gate_state(k, status), "detail": ""} for n, k in _GATE
        ],
        "actions": [
            {
                "name": n,
                "state": _action_state(actions.get(k), tests_passed),
                "detail": str(actions.get(k, "no")),
            }
            for n, k in _ACTIONS
        ],
    }
