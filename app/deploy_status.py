"""Read the CD-hook run records so the API can report deployment status.

Design idea (event-sourcing-lite): the CD hook (``ci/deploy_hook.sh``) is the only
*writer* — it appends one JSON object per deploy attempt to ``DEPLOY_RUNS_PATH``
(default ``~/deploy-runs.jsonl``, one JSON per line = "JSONL"). This module is a pure
*reader*: it parses that file **on demand**, holds no in-memory state, and never
writes. Two nice properties fall out of that split:

  - It always reflects the latest completed deploy, and it survives the very API
    restart that a deploy triggers (the file outlives the process).
  - A missing/corrupt file degrades to an ``unknown`` status instead of raising, so
    the deploy-monitoring feature can never take down prediction serving.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from . import config

log = logging.getLogger("api.deploy_status")

# The set of ``status`` strings the hook writes when a deploy did NOT fully succeed.
# Anything not in here (in practice "success") is treated as a good deploy.
_FAILURE_STATUSES = {"tests_failed", "ff_failed", "fetch_failed", "error"}


def _read_records(limit: int) -> list[dict]:
    """Return up to ``limit`` most-recent run records, newest last.

    The file is tiny (one line per deploy, and deploys are infrequent), so reading it
    whole and slicing the tail is simpler and fast enough. Every line is parsed
    defensively: a line that isn't valid JSON is skipped, which makes a half-written
    final line (a deploy caught mid-append) harmless. Distinct except branches:
    missing file -> empty (normal on a fresh box); unreadable -> log + empty.
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
                    continue  # skip a corrupt/partial line rather than fail the read
    except FileNotFoundError:
        return []
    except OSError as exc:  # unreadable file — report unknown, never crash
        log.warning("deploy-runs unreadable: %s", exc)
        return []
    return records[-limit:] if limit > 0 else records


def _read_retrain() -> dict:
    """Build a {dag_run_id -> retrain outcome record} map from the retrain log.

    Retrains are asynchronous: a deploy *queues* an Airflow DAG run and moves on;
    ``ci/watch_dag.py`` later appends the terminal outcome keyed by ``dag_run_id``.
    Keying by run id lets us reconcile a deploy record with its (possibly much later)
    retrain result. "Last write wins" so a corrected outcome supersedes an earlier one.
    """
    out: dict[str, dict] = {}
    try:
        with open(config.DEPLOY_RETRAIN_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = rec.get("dag_run_id")
                if rid:
                    out[rid] = rec  # last write wins
    except (FileNotFoundError, OSError):
        return {}
    return out


def retrain_state_for(record: Optional[dict]) -> Optional[str]:
    """Resolve a deploy's retrain state: None | running | success | failed | timeout.

    Logic: a deploy that never queued a DAG run has no retrain (None). One that queued
    a run but has no terminal record yet is "running" — the async watcher fills the
    outcome in later. Otherwise we return the recorded terminal state. This is what
    lets the UI honestly show "running" instead of prematurely claiming success.
    """
    if not record:
        return None
    rid = record.get("dag_run_id")
    if not rid:
        return None
    rec = _read_retrain().get(rid)
    return rec.get("state") if rec else "running"


def is_success(record: Optional[dict]) -> bool:
    """True only when a record exists and its status is exactly 'success'."""
    return bool(record) and record.get("status") == "success"


def snapshot(history_limit: Optional[int] = None) -> dict:
    """Full view for /deploy-status: latest run + recent history + counts.

    This is the single object the endpoint and the metrics refresher both consume.
    When there are no records at all we return a well-formed ``unknown`` snapshot (not
    an error) so every caller can rely on the same shape. The latest record is
    enriched with its reconciled ``retrain`` block; ``recent`` is reversed to
    newest-first for display.
    """
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
    latest = dict(recent[-1])  # copy so we don't mutate the cached-read list element
    latest["retrain"] = {"dag_run_id": latest.get("dag_run_id"), "state": retrain_state_for(latest)}
    return {
        "status": latest.get("status", "unknown"),
        "latest": latest,
        "recent": list(reversed(recent)),  # newest first for display
        "total_recorded": len(recent),
    }


def latest() -> Optional[dict]:
    """The single most-recent run record, or None if there are none.

    Cheaper than ``snapshot`` when a caller only needs the last deploy (asks the reader
    for just the tail record).
    """
    recent = _read_records(1)
    return recent[-1] if recent else None


def latest_retrain() -> Optional[dict]:
    """Most recent deploy that queued a retrain, with its reconciled state.

    Note the deliberate difference from ``latest()``: we scan *backwards* for the last
    deploy that actually had a ``dag_run_id``. The Grafana retrain gauges use this so
    the panel keeps showing the last real retrain outcome even across later deploys
    that didn't retrain (e.g. a docs-only push shouldn't blank the retrain panel). The
    per-deploy flowchart, by contrast, stays scoped to the single latest deploy.
    """
    for rec in reversed(_read_records(config.DEPLOY_HISTORY_LIMIT)):
        if rec.get("dag_run_id"):
            return {
                "dag_run_id": rec["dag_run_id"],
                "state": retrain_state_for(rec),
                "deploy_commit": rec.get("new_commit", ""),
            }
    return None


# --- flowchart model -------------------------------------------------------
# The HTML view draws the deploy as a small DAG. The hook always runs the same fixed
# pipeline: fetch a new commit -> fast-forward the checkout -> run the test gate ->
# then, depending on which paths changed, restart the API / trigger Airflow / import
# Grafana. The functions below translate one run *record* into per-node states so the
# renderer (deploy_view.py) can colour each box by what actually happened.
#
# Every node ends up in one of: ok | failed | warn | skipped | pending (| queued | running)

# The three sequential "gate" steps and the JSON keys that describe them.
_GATE = [("New commit", "fetch"), ("Fast-forward", "ff"), ("Test gate", "tests")]
# The three conditional "action" steps that only run after a green gate.
_ACTIONS = [("Restart API", "restart"), ("Trigger Airflow", "trigger"), ("Import Grafana", "import")]


def _gate_state(key: str, status: str) -> str:
    """Colour a gate node given the overall run status.

    Reasoning encoded here:
      - fetch: a record exists at all only because we fetched and saw a new commit,
        so this node is always "ok" once there's a record.
      - ff: red only when the fast-forward itself failed.
      - tests: if ff already failed we never reached the gate (skipped); otherwise red
        on a test failure, green on success.
    """
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
    """Colour a conditional action node from its recorded value.

    Actions are only attempted after a green test gate; before that they're "skipped".
    The recorded value is a short string the hook wrote ("ok"/"no"/"queued"/a failure
    word), which we map to a node state. "queued" is special: a fire-and-forget
    Airflow trigger whose real outcome shows up later on the separate retrain node.
    """
    if not tests_passed:
        return "skipped"  # actions only run after a green test gate
    v = (value or "no").lower()
    if v == "no":
        return "skipped"  # this action wasn't selected by the changed paths
    if "fail" in v:
        return "failed"
    if v == "ok":
        return "ok"
    if v == "queued":
        return "queued"  # fire-and-forget trigger accepted; outcome via retrain node
    return "warn"  # e.g. restarted-but-health-unconfirmed


# Map an async retrain outcome onto a flowchart node state (colour).
_RETRAIN_NODE = {"success": "ok", "failed": "failed", "timeout": "warn", "running": "running"}


def deploy_steps(record: Optional[dict]) -> dict:
    """Turn one run record into the per-node state dict the SVG renderer consumes.

    With no record everything is "pending" (nothing has run). Otherwise we compute
    each gate node, each action node (gated on ``tests_passed``), and the async retrain
    node. The renderer never has to know the deploy semantics — all the judgement lives
    here, cleanly separated from the drawing code.
    """
    if not record:
        return {
            "gate": [{"name": n, "state": "pending", "detail": ""} for n, _ in _GATE],
            "actions": [{"name": n, "state": "pending", "detail": ""} for n, _ in _ACTIONS],
        }
    status = record.get("status", "")
    tests_passed = status not in ("ff_failed", "tests_failed")
    actions = record.get("actions", {})
    rt = retrain_state_for(record)
    return {
        "gate": [
            {"name": n, "state": _gate_state(k, status), "detail": ""} for n, k in _GATE
        ],
        "actions": [
            {
                "name": n,
                "state": _action_state(actions.get(k), tests_passed),
                "detail": str(actions.get(k, "no")),  # full value shown in the table
            }
            for n, k in _ACTIONS
        ],
        "retrain": {
            "name": "Retrain",
            "state": _RETRAIN_NODE.get(rt, "skipped") if rt else "skipped",
            "detail": rt or "not run",
        },
    }
