#!/usr/bin/env python3
"""Append one structured JSON record for a deploy-hook run to ~/deploy-runs.jsonl.

Called by deploy_hook.sh at the end of every *real* deploy attempt (a new commit
on origin/main), including failures. Doing the JSON encoding here — instead of in
bash — keeps changed-path lists and error strings safely escaped, and reuses the
venv python the hook already depends on.

All inputs arrive as environment variables so the hook never has to quote them:

    DR_STARTED_AT, DR_FINISHED_AT   ISO-8601 UTC timestamps
    DR_DURATION_SECONDS             integer seconds
    DR_BRANCH                       deploy branch (e.g. main)
    DR_OLD_COMMIT, DR_NEW_COMMIT    full SHAs
    DR_CHANGED_PATHS                newline-separated changed paths
    DR_STATUS                       success | tests_failed | ff_failed | fetch_failed | error
    DR_RESTART, DR_TRIGGER, DR_IMPORT   per-action result strings
    DEPLOY_RUNS_PATH                output file (default ~/deploy-runs.jsonl)

This script is best-effort: it must never fail the deploy, so it swallows its own
errors and exits 0 regardless. The record is a side-effect, not a gate.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _short(sha: str) -> str:
    return sha[:7] if sha else ""


def main() -> None:
    new_commit = os.getenv("DR_NEW_COMMIT", "")
    finished_at = os.getenv("DR_FINISHED_AT", "")

    changed = [p for p in os.getenv("DR_CHANGED_PATHS", "").splitlines() if p.strip()]

    try:
        duration = int(os.getenv("DR_DURATION_SECONDS", "0") or "0")
    except ValueError:
        duration = 0

    record = {
        "run_id": f"{_short(new_commit)}-{finished_at}",
        "started_at": os.getenv("DR_STARTED_AT", ""),
        "finished_at": finished_at,
        "duration_seconds": duration,
        "branch": os.getenv("DR_BRANCH", ""),
        "old_commit": os.getenv("DR_OLD_COMMIT", ""),
        "new_commit": new_commit,
        "changed_paths": changed,
        "status": os.getenv("DR_STATUS", "error"),
        "actions": {
            "restart": os.getenv("DR_RESTART", "no"),
            "trigger": os.getenv("DR_TRIGGER", "no"),
            "import": os.getenv("DR_IMPORT", "no"),
        },
        "host": os.uname().nodename,
    }

    out = Path(os.path.expanduser(os.getenv("DEPLOY_RUNS_PATH", "~/deploy-runs.jsonl")))
    try:
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # never let a logging failure abort a deploy
        print(f"record_run: could not write {out}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        print(f"record_run: unexpected error: {exc}", file=sys.stderr)
    sys.exit(0)
