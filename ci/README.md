# CI/CD

## CI — GitHub Actions

Continuous integration runs in GitHub Actions (`.github/workflows/ci.yml`): on every
push and pull request it installs the dependencies, runs the test suite
(`pytest -q tests`), builds the API Docker image, and validates
`docker compose config`. A red build blocks the change.

## Deploy monitoring (`record_run.py`)

The API surfaces a small deploy-history feature at `/deploy-status` and via the
`delivery_deploy_*` Prometheus metrics — no separate database. To feed it, call
`record_run.py` at the end of a deploy (from a CD job or a local deploy script). It
appends one structured JSON line per deploy attempt to `~/deploy-runs.jsonl`:

```json
{"run_id":"a1b2c3d-2026-07-07T18:10:00Z","started_at":"...","finished_at":"...",
 "duration_seconds":42,"branch":"main","old_commit":"...","new_commit":"a1b2c3d",
 "changed_paths":["app/main.py"],"status":"success",
 "actions":{"restart":"ok","trigger":"no","import":"no"}}
```

All inputs arrive as `DR_*` environment variables (see the module docstring), so the
caller never has to quote them. `status` records failures too
(`success | tests_failed | ff_failed | error`). Writing the record is best-effort: it
never changes the deploy's outcome.

The FastAPI service reads the file **on demand** (read-only):

- `GET /deploy-status` — JSON `{latest, recent, total_recorded}`; `?format=html` renders
  a pipeline flowchart above a recent-runs table. Missing file → `status: "unknown"`.
- Prometheus gauges `delivery_deploy_last_status`, `delivery_deploy_last_timestamp_seconds`,
  `delivery_deploy_last_duration_seconds`, `delivery_deploy_runs_total`, and
  `delivery_deploy_last_commit_info{commit,status}` — refreshed at scrape time and shown
  in the deployment row of the Grafana dashboard.

The path is overridable with `DEPLOY_RUNS_PATH` (must match between the recorder and the
API's environment).

## Files

| File | Role |
|---|---|
| `record_run.py` | append one JSONL run record to `~/deploy-runs.jsonl` (best-effort) |
