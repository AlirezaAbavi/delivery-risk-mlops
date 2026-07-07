# CI/CD — VM-side continuous deployment

GitLab-native CI/CD isn't available for this project (delivery is a `Developer`
with no rights to CI secret variables, and **no GitLab runner exists**). Instead,
deployment is driven **from the group VM**: a systemd user timer watches
`origin/main` and, on every new commit, does the right thing based on *what*
changed.

## Flow (`deploy_hook.sh`)

```
git fetch origin main
  └─ new commit? ── no ─▶ exit quietly
       │ yes
       ▼
   git pull (ff-only)
   pytest tests/            ── FAIL ─▶ log, do NOT deploy (never ship a red build)
       │ pass
       ▼
   act on changed paths:
     app/** | pipeline/** | requirements.txt   ─▶ restart the API (+ /health check)
     pipeline/** | app/config.py                ─▶ trigger the Airflow retrain DAG
     grafana/dashboards/**                      ─▶ re-import the Grafana dashboards
     (docs / notebooks only)                    ─▶ nothing heavy; checkout advances
```

Everything is appended to `~/deploy-hook.log`. Runs are serialized with an
`flock` so two ticks never overlap.

## Deploy monitoring (run records + status UI)

Because there is no GitLab CI UI to see hook runs, every **real deploy attempt**
(a new commit — not the quiet up-to-date ticks) also writes one structured JSON
line to `~/deploy-runs.jsonl` via `record_run.py`:

```json
{"run_id":"a1b2c3d-2026-07-07T18:10:00Z","started_at":"...","finished_at":"...",
 "duration_seconds":42,"branch":"main","old_commit":"...","new_commit":"a1b2c3d",
 "changed_paths":["app/main.py"],"status":"success",
 "actions":{"restart":"ok","trigger":"no","import":"no"}}
```

`status` is one of `success | tests_failed | ff_failed | error`, so failed
deploys are recorded too. Writing the record is best-effort and runs *after* the
deploy actions — it can never change the hook's outcome or exit code.

**Airflow retrain outcome (async).** The retrain is triggered fire-and-forget, so
the record marks `trigger: queued` and captures the `dag_run_id` — it does *not*
block for the multi-minute run. Each timer tick then runs `record_run`'s sibling
`watch_dag.py`, which polls Airflow once for that run and, when it reaches a
terminal state (or ages out), appends the outcome to `~/deploy-retrain.jsonl`.
This is driven by the existing timer, not a detached process (a child of the
oneshot deploy service would be killed with its cgroup); it is idempotent and
survives restarts. The API joins the two files so the flowchart shows the retrain
as **running → success/failed**, distinct from "was it triggered".

The FastAPI service reads this file **on demand** (read-only) and surfaces it:

- `GET /deploy-status` — JSON `{latest, recent (last 20), total_recorded}`;
  `?format=html` renders a **pipeline flowchart** (fetch → fast-forward → test
  gate → restart / trigger / import, each node green/red/amber/grey by outcome)
  above a recent-runs table. Missing file → `status: "unknown"`.
- Prometheus gauges `delivery_deploy_last_status`, `delivery_deploy_last_timestamp_seconds`,
  `delivery_deploy_last_duration_seconds`, `delivery_deploy_runs_total`, and
  `delivery_deploy_last_commit_info{commit,status}` — refreshed at scrape time and
  shown in the **"Deployment (CD hook)"** row of the Grafana dashboard.

No new service, port, or database: it rides the existing `:8112` Prometheus
scrape and Grafana. The path is overridable with `DEPLOY_RUNS_PATH` (must match
between the hook and the API's environment).

## Files

| File | Role |
|---|---|
| `deploy_hook.sh` | the hook (reconcile retrain → fetch → gate → conditional deploy → record run) |
| `record_run.py` | append one JSONL run record to `~/deploy-runs.jsonl` (best-effort) |
| `watch_dag.py` | reconcile the triggered retrain's outcome into `~/deploy-retrain.jsonl` (per-tick, non-blocking) |
| `trigger_dag.py` | trigger `delivery_capstone_workflow` on the course Airflow (writes back the run id) |
| `import_grafana.py` | POST `grafana/dashboards/*.json` to Grafana (overwrite) |
| `systemd/delivery-deploy.service` | oneshot that runs the hook |
| `systemd/delivery-deploy.timer` | fires the service every ~2 min |

## One-time install on the VM

Secrets live **only on the VM**, never in the repo.

```bash
# 1. non-interactive git auth for ~/project (main is not protected)
git config --global credential.helper store
printf 'http://delivery:<GITLAB_TOKEN>@localhost:8181\n' > ~/.git-credentials
chmod 600 ~/.git-credentials

# 2. deploy secrets (Airflow + Grafana) read by the two Python helpers
cat > ~/.deploy-secrets <<'EOF'
AIRFLOW_URL=http://localhost:33013
AIRFLOW_USER=delivery
AIRFLOW_PASS=<airflow-pass>
GRAFANA_URL=http://localhost:3010
GRAFANA_USER=delivery
GRAFANA_PASS=<grafana-pass>
GRAFANA_FOLDER=delivery
EOF
chmod 600 ~/.deploy-secrets

# 3. install + enable the timer
chmod +x ~/project/ci/deploy_hook.sh
cp ~/project/ci/systemd/delivery-deploy.* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now delivery-deploy.timer
loginctl enable-linger delivery   # keep the timer running when logged out
```

Check it:

```bash
systemctl --user list-timers delivery-deploy.timer
tail -f ~/deploy-hook.log
```

## Notes

- Poll latency is the timer interval (~2 min); tune `OnUnitActiveSec` in the timer.
- The Airflow retrain is heavy, so it fires **only** for `pipeline/**` or
  `app/config.py` changes — not on every push.
- `~/project` must stay on `main` (it is, post-consolidation) for the ff-only pull.
