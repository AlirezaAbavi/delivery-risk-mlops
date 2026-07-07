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

## Files

| File | Role |
|---|---|
| `deploy_hook.sh` | the hook (fetch → gate → conditional deploy) |
| `trigger_dag.py` | trigger `delivery_capstone_workflow` on the course Airflow |
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
