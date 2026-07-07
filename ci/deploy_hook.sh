#!/usr/bin/env bash
# VM-side continuous deployment hook (delivery delivery-risk).
#
# Polled by delivery-deploy.timer. On a new commit to origin/main it: pulls,
# gates on the test suite, then acts only on what changed:
#   - app/** | pipeline/** | requirements.txt   -> restart the API
#   - pipeline/** | app/config.py               -> trigger the Airflow DAG (retrain)
#   - grafana/dashboards/**                      -> re-import the Grafana dashboards
# Docs/notebook-only commits just fast-forward the checkout.
#
# No secrets live here: git auth comes from ~/.git-credentials, and the trigger/
# import scripts read ~/.deploy-secrets. All output is appended to the log.
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/project}"
BRANCH="${DEPLOY_BRANCH:-main}"
PY="$PROJECT_DIR/.venv/bin/python"
LOG="${DEPLOY_LOG:-$HOME/deploy-hook.log}"
API_UNIT="delivery-capstone-api.service"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >>"$LOG"; }

# Single-instance guard: never overlap two hook runs.
exec 9>"$HOME/.deploy-hook.lock"
flock -n 9 || { log "another run holds the lock; skipping"; exit 0; }

cd "$PROJECT_DIR" || { log "ERROR: project dir $PROJECT_DIR missing"; exit 1; }

# 1. fetch; bail out fast when nothing changed.
if ! git fetch --quiet origin "$BRANCH"; then
    log "ERROR: git fetch failed"; exit 1
fi
OLD="$(git rev-parse HEAD)"
NEW="$(git rev-parse "origin/$BRANCH")"
if [ "$OLD" = "$NEW" ]; then
    exit 0   # up to date; stay quiet so the log isn't spammed every 2 min
fi
log "new commit $NEW (was $OLD) on $BRANCH; deploying"

# 2. changed paths (old..new).
CHANGED="$(git diff --name-only "$OLD" "$NEW")"

# 3. fast-forward only (main is not rewritten under us).
if ! git merge --ff-only "origin/$BRANCH" >>"$LOG" 2>&1; then
    log "ERROR: ff-only merge failed (local divergence?); aborting"; exit 1
fi

# 4. test gate — never ship a red build.
if ! "$PY" -m pytest -q "$PROJECT_DIR/tests" >>"$LOG" 2>&1; then
    log "TESTS FAILED at $NEW; not restarting/triggering/importing"; exit 1
fi
log "tests passed"

# helper: does any changed path match a regex?
changed_matches() { echo "$CHANGED" | grep -Eq "$1"; }

restarted="no"; triggered="no"; imported="no"

# 5. restart the API when files the running service loads changed.
if changed_matches '^app/|^pipeline/|^requirements\.txt$'; then
    if systemctl --user restart "$API_UNIT"; then
        sleep 4
        if curl -fsS -m 8 http://127.0.0.1:8112/health >>"$LOG" 2>&1; then
            echo >>"$LOG"; restarted="ok"
        else
            restarted="restarted-but-health-unconfirmed"
        fi
    else
        restarted="restart-FAILED"
    fi
    log "API restart: $restarted"
fi

# 6. trigger the Airflow retrain ONLY when ML-pipeline logic changed.
if changed_matches '^pipeline/|^app/config\.py$'; then
    if "$PY" "$PROJECT_DIR/ci/trigger_dag.py" >>"$LOG" 2>&1; then
        triggered="ok"
    else
        triggered="trigger-FAILED"
    fi
    log "Airflow trigger: $triggered"
fi

# 7. re-import Grafana dashboards when they changed.
if changed_matches '^grafana/dashboards/'; then
    if "$PY" "$PROJECT_DIR/ci/import_grafana.py" >>"$LOG" 2>&1; then
        imported="ok"
    else
        imported="import-FAILED"
    fi
    log "Grafana import: $imported"
fi

if [ "$restarted" = "no" ] && [ "$triggered" = "no" ] && [ "$imported" = "no" ]; then
    log "no relevant change (docs/notebooks); checkout advanced only"
fi
log "deploy done for $NEW (restart=$restarted trigger=$triggered import=$imported)"
