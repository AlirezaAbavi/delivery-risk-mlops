# Delivery — Delivery-Risk Operations (MLOps project)

An end-to-end MLOps platform that predicts **late-delivery risk** for Olist e-commerce
orders **at purchase time**, early enough for an operations team to intervene. It covers
the full lifecycle: raw data → leak-free feature engineering → model training & registry →
a FastAPI serving contract → batch scoring → orchestration → drift monitoring → dashboards →
continuous deployment.

> **📖 Full write-up:** [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) — the
> complete documentation (architecture, data, feature engineering, model selection, every API
> endpoint, Airflow, MLflow, Prometheus/Grafana, CI/CD, security, and the demo checklist).

## Problem & framing

- **Target:** `is_late_delivery` — the order was delivered after its estimated delivery date.
- **Temporal-leakage firewall (graded):** predictions use **only purchase-time inputs**.
  Delivery-outcome, review, and actual-delivery fields are never features. The API enforces
  this at the schema level (`extra="forbid"` on `PredictionInput`) plus an explicit denylist,
  and the feature table is built purchase-time only.
- **Class imbalance:** ~8% positive. Models are compared by **PR-AUC / average precision**
  (the right metric here), with `class_weight` / `sample_weight` for balancing.
- **Temporal validation:** train on the earliest 80% of purchases, validate on the newest
  20%, so metrics respect the real late-rate drift and never see the future.
- **Most impactful feature:** `shipping_window_days` (carrier slack between the seller's
  shipping deadline and the promised delivery date) dominates — see the documentation.

## Architecture

```
 Olist CSVs ──► Postgres (raw)
                    │  pipeline/load_raw.py
                    ▼
             featureset_v1  ──────────────► pipeline/features.py  (leak-free SQL)
                    │
                    ▼
          pipeline/train.py ──► MLflow  (3 models compared, winner → Staging)
             │        │            registry: delivery-risk
             │        ▼
             │   artifacts/model.joblib   (local fallback for the API)
             ▼
      pipeline/batch_predict.py ──► predictions  (Postgres)
             │
             ▼
      pipeline/monitor.py ──► monitoring_metrics (Postgres)  [PSI drift, ROC-AUC, late-rate]
                                   │
   FastAPI (app/) ── /predict ─────┼─────────────► Prometheus → Grafana dashboards
     MLflow Staging model          │
     /metrics (Prometheus)         ▼
                              Airflow DAG orchestrates the whole cycle
```

## Repository layout

| Path | What |
|------|------|
| `app/` | **The FastAPI service.** `main.py` (routes), `schemas.py` (contract + leakage firewall), `model_loader.py` (MLflow→joblib→baseline resolution), `predictor.py`, `metrics.py` (Prometheus), `middleware.py`, `config.py`, `logging_config.py`, `deploy_status.py`/`deploy_view.py` (CD-hook status) |
| `pipeline/` | `db.py`, `load_raw.py`, `features.py`, `train.py`, `register.py`, `batch_predict.py`, `monitor.py`, `smoke_test.py` |
| `airflow/dags/capstone_pipeline.py` | The graded orchestration DAG (course Airflow, SSH-bridge to the VM) |
| `airflow/dags/delivery_delivery_risk_pipeline.py` | Local-Airflow TaskFlow variant |
| `airflow/docker-compose.override.yaml` | Bridges a Dockerized Airflow to this host project |
| `ci/` | VM-side continuous-deployment hook (systemd timer) + its README |
| `grafana/dashboards/` | Prometheus dashboard JSON (instance-pinned) |
| `tests/test_api.py` | Endpoint set, `/predict` contract, leakage, observability, deploy-status |
| `docs/` | Full project documentation + images |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | API image + deps |

## The API contract

Endpoints (all required): `GET /health`, `GET /model-info`, `POST /predict`,
`POST /batch-predict`, `GET /metrics-summary`, `GET /metrics` (plus `GET /deploy-status`).

Each prediction returns exactly:
`order_id`, `late_delivery_probability`, `risk_level`, `recommended_action`,
`model_version`, `latency`.

The service resolves its model in order: **MLflow `Staging`** → **local `artifacts/model.joblib`**
→ **baseline scorer** (so it never hard-fails startup; `/health` reports which is active).

## Local setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # PyPI locally
# On the course server instead: pip install --no-index \
#   --find-links /opt/MLOps/MlOps/project/capstone_stack/wheelhouse -r requirements.txt
.venv/bin/python -m pytest -q tests                      # API tests
```

Configuration is read from `.env` (see `.env.example` for the full variable set — DB URL,
MLflow URI, model name/stage, risk thresholds, temporal-split settings, drift thresholds).

### Run the pieces

```bash
# MLflow tracking server (local)
.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5312

# Data → features → train/register → smoke → score → monitor
.venv/bin/python -m pipeline.load_raw
.venv/bin/python -m pipeline.features
.venv/bin/python -m pipeline.train
.venv/bin/python -m pipeline.register
.venv/bin/python -m pipeline.batch_predict
.venv/bin/python -m pipeline.monitor

# The API
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8112
```

### Docker

```bash
docker build -t delivery-api .
docker run --rm -p 8112:8112 --env-file .env delivery-api
# serve a local joblib model: -v /path/pipeline.pkl:/models/pipeline.pkl -e MODEL_PATH=/models/pipeline.pkl
```

The image bakes in no model, data, or secrets and runs as an unprivileged user.

## Orchestration (Airflow)

`airflow/dags/capstone_pipeline.py` (`dag_id=delivery_capstone_workflow`) runs the full cycle on
the course Airflow: `load_raw_data → build_features → train_model → register_model →
api_smoke_test → batch_predict → monitor → decide_retrain → [flag_retrain | no_retrain]`.

Because the Airflow worker can't reach the VM's services, every step runs **on the group VM** via
the course SSH bridge. Deploying a DAG change = **push to `main`** (the course Airflow git-syncs
`airflow/dags/`). `decide_retrain` is **record-and-alert, not auto-retrain**.

## Observability

- **Service metrics → Prometheus → Grafana.** The API exposes `GET /metrics` (request/error
  counters, latency histograms, prediction counts by risk level, model-loaded gauge, deploy
  gauges). The course Prometheus already scrapes `:8112`; the committed dashboard
  (`grafana/dashboards/delivery_risk_prometheus.json`) is imported into the shared Grafana.
- **Model/data monitoring → Postgres.** `pipeline/monitor.py` computes PSI on the score + key
  features and recent-window ROC-AUC, writes `monitoring_metrics`, and decides retrain
  (`score/feature PSI > 0.2` or `AUC < 0.65`).
- **Structured JSON logs** with a per-request `X-Request-ID` (Loki/Grafana-ready).

## CI/CD

GitLab-native CI/CD is unavailable (Developer role, no runner), so deployment is a **VM-side hook**
(`ci/`): a systemd timer watches `origin/main`, runs the pytest gate, then change-gated
restart-API / trigger-Airflow / re-import-Grafana. See [`ci/README.md`](ci/README.md).

## Version-control hygiene

Never commit credentials, `.venv`, data (`olist_data/`, `*.csv`), generated artifacts
(`artifacts/`, `mlflow/`, `models/`), or course PDFs — all covered by `.gitignore`. Config lives in
`.env` (gitignored); `.env.example` documents the variables without secrets.

## Infrastructure

- **GitLab (origin):** `http://localhost:8181/delivery-mlops-capstone/delivery-project.git`
- **MLflow:** `http://127.0.0.1:5312` · **API:** `:8112` · **Airflow:** `:33013` (course) · **Grafana:** `http://localhost:3010`
