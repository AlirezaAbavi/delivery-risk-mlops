# MLOps project — Delivery
# Delivery-Risk Operations: Full Project Documentation

**Team:** Delivery · **Track:** delivery-risk operations · **Cohort:** MLOps MLOps Bootcamp (Spring 1405 / 2026)
**Author of this deliverable:** Alireza Abavi · **Date:** 2026-07-08

> This document is the end-to-end write-up of the Delivery capstone. It walks through the
> problem, the data, feature engineering, model training and selection, the FastAPI serving
> layer, Airflow orchestration, MLflow tracking, Prometheus/Grafana observability, the CI/CD
> deployment hook, security/isolation, and repository hygiene. Screenshot placeholders
> (`![...]`) mark where to drop demo captures for the presentation.

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [System architecture](#2-system-architecture)
3. [Infrastructure & environments](#3-infrastructure--environments)
4. [Data: the Olist dataset & ingestion](#4-data-the-olist-dataset--ingestion)
5. [Feature engineering (and the single most impactful feature)](#5-feature-engineering)
6. [Temporal-leakage firewall](#6-temporal-leakage-firewall)
7. [Model training & selection](#7-model-training--selection)
8. [MLflow: experiments, runs & the registry](#8-mlflow-experiments-runs--the-registry)
9. [Batch prediction](#9-batch-prediction)
10. [Monitoring, drift detection & the retrain decision](#10-monitoring-drift-detection--the-retrain-decision)
11. [Airflow orchestration](#11-airflow-orchestration)
12. [The FastAPI service (endpoints & code)](#12-the-fastapi-service)
13. [Observability: Prometheus metrics](#13-observability-prometheus-metrics)
14. [Observability: Grafana dashboard](#14-observability-grafana-dashboard)
15. [Observability: structured logging](#15-observability-structured-logging)
16. [CI/CD: the VM-side deployment hook](#16-cicd-the-vm-side-deployment-hook)
17. [Docker & deployment](#17-docker--deployment)
18. [Security & isolation](#18-security--isolation)
19. [Repository organization & version-control hygiene](#19-repository-organization--version-control-hygiene)
20. [Known limitations & future work](#20-known-limitations--future-work)
21. [Appendix A — Demo & screenshot checklist](#appendix-a--demo--screenshot-checklist)
22. [Appendix B — Command reference](#appendix-b--command-reference)

---

## 1. Executive summary

**The problem.** Olist is a Brazilian e-commerce marketplace. A meaningful share of orders are
delivered *later than the estimated delivery date* promised to the customer. If an operations
team could know **at purchase time** which orders are at high risk of being late, they could
intervene early (expedite fulfilment, confirm carrier capacity, proactively notify the customer).

**What we built.** A complete MLOps platform that:

- **loads** the raw Olist CSVs into a Postgres **medallion** database (raw → features →
  predictions → monitoring);
- **engineers** a leak-free, purchase-time-only feature table (`featureset_v1`, 45 features);
- **trains and compares** three credible model families in **MLflow**, on a strict *temporal*
  validation split, and registers the winner (`delivery-risk`) to the **Staging** stage;
- **serves** predictions through a **FastAPI** service with the full contract
  (`/health`, `/model-info`, `/predict`, `/batch-predict`, `/metrics-summary`, `/metrics`);
- **orchestrates** the whole cycle (load → features → train → register → smoke-test → batch →
  monitor → retrain decision) through **Airflow**;
- **monitors** drift (PSI) and performance decay, and emits an explicit **retrain decision**;
- **observes** everything through **Prometheus** metrics and a **Grafana** dashboard, plus
  structured JSON logs;
- **deploys continuously** via a VM-side deployment hook (a pragmatic substitute for
  GitLab-native CI/CD, which the account role could not use).

**Headline result.** The winning model is a **HistGradientBoostingClassifier** with a validation
**ROC-AUC ≈ 0.72 / PR-AUC ≈ 0.15** on an 8.1%-positive problem. The single dominant predictor is
the **shipping window** (the gap between the seller's shipping deadline and the promised delivery
date) — see §5.

---

## 2. System architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │                     Airflow (course, SSH-bridge)               │
                 │  load_raw → build_features → train → register → smoke_test →   │
                 │  batch_predict → monitor → decide_retrain →[flag | no_retrain] │
                 └───────────────┬──────────────────────────────────────────────┘
                                 │ runs each step on the group VM
                                 ▼
 Olist CSVs ──► Postgres (medallion)                     MLflow (:5312)
   raw schema      raw.* ─┐                              experiments + registry
                          │  pipeline/features.py         delivery-risk
   features schema  ◄─────┘  (SQL, purchase-time)           v1 → Staging
   features.featureset_v1 ──► pipeline/train.py ──► logs 3 candidates, registers winner
                          │
   predictions schema ◄───┤  pipeline/batch_predict.py (scores all rows)
                          │
   monitoring schema  ◄───┘  pipeline/monitor.py (PSI drift + AUC → retrain flag)
                                 │
                                 ▼
                          FastAPI service (:8112)  ◄── loads Staging model at startup
                          /predict /health /metrics ...
                                 │  exposes Prometheus metrics
                                 ▼
                          Prometheus (:9091, course) ──► Grafana (:3010, delivery org)
```

> 📸 **Screenshot placeholder:** a clean architecture diagram (redraw the box diagram above in
> draw.io / Excalidraw) — `docs/images/architecture.png`.

---

## 3. Infrastructure & environments

The project runs in **two mirrored environments** so we can develop safely and demo reliably.

| Concern | Local dev (laptop) | Course server (`localhost`, user `delivery`) |
|---|---|---|
| Postgres | `127.0.0.1:5432`, DB `delivery`, single `public` schema | `…:32112`, DB `delivery_mlops_delivery`, **6-schema medallion** |
| MLflow | local server `127.0.0.1:5312` (sqlite + file artifacts) | local server `127.0.0.1:5312` (per-group) |
| Airflow | Dockerized `apache/airflow:3.2.2`, UI `127.0.0.1:8080` | course Airflow `…:33013` (SSH-bridges to the VM) |
| API | uvicorn `…:8112` | **systemd service** `delivery-capstone-api.service` on `…:8112` |
| Prometheus | — | course-managed `…:9091` (already scrapes our `:8112`) |
| Grafana | — | shared `…:3010`, delivery org (role: Editor) |
| GitLab | `origin` | `…:8181` (project id 2; role: Developer) |

**Key infrastructure facts (they shaped design decisions):**

- The course Postgres, MLflow, and API are all **loopback-only on the VM**; the course Airflow
  worker cannot reach them directly. Hence the DAG uses an **SSH bridge** to run each step *on the
  VM itself* (§11).
- The course **Prometheus is managed by the course** — we can't edit scrape targets; we only
  control what our API exposes. It already scrapes `host.docker.internal:8112`.
- Multiple groups emit **unprefixed `delivery_*` metric names** into the *same* Prometheus, so every
  Grafana query is **pinned to `instance="host.docker.internal:8112"`** to avoid blending other
  groups' series (§14).
- All configuration is **environment-driven** (`.env`, gitignored; `.env.example` committed) so the
  *same code and Docker image* run in both environments with zero code changes.

> 📸 **Screenshot placeholder:** `systemctl --user status delivery-capstone-api.service` showing
> `active (running)` and `Restart=always` — `docs/images/systemd-api.png`.

---

## 4. Data: the Olist dataset & ingestion

### 4.1 Source tables

The raw Olist dataset is a set of CSVs modelling the marketplace. We load the eight tables
relevant to late-delivery modelling (marketing / closed-deals CSVs are deliberately skipped):

| Table | Rows | Role in the model |
|---|---:|---|
| `orders` | 99,441 | order lifecycle timestamps + status (the **label** source) |
| `order_items` | 112,650 | per-item price, freight, product, seller, **shipping_limit_date** |
| `order_payments` | 103,886 | payment type, installments, value |
| `products` | 32,951 | category, weight, dimensions |
| `sellers` | 3,095 | seller city / state / zip (geography) |
| `customers` | 99,441 | customer geography (not yet used — see §20) |
| `order_reviews` | 99,224 | review score/text — **excluded** (post-delivery leakage) |
| `product_category_name_translation` | 73 | category-name lookup |

### 4.2 Ingestion (`pipeline/load_raw.py`)

- **Idempotent**: every table is fully replaced (`if_exists="replace"`) on each run, so the step is
  safely re-runnable from Airflow. A `--clear` flag drops all tables in the target schema first for
  a clean reload (removes stray tables from earlier manual loads).
- **Date parsing**: timestamp columns (`order_purchase_timestamp`, `order_approved_at`,
  `shipping_limit_date`, `order_estimated_delivery_date`, the two `order_delivered_*` dates, review
  dates) are parsed to real timestamps at load time, so downstream feature SQL does temporal
  arithmetic without re-parsing.
- **Encoding**: the category-translation CSV carries a UTF-8 BOM; it is read with `utf-8-sig`.
- **Schema-aware**: writes into the env-configured `RAW_SCHEMA` (server: `raw`; local: `public`).

### 4.3 The medallion schema (server)

The server database `delivery_mlops_delivery` uses a **6-schema medallion** layout, each layer owned by
the group role, each written by a distinct pipeline step:

| Schema | Written by | Contents |
|---|---|---|
| `raw` | `load_raw.py` | the 8 source tables, as loaded |
| `staging` | (reserved) | — |
| `features` | `features.py` | `features.featureset_v1` (the model input table) |
| `predictions` | `batch_predict.py` | `predictions.predictions` (scored orders) |
| `monitoring` | `monitor.py` | `monitoring.monitoring_metrics` (drift history) |
| `public` | (default) | fallback for single-schema local dev |

`pipeline/db.py` is the single source of truth for the SQLAlchemy engine and the per-layer schema
names. Each schema name is an env var defaulting to `public`, so the **same code** runs against the
single-schema local DB and the multi-schema server DB with no changes.

> 📸 **Screenshot placeholder:** `psql \dn` (schemas) and `\dt raw.*` (row counts) — 
> `docs/images/db-schemas.png`.

---

## 5. Feature engineering

**File:** `pipeline/features.py` (a single, well-commented SQL query materialised as
`features.featureset_v1`). One row per **delivered** order (96,456 rows). **Label balance:
8.11% late** (imbalanced — this drives our metric choice in §7).

### 5.1 Design principles

1. **One row per order** at the order grain. Two CTEs aggregate the finer grains up:
   `item_agg` (items + product + seller joins) and `pay_agg` (payments), then the outer query
   derives purchase-time windows and calendar features.
2. **Purchase-time only.** Every feature is knowable *at or around* `order_purchase_timestamp`.
   The estimated-delivery date and the per-item `shipping_limit_date` are **commitments set at
   purchase**, so windows derived from them are legal features. The actual delivered date is used
   **only** to compute the label, never as a feature (§6).
3. **Contract is authoritative.** The produced column set is asserted equal to
   `app.config.FEATURE_COLUMNS` — if the SQL ever drifts from the API/model contract, the build
   **fails loudly**. This keeps the training data, the model signature, and the API request schema
   in lock-step.

### 5.2 The 45 features (3 categorical, 42 numeric)

Grouped by domain:

- **Payment** (3): `payment_count`, `payment_type_mode` *(cat)*, `max_installments`
- **Order / items** (3): `order_item_count`, `product_count`, `seller_count`
- **Price / freight / cost** (11): `price_{sum,mean,max,std}`, `freight_{sum,mean,max,std}`,
  `total_cost_{sum,mean,max}`
- **Product category & dimensions** (13): `product_category_count`, `product_category_mode` *(cat)*,
  `is_multi_category`, `product_weight_g_{mean,max}`, `product_{length,height,width}_cm_{mean,max}`,
  `product_volume_{mean,max}`
- **Seller geography** (4): `seller_state_mode` *(cat)*, `seller_state_count`, `seller_city_count`,
  `seller_zip_mode`
- **Purchase calendar** (6): `purchase_hour`, `purchase_dayofweek`, `is_weekend_purchase`,
  `purchase_month`, `purchase_quarter`, `is_month_end`
- **Delivery / shipping windows** (5): `estimated_delivery_days`, `approval_delay_hours`,
  `shipping_limit_min_days`, **`shipping_window_days`**, `seller_margin_days`

**How the window features are derived (the interesting ones):**

- `estimated_delivery_days` = `order_estimated_delivery_date − order_purchase_timestamp` (the length
  of the delivery promise made to the customer).
- `shipping_limit_min_days` = earliest per-item `shipping_limit_date − purchase` (how soon the
  seller must hand the parcel to the carrier).
- **`shipping_window_days`** = `order_estimated_delivery_date − shipping_limit_min` (how much
  calendar slack the *carrier* has between the seller's deadline and the customer's promise).
- `seller_margin_days` = `estimated_delivery_date − latest shipping_limit` (slack against the
  *last* item's deadline).
- `approval_delay_hours` = `order_approved_at − purchase` (payment-approval latency).

Categorical modes (`payment_type_mode`, `product_category_mode`, `seller_state_mode`) and
per-order dispersion (`*_std`) come from Postgres `mode() WITHIN GROUP` and `stddev_samp`.

### 5.3 Which parameter has the greatest impact? — evidence

We measured **permutation importance** (mean drop in ROC-AUC when a single feature is shuffled) of
the trained HistGB model on the held-out newest-20% temporal window (n = 19,292). Result:

| Rank | Feature | ROC-AUC drop when shuffled |
|---:|---|---:|
| **1** | **`shipping_window_days`** | **0.185** |
| **2** | **`estimated_delivery_days`** | **0.106** |
| 3 | `shipping_limit_min_days` | 0.012 |
| 4 | `seller_zip_mode` | 0.009 |
| 5 | `seller_state_mode` | 0.006 |
| 6 | `approval_delay_hours` | 0.005 |
| 7 | `seller_count` | 0.004 |
| 8 | `freight_mean` | 0.003 |
| 9 | `freight_sum` | 0.002 |
| 10 | `price_sum` | 0.002 |

**Interpretation — the story to tell in the demo:**

- **`shipping_window_days` is by a wide margin the single most impactful feature** (0.185 vs the
  next feature at 0.106; everything below rank 2 is an order of magnitude smaller). It captures the
  *carrier's slack*: when the estimated-delivery date leaves little room after the seller's shipping
  deadline, the order is far more likely to arrive late. This is intuitive and operationally
  actionable — it tells the ops team **the promise itself is the biggest risk driver**.
- The two logistics/timing features (`shipping_window_days`, `estimated_delivery_days`) together
  dominate the model. Geography (`seller_zip_mode`, `seller_state_mode`) and payment-approval
  latency (`approval_delay_hours`) contribute a second tier; price/freight/product-dimension
  features add marginal signal.

![Top-10 feature importance](images/feature-importance.png)

*Figure — permutation importance (mean ROC-AUC drop) of the trained HistGB model on the newest-20%
temporal window. Reproducible from `artifacts/model.joblib` + `artifacts/featureset_v1.csv` with
`sklearn.inspection.permutation_importance` (command in Appendix B).*

---

## 6. Temporal-leakage firewall

Leakage is the *core* risk of this problem: any field that reveals the order's eventual outcome
(actual delivery date, review, delay) would make the model useless in production and is explicitly
**penalised in grading**. We enforce a **three-layer firewall**:

1. **At feature-build time (SQL, `features.py`):** `order_delivered_*` and `review_*` columns never
   enter the feature CTEs. The actual delivered date appears *only* in the label expression
   `(order_delivered_customer_date > order_estimated_delivery_date)::int AS is_late_delivery`.
   `purchase_ts` is emitted only as the temporal-split key, not a feature. The produced feature set
   is asserted `== FEATURE_COLUMNS`.
2. **At the API boundary (Pydantic, `schemas.py`):** `PredictionInput` uses
   `model_config = ConfigDict(extra="forbid")`, so *any* field not in the purchase-time contract is
   rejected with a 422 before it can reach the model.
3. **Explicit denylist (`predictor.py` + `config.FORBIDDEN_FIELDS`):** a second, well-labelled layer
   rejects nine named leakage fields (`is_late_delivery`, `order_delivered_customer_date`,
   `order_delivered_carrier_date`, `order_delivery_date`, `actual_delivery_days`,
   `delivery_delay_days`, `review_score`, `review_comment_message`, `review_creation_date`) with a
   clear 400 error naming the offending fields.

The test suite asserts the firewall holds (`test_leakage_firewall_rejects_outcome_fields`,
`test_logging_and_error_observability`).

---

## 7. Model training & selection

**File:** `pipeline/train.py`.

### 7.1 Validation strategy — a strict *temporal* split

We do **not** use a random train/test split. Late-delivery rate **drifts over time** in this
dataset (roughly 6.6% → 9.4% across the observation window), so a random split would leak future
information and overstate performance. Instead we train on the **earliest 80%** of purchases and
validate on the **newest 20%** (split at the 80th percentile of `purchase_ts`). This respects the
arrow of time and mirrors how the model is actually used: score *today's* orders having learned from
the past.

### 7.2 Three credible candidates

All three share the same preprocessing skeleton — median imputation for numerics, one-hot encoding
for the 3 categoricals (`handle_unknown="ignore"`, `min_frequency=20`) — and all handle the 8%
class imbalance:

| Model | Imbalance handling | Notes |
|---|---|---|
| `LogisticRegression` | `class_weight="balanced"` | linear baseline, scaled numerics |
| `RandomForestClassifier` | `class_weight="balanced_subsample"` | 300 trees, `min_samples_leaf=20` |
| **`HistGradientBoostingClassifier`** | per-sample `compute_sample_weight("balanced")` | 400 iters, lr 0.06, L2=1.0 — **winner** |

> **Implementation note (sklearn 1.8 compatibility):** the `ColumnTransformer` selects columns by
> **integer position** (`NUMERIC_IDX`/`CATEGORICAL_IDX`) rather than by name, to sidestep a
> regression in sklearn 1.8 where name-based selection reads `feature_names_in_` before it is set.
> This behaves identically on the host's sklearn 1.5. It matters because the pipeline is trained
> inside the Airflow container (sklearn 1.8) but may be loaded by the host API (sklearn 1.5).

### 7.3 Metrics used for model selection

The problem is **imbalanced (8.1% positive)**, so we do **not** select on accuracy (a model that
predicts "never late" would score ~92% accuracy and be useless). We log four metrics per candidate
and **select on PR-AUC**:

| Metric | Why we track it | Selection role |
|---|---|---|
| **PR-AUC** (average precision) | The right summary for a rare positive class — it focuses on how well the model ranks the minority (late) orders. | **Primary — the winner is chosen by PR-AUC.** |
| **ROC-AUC** | Overall ranking quality; comparable across groups/datasets. | Secondary / sanity. |
| **Brier score** | Probability *calibration* quality (are the probabilities meaningful?). | Diagnostic. |
| **F1 @ 0.5** | Point-classification quality at the default threshold. | Diagnostic. |

### 7.4 Results

On the temporal validation window (train ≈ 77k, valid ≈ 19k):

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| LogisticRegression | 0.066 | 0.586 |
| RandomForest | 0.089 | 0.659 |
| **HistGradientBoosting (winner)** | **0.154** | **0.720** |

The winner is registered to MLflow and promoted to **Staging**, and the same fitted pipeline is
dumped to `artifacts/model.joblib` for the API's local fallback.

**External benchmark (honesty about limitations).** We benchmarked our model head-to-head against
**benchmark-team's** model on an *identical* test window (delivered orders ≥ 2018-07-01, n=12,507). On that
harder slice, ours scored ROC-AUC **0.659** vs their **0.747**. Their edge is **entirely feature
engineering** (same raw data, same algorithm family): they add **customer geography**
(`customer_state`, `customer_seller_same_state`) and cross-state interaction features, letting them
model seller↔customer distance — the dominant late-delivery driver. Our raw `customers` table
already has `customer_state`, so this is a concrete, planned improvement (§20).

> 📸 **Screenshot placeholder:** the MLflow **experiment view** with the three candidate runs and
> their PR-AUC/ROC-AUC columns side by side — `docs/images/mlflow-compare.png`.

---

## 8. MLflow: experiments, runs & the registry

**Tracking server:** `http://127.0.0.1:5312` (per-group, sqlite backend + file artifacts).

### 8.1 Why there are *many* models in MLflow

Every training run logs **four MLflow runs**, not one:

- one run **per candidate** (`logistic_regression`, `random_forest`, `hist_gradient_boosting`) — each
  logs its params, its four metrics, and the fitted sklearn pipeline as an artifact
  (`mlflow.sklearn.log_model`);
- one **`register_staging`** run that logs the winner as a **pyfunc** model, records the
  `winner_*` metrics, and registers it.

So a single pipeline run produces **3 comparable candidate models** in the experiment, and each
**retrain** (triggered manually or by the CD hook) adds a fresh set plus a **new registered
version** — the registry accumulates `delivery-risk` **v1, v2, …**, with exactly one version
in **Staging** at a time (`archive_existing_versions=True`).

### 8.2 The registered model contract

The registered model is **not** the raw sklearn classifier — it is a thin **pyfunc wrapper**
(`ProbaModel`) whose `predict()` returns **P(late)** (the positive-class probability), not a 0/1
label. This guarantees the API and the batch path both get a **probability** regardless of load
path, which is exactly what the response contract (`late_delivery_probability`) requires. (This is a
deliberate contract *strength* over a plain classifier that would only return a label.)

The model **signature** is inferred from the validation data (`infer_signature`) and an
`input_example` is logged, so the registry records the expected input schema.

### 8.3 `register_model` as a separate, idempotent gate

`pipeline/register.py` is a distinct DAG task (not merged into training): it verifies a registered
version exists and ensures the newest version sits in the configured stage, **failing loudly** if
training ever produced no model. This makes `register_model` a meaningful, re-runnable step.

> 📸 **Screenshot placeholder:** MLflow **Models** page showing `delivery-risk` with
> multiple versions and one in **Staging** — `docs/images/mlflow-registry.png`.

---

## 9. Batch prediction

**File:** `pipeline/batch_predict.py`.

Scores **every row** of `features.featureset_v1` and writes a `predictions.predictions` table that
the monitoring step consumes. Crucially, it **loads the exact same model the API serves** (MLflow
Staging first, local joblib fallback) and **imports the risk thresholds and recommended actions from
`app.config`** — so the batch path and the online API can **never diverge**. Output columns:
`order_id`, `late_delivery_probability`, `risk_level`, `recommended_action`, `model_version`,
`scored_at`. On the last full run it scored **96,456 orders**.

---

## 10. Monitoring, drift detection & the retrain decision

**File:** `pipeline/monitor.py`.

### 10.1 What it measures

- **Score drift (PSI):** Population Stability Index of the model's **score distribution** between the
  reference window (training slice) and the current window (newest slice), using quantile bins fixed
  from the reference.
- **Covariate drift (PSI):** PSI on a compact, interpretable set of **key features**
  (`estimated_delivery_days`, `freight_sum`, `price_sum`, `product_count`, `seller_count`,
  `approval_delay_hours`, `shipping_window_days`, `total_cost_sum`). Calendar features are
  deliberately excluded — they trivially "drift" across any temporal window and would be misleading.
- **Performance decay:** ROC-AUC on the current window.
- **Label drift:** the change in observed late-rate between the two windows.

### 10.2 The retrain decision rule

```
retrain  ⇐  score_psi > 0.2  OR  max_feature_psi > 0.2   (drift)
        OR  current_auc < 0.65                            (performance decay)
```

The decision, the reasons, and all metrics are written to `monitoring.monitoring_metrics` (appended,
so Grafana can chart drift over successive runs) and to `artifacts/monitoring_report.json`. The
monitor prints **only** the boolean retrain flag as its last stdout line, which Airflow captures as
**XCom** for the branch task.

### 10.3 Record-and-alert, not auto-retrain (a deliberate choice)

The DAG **surfaces** the retrain recommendation (`decide_retrain → [flag_retrain | no_retrain]`,
both terminal log tasks) rather than chaining straight back into training. Retraining immediately
after `train_register` on the same data is redundant, and a monitor→train→monitor loop risks a
retrain storm. If we ever automate it, the correct pattern is a **scheduled** monitoring DAG that
triggers a **separate** training DAG, bounded by a schedule + a cooldown guard.

> 📸 **Screenshot placeholder:** `artifacts/monitoring_report.json` (or the `monitoring_metrics`
> table) showing the PSI values and the `retrain_recommended` verdict —
> `docs/images/monitoring-report.png`.

---

## 11. Airflow orchestration

**File:** `airflow/dags/capstone_pipeline.py` — `dag_id = delivery_capstone_workflow`.

### 11.1 How it deploys

The course Airflow runs every group's DAG from a **git-synced monorepo**: our repo's
`airflow/dags/*.py` are pulled into `/opt/airflow/dags/delivery-capstone/delivery/`. **Deploying a DAG
change = push to `main`** — there is no deploy script on the server. (Proven: our tracked
`capstone_pipeline.py` is byte-identical to the live DAG.)

### 11.2 Why the SSH-bridge pattern

The Airflow worker has **neither our code, the Olist CSVs, nor our MLflow**, and it can't reach the
VM's loopback services. So each task is a `BashOperator` that forwards its command through the course
**SSH bridge** (`/opt/airflow/plugins/delivery_group_ssh.sh`) to run *on the group VM*, in the project
venv, against the VM's `.env`:

```python
def bridge(command):        # run on the group VM via SSH bridge
    inner = f"cd {PROJECT_DIR} && {command}"
    return f"{BRIDGE} {shlex.quote(GROUP)} {shlex.quote(inner)}"

def step(module):           # run a pipeline module on the VM's venv
    return bridge(f"{PYTHON} -m {module}")
```

**No credentials live in the DAG** (graded isolation) — every step reads the VM's `~/project/.env`.

### 11.3 The task chain

```
load_raw_data → build_features → train_model → register_model
   → api_smoke_test → batch_predict → monitor
   → decide_retrain → [ flag_retrain | no_retrain ]
```

- `api_smoke_test` (`pipeline/smoke_test.py`) hits `GET /health` and `POST /predict` on the running
  service and asserts a well-formed contract response — failing the task on any problem. It reuses
  the schema's own example payload so it can never drift from the contract.
- `decide_retrain` is a `BranchPythonOperator` that reads the monitor's XCom flag; it defaults to
  `no_retrain` if the flag is missing, so an XCom hiccup never forces an unwanted retrain.

The DAG ran **green end-to-end** on the course Airflow (all 10 tasks, run `api_trigger__1783425803`).

> 📸 **Screenshot placeholder:** the Airflow **Graph view** of a fully green
> `delivery_capstone_workflow` run — `docs/images/airflow-dag-green.png`.

---

## 12. The FastAPI service

**Files:** `app/main.py` (thin endpoints), `config.py`, `schemas.py`, `model_loader.py`,
`predictor.py`, `metrics.py`, `middleware.py`, `logging_config.py`.

### 12.1 Model resolution — never hard-fails startup

`app/model_loader.py` resolves the model at startup in a strict, degrading order, so the service
**always answers** and `/health` reports which source is actually serving:

1. **MLflow registry** — `models:/${MODEL_NAME}/${MODEL_STAGE}` from `MLFLOW_TRACKING_URI`;
2. **local joblib** — `${MODEL_PATH}` (only if the registry has no such model);
3. **bounded arithmetic baseline** — an explainable fallback so the service never dies.

A **fast TCP pre-check** plus **capped MLflow retries** (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=0`,
short timeouts) keep startup quick even when MLflow is down — without this, MLflow's retry backoff
can stall startup ~60s.

### 12.2 The endpoints (complete)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model load status/source |
| GET | `/model-info` | Loaded model name/version/stage, feature count, leakage policy |
| POST | `/predict` | Score one order |
| POST | `/batch-predict` | Score a list of orders |
| GET | `/metrics-summary` | Human-readable JSON rollup of the metrics |
| GET | `/metrics` | Prometheus exposition format |
| GET | `/deploy-status` | CD-hook deploy history (JSON, or `?format=html` flowchart) — *our extra* |

#### `GET /health`
Returns `{status, model_loaded, model_source, error}`. `model_loaded` is `true` only when a real
trained model is serving (not the baseline). `model_source` ∈ `loading | mlflow | joblib | baseline`.
Used by the Docker `HEALTHCHECK`, the CD hook's post-restart poll, and the DAG smoke test.

#### `GET /model-info`
Returns `{source, name, version, stage, n_features, is_real_model, load_error,
temporal_leakage_policy}` — reporting the **actually-loaded** model, not a hardcoded string. In the
live deployment it reports `delivery-risk`, version `1`, stage `Staging`,
`n_features: 45`, `is_real_model: true`, `temporal_leakage_policy: "purchase-time features only"`.

#### `POST /predict`
Body = one `PredictionInput` (the 45-feature purchase-time contract + `order_id`, `extra="forbid"`).
Flow: increment the request counter → ensure a model is loaded → run the forbidden-field check →
align the payload to the model's own `feature_names_in_` → score (`predict_proba`, or `predict` for
the pyfunc probability wrapper) → map probability to a risk level → record prediction metrics.
Returns **exactly** the six contract fields:

```json
{
  "order_id": "abc123",
  "late_delivery_probability": 0.31,
  "risk_level": "medium",
  "recommended_action": "confirm carrier capacity",
  "model_version": "delivery-risk:1",
  "latency": 0.0042
}
```

`risk_level` mapping (`app/config.py`, env-tunable): `high` if p ≥ `HIGH_RISK_THRESHOLD` (0.55),
`medium` if p ≥ `MEDIUM_RISK_THRESHOLD` (0.25), else `low`. Each level maps to a
`recommended_action` (`monitor normally` / `confirm carrier capacity` /
`prioritize fulfillment intervention`). **Resilience:** if the loaded model throws while scoring, the
request degrades to the baseline (counted as a scoring error) rather than returning a 500.

#### `POST /batch-predict`
Body = a JSON **array** of `PredictionInput`. Scores each with the same `_measured_predict` path and
returns a list of the same response objects. One request counter increment for the batch, one
prediction metric per item.

#### `GET /metrics-summary`
A **human-readable JSON** rollup (not Prometheus text): total predictions, risk distribution,
requests by endpoint, scoring errors, prediction-latency stats (count/sum/avg), HTTP request/error
totals, plus `model_version`, `model_source`, and `last_deploy`.

#### `GET /metrics`
The **Prometheus exposition** endpoint (`text/plain`) scraped by the course Prometheus. Refreshes the
`model_loaded` and deploy gauges at scrape time, then returns `generate_latest()`.

#### `GET /deploy-status` *(our addition — deploy observability)*
JSON `{latest, recent (last 20), total_recorded, status}` by default; `?format=html` (or a browser
`Accept: text/html`) renders an **inline-SVG pipeline flowchart** (new commit → fast-forward → test
gate → restart / trigger / import / retrain, each node coloured by outcome) above a recent-runs
table. Degrades to `status: "unknown"` if the CD run-log is absent — it can never affect serving.

> 📸 **Screenshot placeholders:** the **Swagger UI** at `/docs` (`docs/images/swagger.png`); a
> `/predict` request/response in Swagger (`docs/images/predict-demo.png`); the `/deploy-status?format=html`
> flowchart (`docs/images/deploy-flowchart.png`).

### 12.3 Cross-cutting middleware

`LoggingMiddleware` assigns/propagates an **`X-Request-ID`** on every request, times it, emits a
structured access log, records the HTTP-level Prometheus metrics, and converts any unhandled
exception into a **structured 500** (logged with traceback) so no stack trace leaks to the client.
It uses the **matched route pattern** as the metric label (not the raw path) to avoid label-cardinality blow-ups.

---

## 13. Observability: Prometheus metrics

**File:** `app/metrics.py`. All metric names are prefixed `delivery_`. Because several groups share the
Prometheus and emit unprefixed `delivery_*` names, **queries must pin `instance="…:8112"`** (§14).

### 13.1 Metric catalogue & why each was chosen

| Metric | Type | Labels | What it answers |
|---|---|---|---|
| `delivery_http_requests_total` | Counter | `method,endpoint,status` | Traffic & status mix per endpoint (RED "Rate"). |
| `delivery_http_errors_total` | Counter | `endpoint,status` | HTTP failures ≥ 400 (RED "Errors"). |
| `delivery_http_request_latency_seconds` | Histogram | `endpoint` | Per-endpoint latency (RED "Duration"). |
| `delivery_prediction_requests_total` | Counter | `endpoint` | Prediction volume via `/predict` vs `/batch-predict`. |
| `delivery_predictions_total` | Counter | `risk_level` | **Risk-level distribution** of model output. |
| `delivery_prediction_errors_total` | Counter | — | Model scoring failures that degraded to baseline. |
| `delivery_prediction_latency_seconds` | Histogram | — | Model scoring latency (p50/p95/p99). |
| `delivery_model_loaded` | Gauge | — | 1 = a real model is serving, 0 = baseline. |
| `delivery_deploy_last_status` | Gauge | — | 1 = last CD deploy succeeded. |
| `delivery_deploy_last_timestamp_seconds` | Gauge | — | When the last deploy finished (→ "time since"). |
| `delivery_deploy_last_duration_seconds` | Gauge | — | How long the last deploy took. |
| `delivery_deploy_runs_total` | Gauge | — | Deploy attempts recorded in the window. |
| `delivery_deploy_last_commit_info` | Gauge | `commit,status` | Info-metric: the last deployed commit + status. |
| `delivery_deploy_last_retrain_status` | Gauge | — | 1 = the last deploy's Airflow retrain succeeded. |
| `delivery_deploy_last_retrain_info` | Gauge | `state,run_id` | Info-metric: retrain state + Airflow run id. |

The first eight cover the classic **RED method** (Rate, Errors, Duration) plus the domain-specific
signals the brief asks for (prediction count, **risk-level distribution**, scoring errors, model-loaded).
The `deploy_*` gauges give **deployment observability** (see §16) without any extra service — the API
reads the CD hook's JSONL run-log **at scrape time** and reflects it into gauges. String values
(commit, retrain state) ride on **labels** of a constant-`1` info-metric, the standard Prometheus
idiom for exposing text.

---

## 14. Observability: Grafana dashboard

**Dashboard:** `grafana/dashboards/delivery_risk_prometheus.json`
(uid `delivery-delivery-risk-prom`, title *"Delivery — Delivery-Risk Service (Prometheus)"*), imported
into the shared Grafana **delivery folder**. Every panel is **instance-pinned** via an `$instance`
template variable (default `host.docker.internal:8112`) so it never blends other groups' series.

The dashboard is deliberately organised into **two rows with no overlapping panels** — service health
on top, deployment health below.

### 14.1 Row — service health (panels 1–9)

| # | Panel | Query (essence) | What it shows |
|---|---|---|---|
| 1 | **API up** (stat) | `up{job="capstone_group_apis", instance=…}` | Is the service being scraped & alive. |
| 2 | **Model loaded** (stat) | `delivery_model_loaded` | 1 = real model, 0 = baseline. |
| 3 | **Total predictions** (stat) | `sum(delivery_predictions_total)` | Lifetime prediction count. |
| 4 | **Scoring errors** (stat) | `sum(delivery_prediction_errors_total)` | Times scoring degraded to baseline. |
| 5 | **Request rate by endpoint** (timeseries) | `sum by (endpoint) (rate(delivery_http_requests_total[5m]))` | Traffic per endpoint. |
| 6 | **Error rate** (timeseries) | `sum by (status) (rate(delivery_http_errors_total[5m]))` | HTTP ≥ 400 over time. |
| 7 | **Prediction latency p50/p95/p99** (timeseries) | `histogram_quantile(…, rate(delivery_prediction_latency_seconds_bucket[5m]))` | Scoring latency percentiles. |
| 8 | **Risk-level distribution** (piechart) | `sum by (risk_level) (delivery_predictions_total)` | low/medium/high mix. |
| 9 | **Predictions by risk over time** (timeseries) | `sum by (risk_level) (rate(delivery_predictions_total[5m]))` | Risk mix trend. |

### 14.2 Row — "Deployment (CD hook)" (panels 21–27)

| # | Panel | Query | What it shows |
|---|---|---|---|
| 21 | **Last deploy status** (stat) | `delivery_deploy_last_status` | Did the last CD deploy succeed. |
| 22 | **Time since last deploy** (stat) | `time() - delivery_deploy_last_timestamp_seconds` | Freshness of the deployment. |
| 23 | **Last deploy duration** (stat) | `delivery_deploy_last_duration_seconds` | How long it took. |
| 24 | **Deploys recorded** (stat) | `delivery_deploy_runs_total` | Deploy attempts in the window. |
| 26 | **Retrain succeeded** (stat) | `delivery_deploy_last_retrain_status` | Did the triggered retrain finish OK. |
| 25 | **Last deploy commit** (table) | `delivery_deploy_last_commit_info` | Commit SHA + status (from labels). |
| 27 | **Last retrain state** (table) | `delivery_deploy_last_retrain_info` | Retrain state + Airflow run id. |

> 📸 **Screenshot placeholders:** the full dashboard (`docs/images/grafana-overview.png`); a close-up
> of the risk-distribution pie + latency panel (`docs/images/grafana-service.png`); the Deployment
> row (`docs/images/grafana-deploy.png`).

> **Note on live-render limitation:** the shared Grafana has **no Image-Renderer plugin**, so
> server-side PNG export (`/render`) is unavailable and the web UI needs a real login session — take
> screenshots from a logged-in browser rather than an unauthenticated URL.

---

## 15. Observability: structured logging

`app/logging_config.py` emits **structured JSON logs to stdout** (Loki/Grafana-ready), with a
per-request `request_id` carried in a `contextvar`. `LOG_LEVEL` and `LOG_FORMAT` (`json` for
ingestion, `text` for local reading) are env-driven. Each `/predict` logs a `prediction` event with
`order_id`, `late_delivery_probability`, `risk_level`, `model_version`, and `scored_by`; each request
logs an access event with `method`, `endpoint`, `status`, `latency_ms`. Metrics scrapes are logged at
`DEBUG` to keep the stream quiet; 5xx responses are logged at `WARNING`.

> 📸 **Screenshot placeholder:** a few lines of the JSON log stream (`journalctl --user -u
> delivery-capstone-api.service`) showing a prediction event with its `request_id` —
> `docs/images/logs.png`.

---

## 16. CI/CD: the VM-side deployment hook

**Directory:** `ci/`.

### 16.1 Why not GitLab-native CI/CD

GitLab-native CI/CD was **not available** to us: the delivery account is a **Developer** on the
project (no rights to CI/CD secret variables or pipeline triggers — confirmed `403` via the API),
and **no GitLab runner exists** (the only prior pipeline never executed). A mentor grant was not
available. `main` is unprotected, so the VM *can* pull freely. We therefore built a **pull-based
continuous-deployment hook that runs on the group VM**.

### 16.2 How the hook works (`ci/deploy_hook.sh`)

A **systemd user timer** (`delivery-deploy.timer`, every ~2 min, `flock`-serialised) runs the hook:

```
0. reconcile the previous deploy's retrain outcome (one Airflow poll, non-blocking)
1. git fetch origin main    → if no new commit, exit quietly
2. compute changed paths (old..new)
3. git merge --ff-only      → never rewrite/clobber; a divergence records `ff_failed`
4. pytest gate              → a red build is NOT shipped (records `tests_failed`)
5. change-gated actions:
     app/** | pipeline/** | requirements.txt  → restart the API (then poll /health ~30s)
     pipeline/** | app/config.py               → trigger the Airflow retrain DAG (fire-and-forget)
     grafana/dashboards/**                     → re-import the Grafana dashboards
     docs/notebooks only                       → advance the checkout, no heavy action
6. record one JSONL run record (best-effort, cannot change the exit code)
```

**Security:** no secrets in the repo — git auth comes from `~/.git-credentials`; the trigger/import
helpers read `~/.deploy-secrets` (both `chmod 600`, VM-only).

### 16.3 Deploy monitoring (because there's no CI UI)

Every **real** deploy attempt appends a structured record to `~/deploy-runs.jsonl`
(`status ∈ success | tests_failed | ff_failed | error`, changed paths, per-action results,
duration). The **Airflow retrain is asynchronous**: the hook records `trigger: queued` + the
`dag_run_id`, and a sibling `ci/watch_dag.py` — driven by the *same* timer, not a detached process —
polls Airflow once per tick and appends the terminal outcome (`success | failed | timeout`) to
`~/deploy-retrain.jsonl`. The FastAPI service reads both files **on demand** and surfaces them via
`/deploy-status` (flowchart) and the `delivery_deploy_*` gauges (§13) — **no new service, port, or
database**. Verified live end-to-end on the VM: pushes redeployed the API, re-imported Grafana,
triggered Airflow, and the flowchart tracked a retrain **running → success**.

### 16.4 Helper scripts

| File | Role |
|---|---|
| `ci/deploy_hook.sh` | the hook (reconcile → fetch → gate → conditional deploy → record) |
| `ci/record_run.py` | append one JSONL deploy record |
| `ci/watch_dag.py` | reconcile the triggered retrain's outcome (per-tick, non-blocking) |
| `ci/trigger_dag.py` | trigger `delivery_capstone_workflow` on the course Airflow, write back the run id |
| `ci/import_grafana.py` | POST `grafana/dashboards/*.json` to Grafana (overwrite) |
| `ci/systemd/delivery-deploy.{service,timer}` | the oneshot + ~2-min timer |

> 📸 **Screenshot placeholder:** `systemctl --user list-timers` showing the deploy timer, and a tail
> of `~/deploy-hook.log` for one successful deploy — `docs/images/cd-hook.png`.

---

## 17. Docker & deployment

**`Dockerfile`** builds the **API service** image only (training/orchestration are separate):

- base `python:3.11-slim`; deps installed in a cached layer; **only `app/` is copied** (see
  `.dockerignore`);
- runs as an **unprivileged user** (uid 10001) — isolation/security hygiene;
- **no model, data, or secrets baked in** — the model is pulled from MLflow at runtime, or a joblib
  model is mounted via `MODEL_PATH`;
- a `HEALTHCHECK` curls `/health`; `CMD` runs uvicorn on `:8112`.

```bash
docker build -t delivery-api .
docker run --rm -p 8112:8112 --env-file .env delivery-api
# serve a local joblib model:
#   -v /path/to/pipeline.pkl:/models/pipeline.pkl -e MODEL_PATH=/models/pipeline.pkl
```

On the server the API runs as a **systemd user service** (`delivery-capstone-api.service`,
`Restart=always`, linger enabled) so it survives logout/crash/reboot.

---

## 18. Security & isolation

Isolation/security is explicitly graded. What we did, and what we observed:

**What we did:**
- **No credentials in git** — `.env`, `PROJECT_CREDENTIALS.txt`, `ssh_credentials.txt` are all
  gitignored; `.env.example` is the committed template. The DAG and CI scripts carry **zero**
  secrets (they read VM-side files).
- **Least-baked images** — the Docker image contains no model, data, or secrets and runs as a
  non-root user.
- **Env-driven config** everywhere, so nothing is hardcoded per environment.
- **`.gitignore`** excludes `.venv/`, `data/`, `artifacts/`, `mlruns/`, `models/`, `*.csv`, the raw
  `olist_data/`, and the local MLflow state.

**Infra gaps we can cite (course-level, not ours):**
- `ps aux` on the shared host leaks every group's Postgres password via each MLflow server's
  `--backend-store-uri`.
- The shared Postgres is reachable from the **public IP** on port 32112.
- Multiple groups emit **unprefixed** `delivery_*` metrics into the shared Prometheus (name
  collisions) — we defend by instance-pinning every query.

---

## 19. Repository organization & version-control hygiene

Per the mentor's guidance, the GitLab repository has been organised into a clean, conventional
layout. **Committed** (source, tests, docs, infra-as-code, dependency manifest):

```
delivery-project/
├── app/                     # FastAPI service (config, schemas, model_loader, predictor,
│                            #   metrics, middleware, logging, deploy_status/view, main)
├── pipeline/                # load_raw, features, train, register, batch_predict, monitor,
│                            #   smoke_test, db
├── airflow/
│   ├── dags/capstone_pipeline.py          # the graded SSH-bridge DAG
│   ├── dags/delivery_delivery_risk_pipeline.py  # local-Airflow TaskFlow variant
│   └── docker-compose.override.yaml       # local Airflow bridge
├── ci/                      # VM-side CD hook + systemd units + README
├── grafana/dashboards/      # dashboard JSON (+ README)
├── tests/test_api.py        # contract + leakage + observability + deploy-status tests
├── docs/                    # ← this documentation (+ images/)
├── Dockerfile, .dockerignore
├── requirements.txt
├── .env.example             # committed template (real .env is gitignored)
├── STUDENT_BRIEF.md, README.md
└── .gitignore
```

**Never committed** (gitignored): `.venv/`, `.env`, `olist_data/` and any `*.csv`, `artifacts/`,
`models/`, `mlruns/`, local MLflow state, and the two credential files.

**Cleanup performed / recommended for the final push:**
- Consolidate the three overlapping READMEs (`README.md`, `README-1.md`, `README.API.md`) into a
  single top-level `README.md` and move the API-specific notes under `docs/`.
- Remove stray artifacts from the working tree before the final commit:
  `empty_test_text.txt`, `README-1.md`, and the binary `project.pdf`
  (the brief is already captured in `STUDENT_BRIEF.md`).
- Keep `TODO.md` (or fold it into this doc) — it's a useful audit trail of what shipped.
- Branch hygiene: work merged to `main` via merge requests (MR !4 consolidated `Alireza → main`;
  MR !5 added the deploy-monitoring UI); `main` is the canonical checkout on the VM.

> ✅ **For the presentation:** open the GitLab project, show this clean tree, the merged MRs, and
> the green commit history — then state explicitly that the repository was organised (secrets and
> generated artifacts excluded via `.gitignore`, one clear directory per concern, docs under `docs/`).

> 📸 **Screenshot placeholder:** the GitLab repository file tree + the merge-requests page —
> `docs/images/gitlab-repo.png`.

---

## 20. Known limitations & future work

- **Model quality vs benchmark-team.** On an identical test window we trail benchmark-team (ROC-AUC 0.659 vs
  0.747). The gap is **feature engineering**, not the algorithm: they use **customer geography**
  (`customer_state`, seller↔customer same-state) and cross-state interactions. Our `customers` table
  already holds `customer_state`, so the concrete next step is to extend `featureset_v1` (and the API
  contract) with customer-geography + distance-interaction features and retrain. Explicitly deferred,
  not abandoned.
- **Probability calibration.** The winner was trained class-balanced, so mean predicted P(late)
  (~0.41) sits well above the 8% base rate; **ranking** (ROC-AUC 0.72) is fine but the absolute
  probabilities are optimistic. Recalibrate (e.g. Platt/isotonic) or raise the risk thresholds for a
  realistic risk mix in the demo.
- **Shared-DB flakiness (environmental).** The shared Postgres periodically hits global
  `max_connections` ("reserved for SUPERuser"), which can fail a pipeline/DAG run transiently. The
  API is unaffected (model held in memory, no per-request DB). Recommended hardening: bump the DAG
  `default_args.retries` 0 → 2 with a delay so these blips self-heal.
- **Drift-demo data.** Because the source is a fixed historical dataset, "drift" is demonstrated over
  temporal windows; a live stream would use rolling windows.

---

## Appendix A — Demo & screenshot checklist

A suggested order for the live evaluator demo, with the capture to take at each step:

1. **GitLab** — clean repo tree + merged MRs → *"repository organised"*. `gitlab-repo.png`
2. **Airflow** — trigger `delivery_capstone_workflow`, show the fully green graph. `airflow-dag-green.png`
3. **MLflow** — the experiment with 3 candidate runs compared; the registry with multiple
   `delivery-risk` versions and one in **Staging**. `mlflow-compare.png`, `mlflow-registry.png`
4. **Feature importance** — the bar chart; call out `shipping_window_days` as the dominant driver.
   `feature-importance.png`
5. **API** — Swagger `/docs`; a live `/predict` (low & high risk); `/model-info` showing the real
   Staging model; `/health`. `swagger.png`, `predict-demo.png`
6. **Monitoring** — the `monitoring_report.json` / `monitoring_metrics` drift verdict. `monitoring-report.png`
7. **Grafana** — the full dashboard (service row + deployment row), live numbers. `grafana-overview.png`
8. **CD hook** — push a trivial change, watch the timer redeploy; show `/deploy-status?format=html`.
   `deploy-flowchart.png`, `cd-hook.png`
9. **Logs** — the structured JSON log stream with `request_id` correlation. `logs.png`

---

## Appendix B — Command reference

```bash
# ---- environment ----
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt          # local (PyPI)
# server: pip install --no-index --find-links /opt/MLOps/MlOps/project/capstone_stack/wheelhouse ...

# ---- tests ----
.venv/bin/python -m pytest -q tests

# ---- run the API ----
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8112

# ---- run the pipeline by hand (each is an Airflow task) ----
.venv/bin/python -m pipeline.load_raw --clear
.venv/bin/python -m pipeline.features
.venv/bin/python -m pipeline.train
.venv/bin/python -m pipeline.register
.venv/bin/python -m pipeline.smoke_test
.venv/bin/python -m pipeline.batch_predict
.venv/bin/python -m pipeline.monitor

# ---- reproduce the feature-importance table (§5.3) ----
.venv/bin/python -c "import joblib,pandas as pd; \
from app.config import FEATURE_COLUMNS; from sklearn.inspection import permutation_importance; \
df=pd.read_csv('artifacts/featureset_v1.csv'); ts=pd.to_datetime(df.purchase_ts); ev=df[ts>=ts.quantile(0.8)]; \
m=joblib.load('artifacts/model.joblib'); \
r=permutation_importance(m,ev[FEATURE_COLUMNS],ev.is_late_delivery,scoring='roc_auc',n_repeats=5,random_state=42,n_jobs=-1); \
print(pd.Series(r.importances_mean,index=FEATURE_COLUMNS).sort_values(ascending=False).head(10))"
```

---

*End of document. Replace each 📸 placeholder with a screenshot under `docs/images/` before
exporting to PDF for the presentation.*
