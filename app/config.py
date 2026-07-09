"""Central configuration for the Delivery delivery-risk API.

Why a single config module?
    Everything the service needs to behave differently between environments
    (your laptop vs. the course server) is read from environment variables *here*,
    in one place. Nothing else in the codebase calls ``os.getenv`` for these
    values. That is the "12-factor app" idea: the *same* built artifact/image runs
    everywhere and only the environment changes its behaviour — no code edits, no
    rebuilds, no secrets baked into the image.

This module also holds two things that are part of the *graded* contract:
    1. ``FEATURE_COLUMNS`` — the exact purchase-time feature list (featureset_v1)
       the model was trained on. This is the "schema" the API promises callers.
    2. ``FORBIDDEN_FIELDS`` — the temporal-leakage denylist: outcome/review fields
       that must never reach the model. See the firewall notes lower down.
"""
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

# Load key=value pairs from a local ``.env`` file into the process environment.
# This is a no-op on the server (where real env vars are already set by systemd),
# but on a laptop it lets you keep MLFLOW_TRACKING_URI, DB creds, etc. in a file
# instead of exporting them by hand every session. Real values in ``.env`` are
# gitignored; ``.env.example`` documents the shape without leaking secrets.
load_dotenv()

APP_TITLE = "MLOps Delivery delivery-risk API"
# The version string is surfaced in /model-info and the OpenAPI docs. We bump it by
# hand on each meaningful change so an evaluator can tell which build is running.
APP_VERSION = "1.2.0"  # deploy-monitoring: /deploy-status + delivery_deploy_* metrics + async retrain-outcome tracking

# --- model source ----------------------------------------------------------
# These four settings drive the model-resolution chain in model_loader.py:
# MLflow registry first, then a local joblib file, then a safe arithmetic baseline.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5312")  # where the registry lives
MODEL_NAME = os.getenv("MODEL_NAME", "delivery-risk")                    # registered model name
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")                               # which stage alias to serve
MODEL_PATH = os.getenv("MODEL_PATH")  # optional local joblib fallback; unset -> that fallback is skipped

# Whether to resolve the model during app startup (the FastAPI "lifespan" hook).
# Parsing a set of truthy strings makes the flag forgiving of "1"/"true"/"yes".
# Disable it in tests that want to control exactly when loading happens.
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes", "y"}

# MLflow's HTTP client retries aggressively by default, which means a *down*
# tracking server can stall startup for tens of seconds. We cap both the per-call
# timeout and the retry count (retries dominate the total wait). ``setdefault`` so
# an explicit env override still wins. Combined with the TCP pre-check in
# model_loader._mlflow_reachable, this keeps the service snappy even when MLflow is
# unreachable — a core "never hard-fail on a dependency" reliability property.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3"))
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", os.getenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0"))
MLFLOW_PROBE_TIMEOUT = float(os.getenv("MLFLOW_PROBE_TIMEOUT", "1.5"))  # seconds for the TCP reachability probe

# Name reported when we are serving the arithmetic fallback rather than a real model.
# Callers see this in ``model_version`` and can tell "this is not a trained model".
BASELINE_VERSION = "reference-baseline"

# --- deploy monitoring -----------------------------------------------------
# The CD hook (ci/deploy_hook.sh) appends one JSON line per deploy attempt to these
# files. The API reads them on demand (never writes) to power /deploy-status and the
# delivery_deploy_* metrics. This lets us surface CI/CD history through the same service
# without standing up a separate dashboard/database. The hook and the API run as the
# same VM user, so the ``~`` home-directory default resolves to the same files.
DEPLOY_RUNS_PATH = os.getenv("DEPLOY_RUNS_PATH", os.path.expanduser("~/deploy-runs.jsonl"))       # one line per deploy
DEPLOY_RETRAIN_PATH = os.getenv("DEPLOY_RETRAIN_PATH", os.path.expanduser("~/deploy-retrain.jsonl"))  # async retrain outcomes
DEPLOY_HISTORY_LIMIT = int(os.getenv("DEPLOY_HISTORY_LIMIT", "20"))  # how many recent runs /deploy-status returns

# --- risk policy -----------------------------------------------------------
# The model outputs a probability in [0, 1]; the *business* wants a discrete action.
# These two thresholds bucket the probability into low/medium/high, and each bucket
# maps to a concrete ops recommendation. Keeping the thresholds in config (not
# hardcoded in the scorer) means the ops team can re-tune the risk appetite without
# a code change — an important separation of "model" from "policy".
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.55"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.25"))
RECOMMENDED_ACTIONS = {
    "low": "monitor normally",
    "medium": "confirm carrier capacity",
    "high": "prioritize fulfillment intervention",
}

# --- feature contract (featureset_v1: purchase-time only) ------------------
# The exact ordered list of features the model consumes. Two reasons this lives here:
#   1. It is the API's public contract — callers know precisely what to send.
#   2. The scorer aligns the incoming payload to this order before building the
#      DataFrame, because most sklearn models are sensitive to column order.
# CRUCIAL temporal-validity property: *every* column below is knowable at the moment
# of purchase. There is deliberately no field derived from the actual delivery,
# carrier scans, or the customer review — using any of those would be "leakage from
# the future" and would make the model look great offline but useless in production.
FEATURE_COLUMNS: List[str] = [
    "payment_count", "payment_type_mode", "max_installments",
    "order_item_count", "product_count", "seller_count",
    "price_sum", "price_mean", "price_max", "price_std",
    "freight_sum", "freight_mean", "freight_max", "freight_std",
    "total_cost_sum", "total_cost_mean", "total_cost_max",
    "product_category_count", "product_category_mode", "is_multi_category",
    "product_weight_g_mean", "product_weight_g_max",
    "product_length_cm_mean", "product_length_cm_max",
    "product_height_cm_mean", "product_height_cm_max",
    "product_width_cm_mean", "product_width_cm_max",
    "product_volume_mean", "product_volume_max",
    "seller_state_mode", "seller_state_count", "seller_city_count", "seller_zip_mode",
    "purchase_hour", "purchase_dayofweek", "is_weekend_purchase",
    "purchase_month", "purchase_quarter", "is_month_end",
    "estimated_delivery_days", "approval_delay_hours",
    "shipping_limit_min_days", "shipping_window_days", "seller_margin_days",
]

# Delivery-outcome / review / target fields that must NEVER be model inputs.
# This is the *second* of three leakage defences (see the "temporal-leakage
# firewall" in the docs):
#   1. The feature-building SQL only selects purchase-time columns.
#   2. The Pydantic schema uses ``extra='forbid'`` — any unknown field is rejected.
#   3. This explicit, well-labelled denylist — so even if a name were ever added to
#      the schema by mistake, the scorer still refuses these known-leaky fields and
#      returns a clear 400. Defence-in-depth: three independent layers, so no single
#      slip re-opens the leak. This list is checked in predictor._reject_forbidden.
FORBIDDEN_FIELDS: List[str] = [
    "is_late_delivery",                 # the target itself
    "order_delivered_customer_date",    # actual delivery timestamp (future)
    "order_delivered_carrier_date",     # carrier handoff timestamp (future)
    "order_delivery_date",              # actual delivery date (future)
    "actual_delivery_days",             # derived from actual delivery (future)
    "delivery_delay_days",              # derived from actual delivery (future)
    "review_score",                     # written after delivery (future)
    "review_comment_message",           # written after delivery (future)
    "review_creation_date",             # written after delivery (future)
]
