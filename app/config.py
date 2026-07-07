"""Central configuration for the Delivery delivery-risk API.

Everything env-driven so the same image runs locally and on the server. Holds the
purchase-time feature contract (featureset_v1) and the leakage denylist.
"""
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "MLOps Delivery delivery-risk API"
APP_VERSION = "1.2.0"  # deploy-monitoring: /deploy-status + delivery_deploy_* metrics + async retrain-outcome tracking

# --- model source ----------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5312")
MODEL_NAME = os.getenv("MODEL_NAME", "delivery-risk")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
MODEL_PATH = os.getenv("MODEL_PATH")  # local joblib fallback; unset -> skip

# Load in a background thread so a slow/unreachable MLflow can't block startup.
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes", "y"}
# Keep MLflow probes short: cap per-request timeout AND retries (retries dominate).
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3"))
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", os.getenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0"))
MLFLOW_PROBE_TIMEOUT = float(os.getenv("MLFLOW_PROBE_TIMEOUT", "1.5"))

BASELINE_VERSION = "reference-baseline"

# --- deploy monitoring -----------------------------------------------------
# JSONL run-records written by ci/deploy_hook.sh. Read-only, on demand, so the
# service can surface CD-hook history without a separate platform. Both the hook
# and this service run as the same VM user, so the home-dir default resolves.
DEPLOY_RUNS_PATH = os.getenv("DEPLOY_RUNS_PATH", os.path.expanduser("~/deploy-runs.jsonl"))
DEPLOY_RETRAIN_PATH = os.getenv("DEPLOY_RETRAIN_PATH", os.path.expanduser("~/deploy-retrain.jsonl"))
DEPLOY_HISTORY_LIMIT = int(os.getenv("DEPLOY_HISTORY_LIMIT", "20"))

# --- risk policy -----------------------------------------------------------
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.55"))
MEDIUM_RISK_THRESHOLD = float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.25"))
RECOMMENDED_ACTIONS = {
    "low": "monitor normally",
    "medium": "confirm carrier capacity",
    "high": "prioritize fulfillment intervention",
}

# --- feature contract (featureset_v1: purchase-time only) ------------------
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
# `extra='forbid'` on the schema already rejects unknown fields; this list gives
# an explicit, well-labelled second layer (graded temporal-leakage firewall).
FORBIDDEN_FIELDS: List[str] = [
    "is_late_delivery",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_delivery_date",
    "actual_delivery_days",
    "delivery_delay_days",
    "review_score",
    "review_comment_message",
    "review_creation_date",
]
