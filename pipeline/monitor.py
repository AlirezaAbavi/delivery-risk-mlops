"""Drift monitoring + retraining decision.

Compares a reference window (training slice) against the current window (newest
slice) using PSI (Population Stability Index) on the model's score distribution and
on key features, and checks recent-window ROC-AUC. Emits a decision the Airflow DAG
branches on, and writes ``monitoring_metrics`` to Postgres for Grafana.

Decision rule (retrain if either fires):
  * score PSI > DRIFT_PSI_THRESHOLD        (prediction/covariate drift)
  * recent ROC-AUC  < RETRAIN_MIN_AUC      (performance decay)

Usage:
    python -m pipeline.monitor
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app import config
from .batch_predict import load_scoring_model
from .db import MONITORING_SCHEMA, ensure_schema, get_engine
from .train import FEATURE_COLUMNS, TARGET, load_featureset, temporal_split

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline.monitor")

DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
RETRAIN_MIN_AUC = float(os.getenv("RETRAIN_MIN_AUC", "0.65"))
MONITORING_TABLE = os.getenv("MONITORING_TABLE", "monitoring_metrics")
REPORT_PATH = os.getenv("MONITOR_REPORT_PATH", "artifacts/monitoring_report.json")

# A compact, interpretable set of features to track for covariate drift. Calendar
# features (purchase_month/quarter) are deliberately excluded: they trivially "drift"
# across any temporal window, which would be misleading rather than real concept drift.
KEY_FEATURES = [
    "estimated_delivery_days", "freight_sum", "price_sum", "product_count",
    "seller_count", "approval_delay_hours", "shipping_window_days", "total_cost_sum",
]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins fixed from the reference."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    ref_hist = np.histogram(reference, bins=edges)[0].astype(float)
    cur_hist = np.histogram(current, bins=edges)[0].astype(float)
    eps = 1e-6
    ref_pct = np.clip(ref_hist / ref_hist.sum(), eps, None)
    cur_pct = np.clip(cur_hist / cur_hist.sum(), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _persist(metrics: dict[str, float]) -> None:
    now = datetime.now(timezone.utc)
    rows = pd.DataFrame(
        [{"metric": k, "value": float(v), "computed_at": now} for k, v in metrics.items()]
    )
    # Append so Grafana can chart drift over successive runs.
    ensure_schema(MONITORING_SCHEMA)
    rows.to_sql(MONITORING_TABLE, get_engine(), schema=MONITORING_SCHEMA,
                if_exists="append", index=False)


def run() -> dict:
    frame = load_featureset()
    reference, current, split_ts = temporal_split(frame)

    score, version = load_scoring_model()
    ref_scores = np.clip(score(reference[FEATURE_COLUMNS]), 0.0, 1.0)
    cur_scores = np.clip(score(current[FEATURE_COLUMNS]), 0.0, 1.0)

    score_psi = psi(ref_scores, cur_scores)
    feature_psi = {
        f"psi_{f}": psi(reference[f].to_numpy(dtype=float), current[f].to_numpy(dtype=float))
        for f in KEY_FEATURES
    }
    max_feature_psi = max(feature_psi.values()) if feature_psi else 0.0

    current_auc = float(roc_auc_score(current[TARGET], cur_scores))
    ref_late_rate = float(reference[TARGET].mean())
    cur_late_rate = float(current[TARGET].mean())

    drift_detected = score_psi > DRIFT_PSI_THRESHOLD or max_feature_psi > DRIFT_PSI_THRESHOLD
    performance_low = current_auc < RETRAIN_MIN_AUC
    retrain = bool(drift_detected or performance_low)

    metrics = {
        "score_psi": score_psi,
        "max_feature_psi": max_feature_psi,
        "current_auc": current_auc,
        "reference_late_rate": ref_late_rate,
        "current_late_rate": cur_late_rate,
        "label_drift": abs(cur_late_rate - ref_late_rate),
        **feature_psi,
        "drift_detected": float(drift_detected),
        "performance_low": float(performance_low),
        "retrain_recommended": float(retrain),
    }
    _persist(metrics)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "split_ts": str(split_ts),
        "thresholds": {"psi": DRIFT_PSI_THRESHOLD, "min_auc": RETRAIN_MIN_AUC},
        "metrics": metrics,
        "decision": {
            "drift_detected": drift_detected,
            "performance_low": performance_low,
            "retrain_recommended": retrain,
            "reasons": [
                *(["score_psi>threshold"] if score_psi > DRIFT_PSI_THRESHOLD else []),
                *(["feature_psi>threshold"] if max_feature_psi > DRIFT_PSI_THRESHOLD else []),
                *(["auc<min"] if performance_low else []),
            ] or ["within_thresholds"],
        },
    }
    os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    log.info("monitor: score_psi=%.4f max_feat_psi=%.4f auc=%.4f -> retrain=%s (%s)",
             score_psi, max_feature_psi, current_auc, retrain, report["decision"]["reasons"])
    return report


def main() -> None:
    report = run()
    # Emit ONLY the retrain flag on stdout (last line) so the Airflow BashOperator
    # captures it as XCom for the decide_retrain branch. All logs go to stderr.
    print(str(report["decision"]["retrain_recommended"]).lower())


if __name__ == "__main__":
    main()
