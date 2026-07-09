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
    """Population Stability Index: how much has a distribution shifted vs. a reference?

    PSI is the standard drift metric. Intuition: bin the *reference* distribution into
    deciles, then ask what fraction of the *current* data falls in each of those same
    bins. If the shape is unchanged PSI≈0; the more mass has moved between bins, the
    larger PSI. Rough industry reading: <0.1 stable, 0.1–0.2 moderate, >0.2 significant
    shift (our DRIFT_PSI_THRESHOLD default).

    Formula: Σ (cur% − ref%) · ln(cur% / ref%) over the bins. Implementation notes:
      - Fix the bin *edges* from the reference quantiles so both windows are compared on
        the same ruler.
      - Drop NaNs; bail out to 0.0 on empty/degenerate input (a constant column yields
        <2 unique edges and can't drift meaningfully).
      - ``eps`` (a tiny floor) avoids ln(0)/divide-by-zero when a bin is empty.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    # Quantile edges => equal-population bins on the reference (robust to skew/outliers).
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    ref_hist = np.histogram(reference, bins=edges)[0].astype(float)
    cur_hist = np.histogram(current, bins=edges)[0].astype(float)
    eps = 1e-6
    # Convert counts to proportions, flooring at eps so the log is always finite.
    ref_pct = np.clip(ref_hist / ref_hist.sum(), eps, None)
    cur_pct = np.clip(cur_hist / cur_hist.sum(), eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _persist(metrics: dict[str, float]) -> None:
    """Append this run's metrics to the monitoring table (one row per metric).

    We APPEND (not replace) so each monitor run adds a timestamped snapshot, letting
    Grafana chart how drift/AUC evolve over successive runs — a time series, not a
    single value. The long/tall shape (metric, value, computed_at) is easy to query.
    """
    now = datetime.now(timezone.utc)
    rows = pd.DataFrame(
        [{"metric": k, "value": float(v), "computed_at": now} for k, v in metrics.items()]
    )
    ensure_schema(MONITORING_SCHEMA)
    rows.to_sql(MONITORING_TABLE, get_engine(), schema=MONITORING_SCHEMA,
                if_exists="append", index=False)


def run() -> dict:
    """Compute drift + performance metrics, decide whether to retrain, persist, report.

    The comparison mirrors training's temporal split exactly: the training window is the
    "reference" (what the model learned) and the newest window is "current" (recent
    production-like data). We score both with the *live* model and compare distributions.
    """
    frame = load_featureset()
    reference, current, split_ts = temporal_split(frame)

    # Score both windows with the same model the API serves, so drift is measured on the
    # thing that actually matters — the deployed model's outputs.
    score, version = load_scoring_model()
    ref_scores = np.clip(score(reference[FEATURE_COLUMNS]), 0.0, 1.0)
    cur_scores = np.clip(score(current[FEATURE_COLUMNS]), 0.0, 1.0)

    # PSI on the *score* distribution: catches prediction drift (the model's outputs
    # shifting), which summarises many small covariate shifts in one number.
    score_psi = psi(ref_scores, cur_scores)
    # Also compute PSI per key input feature: score PSI tells us *that* something moved,
    # while feature PSI tells us *which* input moved — useful for diagnosing the cause of
    # drift. We track the max across features as a single "worst offender" summary.
    feature_psi = {
        f"psi_{f}": psi(reference[f].to_numpy(dtype=float), current[f].to_numpy(dtype=float))
        for f in KEY_FEATURES
    }
    max_feature_psi = max(feature_psi.values()) if feature_psi else 0.0

    # Performance signal: ROC-AUC on the recent window. Because the featureset is
    # purchase-time-only labelled with the eventual outcome, we can measure how well the
    # deployed model still ranks late vs. on-time orders on fresh data.
    current_auc = float(roc_auc_score(current[TARGET], cur_scores))
    # Base late-rate in each window; a big gap is "label drift" (the thing we predict is
    # getting more/less common) even if the inputs look stable.
    ref_late_rate = float(reference[TARGET].mean())
    cur_late_rate = float(current[TARGET].mean())

    # Two independent triggers, either of which justifies a retrain:
    #   - drift_detected: the input/output distribution has shifted (covariate/prediction drift)
    #   - performance_low: the model is measurably worse on recent data (performance decay)
    drift_detected = score_psi > DRIFT_PSI_THRESHOLD or max_feature_psi > DRIFT_PSI_THRESHOLD
    performance_low = current_auc < RETRAIN_MIN_AUC
    retrain = bool(drift_detected or performance_low)

    # Flatten everything into a single flat metric->value dict. The three boolean
    # decisions are stored as 0.0/1.0 floats so they live in the same numeric time series
    # as the PSI/AUC values and can be graphed as step lines in Grafana.
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

    # Build a human/machine-readable report alongside the DB row. This JSON artifact is a
    # durable audit trail: it records not just the numbers but the *thresholds* in force
    # and the specific *reasons* the decision came out as it did — so a later reviewer can
    # reconstruct exactly why a retrain was (or wasn't) triggered on a given run.
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
            # Collect the specific fired conditions into a list. Each `*([...] if cond)`
            # splices in a reason string only when its condition is true; if none fire we
            # fall back to the explicit ["within_thresholds"] so the report is never empty.
            "reasons": [
                *(["score_psi>threshold"] if score_psi > DRIFT_PSI_THRESHOLD else []),
                *(["feature_psi>threshold"] if max_feature_psi > DRIFT_PSI_THRESHOLD else []),
                *(["auc<min"] if performance_low else []),
            ] or ["within_thresholds"],
        },
    }
    # Ensure the artifacts/ directory exists before writing (first run on a clean checkout
    # won't have it). `or "."` guards against REPORT_PATH being a bare filename.
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
