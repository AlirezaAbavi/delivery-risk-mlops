"""Prometheus metrics for the delivery-risk API (Delivery observability requirement)."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- HTTP-level (every request) --------------------------------------------
HTTP_REQUESTS = Counter("delivery_http_requests_total", "HTTP requests", ["method", "endpoint", "status"])
HTTP_ERRORS = Counter("delivery_http_errors_total", "HTTP responses with status >= 400", ["endpoint", "status"])
HTTP_LATENCY = Histogram("delivery_http_request_latency_seconds", "HTTP request latency in seconds", ["endpoint"])

# --- prediction-domain ------------------------------------------------------
REQUESTS = Counter("delivery_prediction_requests_total", "Prediction requests received", ["endpoint"])
PREDICTIONS = Counter("delivery_predictions_total", "Predictions produced", ["risk_level"])
ERRORS = Counter("delivery_prediction_errors_total", "Model scoring failures (degraded to baseline)")
LATENCY = Histogram("delivery_prediction_latency_seconds", "Per-prediction latency in seconds")
MODEL_LOADED = Gauge("delivery_model_loaded", "1 if a trained model is serving, 0 if baseline")

# --- deploy monitoring (CD-hook run records) --------------------------------
# Refreshed from ~/deploy-runs.jsonl at scrape time (see refresh_deploy_gauges).
DEPLOY_LAST_STATUS = Gauge("delivery_deploy_last_status", "1 if the last CD-hook deploy succeeded, 0 otherwise")
DEPLOY_LAST_TIMESTAMP = Gauge("delivery_deploy_last_timestamp_seconds", "Unix time the last deploy finished")
DEPLOY_LAST_DURATION = Gauge("delivery_deploy_last_duration_seconds", "Duration of the last deploy in seconds")
DEPLOY_RUNS = Gauge("delivery_deploy_runs_total", "Deploy attempts recorded in the current run-log window")
# Commit rides a label because Prometheus can't store strings (info-metric idiom).
DEPLOY_LAST_COMMIT = Gauge("delivery_deploy_last_commit_info", "Last deploy commit as a label; value is always 1", ["commit", "status"])
# Retrain outcome for the last deploy that queued one (reconciled async by watch_dag).
DEPLOY_LAST_RETRAIN = Gauge("delivery_deploy_last_retrain_status", "1 if the last deploy's Airflow retrain succeeded, 0 otherwise")
DEPLOY_LAST_RETRAIN_INFO = Gauge("delivery_deploy_last_retrain_info", "Last retrain state as a label; value is always 1", ["state", "run_id"])


def _iso_to_epoch(value: str) -> float:
    from datetime import datetime

    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def refresh_deploy_gauges() -> None:
    """Pull the latest CD-hook record and reflect it in the deploy gauges.

    Called at scrape time so Prometheus/Grafana see deploy status without any
    background thread. Read-only and defensive: any failure leaves gauges as-is.
    """
    from . import deploy_status

    snap = deploy_status.snapshot()
    DEPLOY_RUNS.set(snap.get("total_recorded", 0))
    latest = snap.get("latest")
    if not latest:
        DEPLOY_LAST_STATUS.set(0)
        return
    DEPLOY_LAST_STATUS.set(1 if deploy_status.is_success(latest) else 0)
    DEPLOY_LAST_TIMESTAMP.set(_iso_to_epoch(latest.get("finished_at", "")))
    DEPLOY_LAST_DURATION.set(float(latest.get("duration_seconds", 0) or 0))
    commit = (latest.get("new_commit") or "")[:7]
    DEPLOY_LAST_COMMIT.clear()  # keep only the current commit series
    DEPLOY_LAST_COMMIT.labels(commit=commit, status=latest.get("status", "unknown")).set(1)

    # retrain outcome: the most recent deploy that actually queued a retrain, so
    # the panel persists across later non-retrain deploys (may still be running).
    retrain = deploy_status.latest_retrain() or {}
    rt_state = retrain.get("state")
    DEPLOY_LAST_RETRAIN.set(1 if rt_state == "success" else 0)
    DEPLOY_LAST_RETRAIN_INFO.clear()
    DEPLOY_LAST_RETRAIN_INFO.labels(state=rt_state or "none", run_id=(retrain.get("dag_run_id") or "none")).set(1)


def _counter_by_label(counter, label: str) -> dict:
    out: dict[str, float] = {}
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                out[sample.labels.get(label, "")] = sample.value
    return out


def _counter_total(counter) -> float:
    return sum(s.value for m in counter.collect() for s in m.samples if s.name.endswith("_total"))


def _histogram_stats(histogram) -> dict:
    count = total = 0.0
    for metric in histogram.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count"):
                count = sample.value
            elif sample.name.endswith("_sum"):
                total = sample.value
    return {
        "count": count,
        "sum_seconds": round(total, 6),
        "avg_seconds": round(total / count, 6) if count else 0.0,
    }


def summary() -> dict:
    """Roll the raw counters/histogram up into a human-readable JSON summary."""
    risk = _counter_by_label(PREDICTIONS, "risk_level")
    return {
        "total_predictions": _counter_total(PREDICTIONS),
        "risk_distribution": {level: risk.get(level, 0.0) for level in ("low", "medium", "high")},
        "requests_by_endpoint": _counter_by_label(REQUESTS, "endpoint"),
        "scoring_errors": _counter_total(ERRORS),
        "prediction_latency": _histogram_stats(LATENCY),
        "http_requests_total": _counter_total(HTTP_REQUESTS),
        "http_errors_total": _counter_total(HTTP_ERRORS),
        "http_latency": _histogram_stats(HTTP_LATENCY),
    }
