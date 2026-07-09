"""Prometheus metrics for the delivery-risk API.

A quick primer on the three metric types used here:
  - Counter: a value that only ever goes up (request counts, error counts). You graph
    its *rate* in Grafana, e.g. requests/sec.
  - Histogram: samples observations into buckets AND tracks _count and _sum, so you can
    derive averages and quantiles (used for latency).
  - Gauge: a value that can go up or down and represents "current state" (is a model
    loaded? did the last deploy succeed?).

Naming note: every metric is prefixed ``delivery_*`` so it sits in its own namespace
and is easy to select in Prometheus/Grafana without colliding with unrelated series.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --- HTTP-level (recorded by the middleware for *every* request) ------------
# Labelled by method/endpoint/status so we can slice traffic and error rates by route.
# ``endpoint`` uses the matched *route pattern* (e.g. "/predict"), not the raw path, to
# avoid unbounded label cardinality — a key Prometheus discipline.
HTTP_REQUESTS = Counter("delivery_http_requests_total", "HTTP requests", ["method", "endpoint", "status"])
HTTP_ERRORS = Counter("delivery_http_errors_total", "HTTP responses with status >= 400", ["endpoint", "status"])
HTTP_LATENCY = Histogram("delivery_http_request_latency_seconds", "HTTP request latency in seconds", ["endpoint"])

# --- prediction-domain (business-level signals) -----------------------------
REQUESTS = Counter("delivery_prediction_requests_total", "Prediction requests received", ["endpoint"])
PREDICTIONS = Counter("delivery_predictions_total", "Predictions produced", ["risk_level"])  # -> risk distribution
ERRORS = Counter("delivery_prediction_errors_total", "Model scoring failures (degraded to baseline)")
LATENCY = Histogram("delivery_prediction_latency_seconds", "Per-prediction latency in seconds")
MODEL_LOADED = Gauge("delivery_model_loaded", "1 if a trained model is serving, 0 if baseline")

# --- deploy monitoring (CD run records) -------------------------------------
# These gauges are NOT updated by a background thread; they are recomputed from
# ~/deploy-runs.jsonl at scrape time (see refresh_deploy_gauges). That keeps the
# service stateless — the JSONL files written by the CD recorder are the source of truth.
DEPLOY_LAST_STATUS = Gauge("delivery_deploy_last_status", "1 if the last CD-hook deploy succeeded, 0 otherwise")
DEPLOY_LAST_TIMESTAMP = Gauge("delivery_deploy_last_timestamp_seconds", "Unix time the last deploy finished")
DEPLOY_LAST_DURATION = Gauge("delivery_deploy_last_duration_seconds", "Duration of the last deploy in seconds")
DEPLOY_RUNS = Gauge("delivery_deploy_runs_total", "Deploy attempts recorded in the current run-log window")
# Prometheus values are numeric only, so a *string* like a commit hash can't be a value.
# The idiom is an "info metric": put the string in a label and set the value to 1.
DEPLOY_LAST_COMMIT = Gauge("delivery_deploy_last_commit_info", "Last deploy commit as a label; value is always 1", ["commit", "status"])
# Retrain outcome for the last deploy that queued one. It's filled in asynchronously
# once Airflow finishes, so it can legitimately read "running" for a while.
DEPLOY_LAST_RETRAIN = Gauge("delivery_deploy_last_retrain_status", "1 if the last deploy's Airflow retrain succeeded, 0 otherwise")
DEPLOY_LAST_RETRAIN_INFO = Gauge("delivery_deploy_last_retrain_info", "Last retrain state as a label; value is always 1", ["state", "run_id"])


def _iso_to_epoch(value: str) -> float:
    """Convert an ISO-8601 timestamp string into a Unix epoch float for a gauge.

    Prometheus timestamps are seconds-since-epoch. The CD recorder writes ISO strings
    (human-friendly), so we convert here. ``Z`` (UTC) isn't understood by
    ``fromisoformat`` on older Pythons, so we normalise it to ``+00:00``. Any parse
    failure returns 0.0 rather than raising — a bad record must not break a scrape.
    """
    from datetime import datetime

    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def refresh_deploy_gauges() -> None:
    """Pull the latest CD record and reflect it in the deploy gauges.

    Called at scrape time (from /metrics and /metrics-summary) so Prometheus/Grafana
    see current deploy status with no background thread. Fully read-only and
    defensive: if there's no record we set status=0 and return; any per-field parse
    issue leaves that gauge as-is rather than raising.
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
    commit = (latest.get("new_commit") or "")[:7]  # short SHA is enough to identify a commit
    # ``clear()`` drops the previous commit's label series so only the current commit is
    # exported — otherwise every past commit would linger as its own time series forever.
    DEPLOY_LAST_COMMIT.clear()
    DEPLOY_LAST_COMMIT.labels(commit=commit, status=latest.get("status", "unknown")).set(1)

    # Retrain outcome: we surface the most recent deploy that *actually queued* a
    # retrain, not simply the latest deploy. That way the retrain panel keeps showing
    # the last real retrain result even across later deploys that didn't retrain (and
    # the state may still be "running" until the terminal outcome is recorded).
    retrain = deploy_status.latest_retrain() or {}
    rt_state = retrain.get("state")
    DEPLOY_LAST_RETRAIN.set(1 if rt_state == "success" else 0)
    DEPLOY_LAST_RETRAIN_INFO.clear()
    DEPLOY_LAST_RETRAIN_INFO.labels(state=rt_state or "none", run_id=(retrain.get("dag_run_id") or "none")).set(1)


def _counter_by_label(counter, label: str) -> dict:
    """Read a labelled Counter's current per-label totals into a plain dict.

    prometheus_client doesn't expose "current value" directly, so we walk the
    collected samples and keep the ``*_total`` ones, keyed by the requested label.
    Used to turn e.g. the risk-level counter into {"low": n, "medium": m, ...}.
    """
    out: dict[str, float] = {}
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                out[sample.labels.get(label, "")] = sample.value
    return out


def _counter_total(counter) -> float:
    """Sum all label series of a Counter into a single grand total."""
    return sum(s.value for m in counter.collect() for s in m.samples if s.name.endswith("_total"))


def _histogram_stats(histogram) -> dict:
    """Derive count / total / average seconds from a Histogram's _count and _sum samples.

    A histogram exports a ``*_count`` (number of observations) and a ``*_sum`` (total
    of all observed values). Average = sum / count (guarded against divide-by-zero).
    """
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
    """Roll the raw counters/histogram up into a human-readable JSON summary.

    This is what /metrics-summary returns: the same data Prometheus scrapes, but
    pre-aggregated so a human (or a demo) can read prediction volume, the risk-level
    distribution, per-endpoint request counts, error totals, and latency stats at a
    glance — without writing a single PromQL query.
    """
    risk = _counter_by_label(PREDICTIONS, "risk_level")
    return {
        "total_predictions": _counter_total(PREDICTIONS),
        # Force all three buckets to appear (defaulting to 0) so the shape is stable.
        "risk_distribution": {level: risk.get(level, 0.0) for level in ("low", "medium", "high")},
        "requests_by_endpoint": _counter_by_label(REQUESTS, "endpoint"),
        "scoring_errors": _counter_total(ERRORS),
        "prediction_latency": _histogram_stats(LATENCY),
        "http_requests_total": _counter_total(HTTP_REQUESTS),
        "http_errors_total": _counter_total(HTTP_ERRORS),
        "http_latency": _histogram_stats(HTTP_LATENCY),
    }
