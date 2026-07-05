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
