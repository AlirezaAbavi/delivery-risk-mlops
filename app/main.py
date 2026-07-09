"""FastAPI application entry point — the Delivery graded deliverable.

This module wires the whole service together but keeps almost no logic of its own:
it defines the six contract endpoints (+ /deploy-status) and delegates to the
focused modules — model_loader (what to serve), predictor (how to score), metrics
(observability), deploy_status/deploy_view (CI/CD surfacing). Keeping the HTTP layer
thin means each concern is testable in isolation.

The object FastAPI serves is ``app`` at the bottom; ``uvicorn app.main:app`` runs it.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import HTMLResponse, JSONResponse, Response

from . import config, deploy_status, metrics
from .deploy_view import render_deploy_html
from .logging_config import configure_logging
from .middleware import LoggingMiddleware
from .model_loader import ModelService
from .predictor import predict_one
from .schemas import HealthResponse, ModelInfoResponse, PredictionInput, PredictionResponse

# Configure structured JSON logging once, at import time, before anything logs.
configure_logging()
log = logging.getLogger("api.main")
# One long-lived model service for the whole process (holds the current model state).
model_service = ModelService()


def _load_and_log() -> None:
    """Resolve the model and emit a single structured line describing the outcome.

    Also mirrors the "is a real model serving?" fact into the delivery_model_loaded
    gauge so Prometheus/Grafana can alert if the service silently drops to baseline.
    """
    model_service.load()
    state = model_service.state
    metrics.MODEL_LOADED.set(1 if state.is_real else 0)
    log.info("model_loaded", extra={
        "source": state.source, "is_real_model": state.is_real,
        "model_version": state.version_string, "load_error": state.error,
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook: runs once on startup (before the ``yield``) and once on
    shutdown (after). We use it to resolve the model *before* the first request, so
    /predict is never the call that pays the load cost.

    The fast TCP pre-check + capped MLflow retries (see config/model_loader) keep this
    quick even when the tracking server is down, so startup stays responsive.
    """
    log.info("service_starting", extra={"app": config.APP_TITLE, "version": config.APP_VERSION})
    if config.LOAD_MODEL_ON_STARTUP:
        _load_and_log()
    yield
    # (nothing to tear down on shutdown; the model has no open handles)


# Construct the app and attach the request-logging / HTTP-metrics middleware.
app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)
app.add_middleware(LoggingMiddleware)


def _measured_predict(payload: PredictionInput) -> PredictionResponse:
    """Score one payload while measuring wall-clock latency.

    ``perf_counter`` is a monotonic high-resolution timer (immune to clock changes),
    the right tool for measuring durations. We record the latency into the histogram
    (for Prometheus) and also stamp it onto the response (for the caller).
    """
    started = perf_counter()
    response = predict_one(model_service.state, payload)
    latency = perf_counter() - started
    metrics.LATENCY.observe(latency)
    response.latency = round(latency, 6)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + model-load status. Cheap and dependency-free so Docker's HEALTHCHECK
    and any uptime probe can hit it constantly. Reports 'loading' only while the very
    first load is still in flight; otherwise 'ok' plus which source is serving."""
    state = model_service.state
    status_str = "loading" if state.source == "loading" else "ok"
    return HealthResponse(
        status=status_str, model_loaded=state.is_real,
        model_source=state.source, error=state.error,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Describe the actually-loaded model (name/version/stage/#features + leakage
    policy). Everything is read live from the model state, not hardcoded, so it always
    reflects the real serving version."""
    return ModelInfoResponse(**model_service.model_info())


@app.post("/predict", response_model=PredictionResponse)
def single_prediction(payload: PredictionInput) -> PredictionResponse:
    """Score one order. FastAPI has already validated ``payload`` against the schema
    (rejecting bad types / leaky fields) before we get here."""
    metrics.REQUESTS.labels("/predict").inc()   # count by endpoint for the summary/dashboard
    model_service.ensure_loaded()                # lazy-load safety net if startup load was off
    return _measured_predict(payload)


@app.post("/batch-predict", response_model=list[PredictionResponse])
def batch_prediction(payloads: list[PredictionInput]) -> list[PredictionResponse]:
    """Score many orders in one call — the shape an ops batch job uses. Each element is
    validated independently; we score them sequentially and return the aligned list."""
    metrics.REQUESTS.labels("/batch-predict").inc()
    model_service.ensure_loaded()
    return [_measured_predict(payload) for payload in payloads]


@app.get("/metrics-summary")
def metrics_summary() -> dict:
    """Human-readable JSON rollup of the raw Prometheus counters/histograms.

    Useful for a quick eyeball or a demo without a PromQL query. We refresh the
    model-loaded and deploy gauges first so the snapshot is current, then merge the
    metrics summary with model/deploy context.
    """
    metrics.MODEL_LOADED.set(1 if model_service.state.is_real else 0)
    metrics.refresh_deploy_gauges()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_service.state.version_string,
        "model_source": model_service.state.source,
        "last_deploy": deploy_status.latest(),
        **metrics.summary(),  # spread the counter/histogram rollup into this dict
    }


@app.get("/deploy-status")
def deploy_status_route(request: Request, format: str | None = None) -> Response:
    """CD-hook deploy history (latest run + recent attempts).

    Content negotiation: returns JSON by default, but ``?format=html`` — or a browser
    sending ``Accept: text/html`` — gets a small standalone HTML view (a flowchart of
    the deploy pipeline). This lets the same endpoint serve both machines and a human
    clicking it in a browser. Robust to a missing run-log: degrades to status=unknown
    rather than erroring, so deploy monitoring can never impact serving.
    """
    snapshot = deploy_status.snapshot()
    wants_html = format == "html" or (
        format is None and "text/html" in request.headers.get("accept", "")
    )
    if wants_html:
        return HTMLResponse(render_deploy_html(snapshot))
    return JSONResponse(snapshot)


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """The Prometheus scrape endpoint. Prometheus polls this on a schedule; we refresh
    the gauges that are computed on demand (model-loaded, deploy status) right before
    serialising, then hand back the standard text exposition format."""
    metrics.MODEL_LOADED.set(1 if model_service.state.is_real else 0)
    metrics.refresh_deploy_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
