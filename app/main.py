from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from . import config, metrics
from .logging_config import configure_logging
from .middleware import LoggingMiddleware
from .model_loader import ModelService
from .predictor import predict_one
from .schemas import HealthResponse, ModelInfoResponse, PredictionInput, PredictionResponse

configure_logging()
log = logging.getLogger("api.main")
model_service = ModelService()


def _load_and_log() -> None:
    model_service.load()
    state = model_service.state
    metrics.MODEL_LOADED.set(1 if state.is_real else 0)
    log.info("model_loaded", extra={
        "source": state.source, "is_real_model": state.is_real,
        "model_version": state.version_string, "load_error": state.error,
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("service_starting", extra={"app": config.APP_TITLE, "version": config.APP_VERSION})
    # Resolve the model at startup. A fast TCP pre-check + capped retries keep this
    # quick even when MLflow is down, so the model is ready before the first request.
    if config.LOAD_MODEL_ON_STARTUP:
        _load_and_log()
    yield


app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION, lifespan=lifespan)
app.add_middleware(LoggingMiddleware)


def _measured_predict(payload: PredictionInput) -> PredictionResponse:
    started = perf_counter()
    response = predict_one(model_service.state, payload)
    latency = perf_counter() - started
    metrics.LATENCY.observe(latency)
    response.latency = round(latency, 6)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = model_service.state
    status_str = "loading" if state.source == "loading" else "ok"
    return HealthResponse(
        status=status_str, model_loaded=state.is_real,
        model_source=state.source, error=state.error,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.model_info())


@app.post("/predict", response_model=PredictionResponse)
def single_prediction(payload: PredictionInput) -> PredictionResponse:
    metrics.REQUESTS.labels("/predict").inc()
    model_service.ensure_loaded()
    return _measured_predict(payload)


@app.post("/batch-predict", response_model=list[PredictionResponse])
def batch_prediction(payloads: list[PredictionInput]) -> list[PredictionResponse]:
    metrics.REQUESTS.labels("/batch-predict").inc()
    model_service.ensure_loaded()
    return [_measured_predict(payload) for payload in payloads]


@app.get("/metrics-summary")
def metrics_summary() -> dict:
    metrics.MODEL_LOADED.set(1 if model_service.state.is_real else 0)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_service.state.version_string,
        "model_source": model_service.state.source,
        **metrics.summary(),
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    metrics.MODEL_LOADED.set(1 if model_service.state.is_real else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
