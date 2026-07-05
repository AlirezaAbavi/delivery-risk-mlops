"""Turn a validated payload into a scored prediction.

Prefers the loaded model (``predict_proba``, falling back to ``predict`` for pyfunc
flavours that only expose point predictions), and degrades to a bounded arithmetic
baseline whenever no trained model is available or scoring fails.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, status

from . import config, metrics
from .model_loader import LoadedModelState
from .schemas import PredictionInput, PredictionResponse

log = logging.getLogger("api.predict")


def _reject_forbidden(features: dict) -> None:
    """Explicit second leakage layer (schema ``extra='forbid'`` is the first)."""
    leaked = sorted(set(features) & set(config.FORBIDDEN_FIELDS))
    if leaked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Request contains forbidden leakage/outcome fields.", "forbidden_fields": leaked},
        )


def _baseline_probability(payload: PredictionInput) -> float:
    # Bounded, explainable fallback when no trained model is loaded.
    return min(0.98, max(0.01,
        0.06
        + 0.018 * payload.estimated_delivery_days
        + 0.0012 * payload.freight_sum
        + 0.012 * (payload.purchase_hour >= 18)
    ))


def _model_probability(state: LoadedModelState, features: dict) -> float:
    import pandas as pd

    cols = state.feature_names or config.FEATURE_COLUMNS
    row = {c: features.get(c, 0) for c in cols}
    frame = pd.DataFrame([row], columns=cols)

    model = state.model
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        value = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
    else:  # pyfunc / regressor-style: treat the scalar output as the score
        out = model.predict(frame)
        value = float(out[0] if hasattr(out, "__len__") else out)
    return min(1.0, max(0.0, value))


def _risk_level(probability: float) -> str:
    if probability >= config.HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= config.MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def predict_one(state: LoadedModelState, payload: PredictionInput) -> PredictionResponse:
    """Score a single payload and record prediction metrics."""
    features = payload.model_dump(exclude={"order_id"})
    _reject_forbidden(features)

    scored_by = state.source
    if state.is_real:
        try:
            probability = _model_probability(state, features)
        except Exception:  # noqa: BLE001 - degrade to baseline rather than 500
            metrics.ERRORS.inc()
            scored_by = "baseline_fallback"
            log.warning("model_scoring_failed", exc_info=True,
                        extra={"order_id": payload.order_id, "model_version": state.version_string})
            probability = _baseline_probability(payload)
    else:
        probability = _baseline_probability(payload)

    level = _risk_level(probability)
    metrics.PREDICTIONS.labels(level).inc()
    log.info("prediction", extra={
        "order_id": payload.order_id,
        "late_delivery_probability": round(probability, 4),
        "risk_level": level,
        "model_version": state.version_string,
        "scored_by": scored_by,
    })
    return PredictionResponse(
        order_id=payload.order_id,
        late_delivery_probability=round(probability, 4),
        risk_level=level,
        recommended_action=config.RECOMMENDED_ACTIONS[level],
        model_version=state.version_string,
        latency=0.0,
    )
