"""Turn a validated payload into a scored prediction.

This is the heart of the serving path. Given the current model state and a payload
that has already passed schema validation, it produces the six-field prediction the
API contract promises. Two design principles drive the code:

  1. Graceful degradation — if a real model is loaded we use it, but any scoring
     failure falls back to a bounded arithmetic baseline instead of 500-ing. A
     serving service that stays up on a degraded answer beats one that crashes.
  2. Defence in depth on leakage — even after schema validation, we re-check the
     payload against the explicit forbidden-field denylist before scoring.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException, status

from . import config, metrics
from .model_loader import LoadedModelState
from .schemas import PredictionInput, PredictionResponse

log = logging.getLogger("api.predict")


def _reject_forbidden(features: dict) -> None:
    """Explicit second leakage layer (schema ``extra='forbid'`` is the first).

    Even though Pydantic already rejects unknown fields, we intersect the incoming
    keys with the known outcome/review denylist and raise a *clear* 400 naming the
    offending fields. This makes the leakage guarantee auditable: anyone
    can send ``is_late_delivery`` and see it explicitly refused, not just dropped.
    """
    leaked = sorted(set(features) & set(config.FORBIDDEN_FIELDS))
    if leaked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Request contains forbidden leakage/outcome fields.", "forbidden_fields": leaked},
        )


def _baseline_probability(payload: PredictionInput) -> float:
    """Bounded, explainable fallback used when no trained model is available.

    This is deliberately simple and hand-tuned, not learned: a base rate plus small
    bumps for a longer promised ETA, higher freight, and evening purchases. Its job
    is only to keep the service answering something sane. ``min/max`` clamp the result
    to [0.01, 0.98] so it never returns an absurd 0 or 1.
    """
    return min(0.98, max(0.01,
        0.06                                       # base late rate
        + 0.018 * payload.estimated_delivery_days  # longer promised window => more exposure
        + 0.0012 * payload.freight_sum             # pricier/heavier freight trends later
        + 0.012 * (payload.purchase_hour >= 18)    # evening orders miss same-day cutoffs
    ))


def _model_probability(state: LoadedModelState, features: dict) -> float:
    """Score with the real model, returning P(late) as a plain float in [0, 1].

    The subtle-but-important parts:
      - Column alignment: we build the row in the model's own expected order
        (``feature_names_in_`` if it exposes one, else the config contract). sklearn
        models key on column *position/name*, so a misaligned frame scores garbage.
      - Missing values default to 0 so a caller omitting an Optional field still works.
      - Two output shapes are handled: classifiers expose ``predict_proba`` (take the
        positive-class column); pyfunc/regressor flavours only expose ``predict`` and
        return a scalar we treat as the score directly.
    """
    import pandas as pd

    cols = state.feature_names or config.FEATURE_COLUMNS
    row = {c: features.get(c, 0) for c in cols}
    # Pass ``columns=cols`` so the DataFrame's column order is pinned to the model's.
    frame = pd.DataFrame([row], columns=cols)

    model = state.model
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        # A binary classifier returns [[P(0), P(1)]]; take column 1 = P(late). Guard the
        # degenerate single-column case (a model that only ever saw one class).
        value = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
    else:  # pyfunc / regressor-style: treat the scalar output as the score
        out = model.predict(frame)
        value = float(out[0] if hasattr(out, "__len__") else out)
    # Final clamp: never let a model quirk push the probability outside [0, 1].
    return min(1.0, max(0.0, value))


def _risk_level(probability: float) -> str:
    """Bucket a probability into the ops-facing risk band using the config thresholds.

    Order matters: check high first, then medium, else low. Thresholds live in config
    so the business can re-tune risk appetite without touching this logic.
    """
    if probability >= config.HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= config.MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def predict_one(state: LoadedModelState, payload: PredictionInput) -> PredictionResponse:
    """Score a single payload, record metrics/logs, and return the contract response.

    This orchestrates the whole per-prediction flow: strip the id, enforce the
    leakage denylist, score (real model with baseline fallback), bucket into a risk
    level, emit observability signals, and assemble the response.
    """
    # ``model_dump`` turns the validated model back into a plain dict. We drop
    # ``order_id`` here so it is impossible for the identifier to be used as a feature.
    features = payload.model_dump(exclude={"order_id"})
    _reject_forbidden(features)

    # ``scored_by`` records which path produced the number, for the structured log.
    scored_by = state.source
    if state.is_real:
        try:
            probability = _model_probability(state, features)
        except Exception:  # noqa: BLE001 - degrade to baseline rather than 500
            # A real model is loaded but this particular row blew up (bad category,
            # shape mismatch, etc.). We count the error for alerting and fall back to
            # the baseline so the caller still gets a usable answer.
            metrics.ERRORS.inc()
            scored_by = "baseline_fallback"
            log.warning("model_scoring_failed", exc_info=True,
                        extra={"order_id": payload.order_id, "model_version": state.version_string})
            probability = _baseline_probability(payload)
    else:
        # No trained model resolved at all — serve the baseline by design.
        probability = _baseline_probability(payload)

    level = _risk_level(probability)
    # Count this prediction by risk band; this Counter powers the risk-distribution
    # panel in Grafana and the /metrics-summary rollup.
    metrics.PREDICTIONS.labels(level).inc()
    # One structured JSON log line per prediction — the audit trail an ops team needs
    # (rounded probability keeps logs readable; scored_by shows model vs. fallback).
    log.info("prediction", extra={
        "order_id": payload.order_id,
        "late_delivery_probability": round(probability, 4),
        "risk_level": level,
        "model_version": state.version_string,
        "scored_by": scored_by,
    })
    # ``latency`` is filled in by the caller (main._measured_predict) which times the
    # whole call; we return 0.0 here as a placeholder to satisfy the response model.
    return PredictionResponse(
        order_id=payload.order_id,
        late_delivery_probability=round(probability, 4),
        risk_level=level,
        recommended_action=config.RECOMMENDED_ACTIONS[level],
        model_version=state.version_string,
        latency=0.0,
    )
