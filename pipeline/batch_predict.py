"""Batch-score the feature table and persist predictions to Postgres.

Loads the same model the API serves (MLflow Staging first, local joblib fallback),
scores every row of ``featureset_v1``, and writes a ``predictions`` table that the
monitoring/drift step consumes. Risk thresholds and recommended actions come from
``app.config`` so the batch path and the API never diverge.

Usage:
    python -m pipeline.batch_predict            # score the whole feature table
    python -m pipeline.batch_predict --limit 100
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from app import config
from .db import FEATURES_SCHEMA, PREDICTIONS_SCHEMA, ensure_schema, get_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline.batch_predict")

FEATURESET_TABLE = os.getenv("FEATURESET_TABLE", "featureset_v1")
PREDICTIONS_TABLE = os.getenv("PREDICTIONS_TABLE", "predictions")


def load_scoring_model() -> tuple[Callable[[pd.DataFrame], np.ndarray], str]:
    """Return (score_fn -> P(late) array, model_version). Mirrors the API's resolution.

    Why mirror the API's own model-resolution order here? So the batch path and the online
    path *cannot* score with different models. We try the MLflow registry (the source of
    truth) first, and fall back to a local joblib artifact only if the tracking server is
    unreachable — the same two-tier strategy the FastAPI service uses. The returned
    ``score_fn`` normalises both backends to a single interface: DataFrame -> P(late) array.
    """
    try:
        import mlflow

        # Preferred path: pull the model that's currently in the configured stage
        # (e.g. Staging) straight from the registry, so we always score with whatever
        # was most recently promoted.
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}/{config.MODEL_STAGE}")
        version = f"{config.MODEL_NAME}:{config.MODEL_STAGE}"
        log.info("scoring with MLflow model %s", version)
        # Our registered pyfunc wrapper already returns P(late); ravel() flattens any
        # (n, 1) shape to a plain 1-D array so downstream code sees a consistent vector.
        return (lambda X: np.asarray(model.predict(X), dtype=float).ravel()), version
    except Exception as exc:  # noqa: BLE001 - fall back to the local artifact
        # The registry being down must not stop a scheduled batch job. Fall back to the
        # last known-good model checked into MODEL_PATH. We tag the version string with
        # "joblib:" so the persisted predictions record *which* backend produced them.
        log.warning("MLflow load failed (%s); falling back to joblib", exc)
        import joblib

        pipe = joblib.load(config.MODEL_PATH)
        version = f"joblib:{os.path.basename(config.MODEL_PATH)}"
        log.info("scoring with local joblib model %s", version)
        # A raw sklearn pipeline exposes predict_proba; column 1 is P(positive class)=P(late).
        return (lambda X: pipe.predict_proba(X)[:, 1]), version


def _risk_levels(proba: np.ndarray) -> np.ndarray:
    """Vectorised probability -> {high, medium, low} bucketing.

    Uses the *same* thresholds as the API (from app.config) so a given probability maps
    to the same risk label whether it came from the online or the batch path. np.where is
    a vectorised if/else over the whole array — far faster than a Python loop per row.
    """
    return np.where(
        proba >= config.HIGH_RISK_THRESHOLD, "high",
        np.where(proba >= config.MEDIUM_RISK_THRESHOLD, "medium", "low"),
    )


def run(limit: int | None = None) -> pd.DataFrame:
    """Score the feature table and overwrite the predictions table; return the frame."""
    engine = get_engine()
    # Pull only order_id + the exact model feature columns. Quoting the schema/table names
    # keeps the identifier safe; the optional LIMIT is handy for quick local smoke runs.
    cols = ", ".join(["order_id", *config.FEATURE_COLUMNS])
    src = f'"{FEATURES_SCHEMA}"."{FEATURESET_TABLE}"'
    sql = f"SELECT {cols} FROM {src}" + (f" LIMIT {int(limit)}" if limit else "")
    frame = pd.read_sql(sql, engine)
    log.info("loaded %d rows from %s", len(frame), src)

    # Score every row in one vectorised call, then clip to [0, 1] as a defensive guard
    # (a mis-calibrated model or fallback could in principle return a value slightly out
    # of range; probabilities that leave the table must always be valid).
    score, version = load_scoring_model()
    proba = np.clip(score(frame[config.FEATURE_COLUMNS]), 0.0, 1.0)
    levels = _risk_levels(proba)

    # Assemble the output contract: the same fields the API returns per prediction, plus a
    # scored_at timestamp so monitoring can reason about freshness.
    out = pd.DataFrame({
        "order_id": frame["order_id"].values,
        "late_delivery_probability": np.round(proba, 6),
        "risk_level": levels,
        "recommended_action": [config.RECOMMENDED_ACTIONS[l] for l in levels],
        "model_version": version,
        "scored_at": datetime.now(timezone.utc),
    })
    # Overwrite (if_exists="replace") because this table holds the *current* full-table
    # scoring, not history — each run is a fresh snapshot. chunksize + method="multi"
    # batch the INSERTs so writing tens of thousands of rows stays fast.
    ensure_schema(PREDICTIONS_SCHEMA)
    out.to_sql(PREDICTIONS_TABLE, engine, schema=PREDICTIONS_SCHEMA,
               if_exists="replace", index=False, chunksize=10_000, method="multi")

    # Log a quick sanity summary (risk distribution + mean probability) so a scheduled
    # run leaves an at-a-glance record of whether the output looks reasonable.
    dist = pd.Series(levels).value_counts().to_dict()
    log.info("wrote %d predictions to %s.%s | risk dist=%s | mean P(late)=%.4f",
             len(out), PREDICTIONS_SCHEMA, PREDICTIONS_TABLE, dist, proba.mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score featureset_v1 into a predictions table.")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
