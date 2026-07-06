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
    """Return (score_fn -> P(late) array, model_version). Mirrors the API's resolution."""
    try:
        import mlflow

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}/{config.MODEL_STAGE}")
        version = f"{config.MODEL_NAME}:{config.MODEL_STAGE}"
        log.info("scoring with MLflow model %s", version)
        # Our registered pyfunc wrapper already returns P(late).
        return (lambda X: np.asarray(model.predict(X), dtype=float).ravel()), version
    except Exception as exc:  # noqa: BLE001 - fall back to the local artifact
        log.warning("MLflow load failed (%s); falling back to joblib", exc)
        import joblib

        pipe = joblib.load(config.MODEL_PATH)
        version = f"joblib:{os.path.basename(config.MODEL_PATH)}"
        log.info("scoring with local joblib model %s", version)
        return (lambda X: pipe.predict_proba(X)[:, 1]), version


def _risk_levels(proba: np.ndarray) -> np.ndarray:
    return np.where(
        proba >= config.HIGH_RISK_THRESHOLD, "high",
        np.where(proba >= config.MEDIUM_RISK_THRESHOLD, "medium", "low"),
    )


def run(limit: int | None = None) -> pd.DataFrame:
    engine = get_engine()
    cols = ", ".join(["order_id", *config.FEATURE_COLUMNS])
    src = f'"{FEATURES_SCHEMA}"."{FEATURESET_TABLE}"'
    sql = f"SELECT {cols} FROM {src}" + (f" LIMIT {int(limit)}" if limit else "")
    frame = pd.read_sql(sql, engine)
    log.info("loaded %d rows from %s", len(frame), src)

    score, version = load_scoring_model()
    proba = np.clip(score(frame[config.FEATURE_COLUMNS]), 0.0, 1.0)
    levels = _risk_levels(proba)

    out = pd.DataFrame({
        "order_id": frame["order_id"].values,
        "late_delivery_probability": np.round(proba, 6),
        "risk_level": levels,
        "recommended_action": [config.RECOMMENDED_ACTIONS[l] for l in levels],
        "model_version": version,
        "scored_at": datetime.now(timezone.utc),
    })
    ensure_schema(PREDICTIONS_SCHEMA)
    out.to_sql(PREDICTIONS_TABLE, engine, schema=PREDICTIONS_SCHEMA,
               if_exists="replace", index=False, chunksize=10_000, method="multi")

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
