"""Idempotent model registration / promotion — the DAG's ``register_model`` task.

``pipeline.train`` already logs three candidates, registers the PR-AUC winner as
``delivery-risk``, and promotes it to Staging. This step is a separate,
idempotent gate: it verifies a registered version exists and ensures the newest
version sits in the configured stage. So ``register_model`` is a meaningful,
re-runnable task that fails loudly if training ever produced no model.

Usage:
    python -m pipeline.register
"""
from __future__ import annotations

import logging
import os

from mlflow.tracking import MlflowClient

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline.register")

MODEL_NAME = os.getenv("MODEL_NAME", "delivery-risk")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5312")


def run() -> str:
    """Verify a registered model exists and make sure its newest version is in-stage.

    This is intentionally idempotent: running it twice is harmless. That matters in an
    Airflow context where a task can be retried — the second run just observes the model
    is already promoted and no-ops, rather than doing something different.
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)
    # Fail loudly (non-zero exit -> failed Airflow task) if training never produced a
    # registered model. A silent success here would let a broken pipeline appear healthy.
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"no registered versions for {MODEL_NAME!r}; run pipeline.train first")

    # Registry versions are strings ("1", "2", ...); compare as ints so "10" > "9".
    latest = max(versions, key=lambda v: int(v.version))
    if latest.current_stage != MODEL_STAGE:
        # Promote the newest version and archive whatever used to hold the stage, so there
        # is exactly one model in Staging at a time (no ambiguity about what serves).
        client.transition_model_version_stage(
            MODEL_NAME, latest.version, stage=MODEL_STAGE, archive_existing_versions=True,
        )
        log.info("promoted %s v%s -> %s", MODEL_NAME, latest.version, MODEL_STAGE)
    else:
        # Already in the right stage (e.g. train.py promoted it) — nothing to do.
        log.info("%s v%s already in %s (no-op)", MODEL_NAME, latest.version, MODEL_STAGE)

    log.info("register_model OK: %s v%s in %s", MODEL_NAME, latest.version, MODEL_STAGE)
    return latest.version


def main() -> None:
    run()


if __name__ == "__main__":
    main()
