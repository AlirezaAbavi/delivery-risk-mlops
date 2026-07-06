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
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"no registered versions for {MODEL_NAME!r}; run pipeline.train first")

    latest = max(versions, key=lambda v: int(v.version))
    if latest.current_stage != MODEL_STAGE:
        client.transition_model_version_stage(
            MODEL_NAME, latest.version, stage=MODEL_STAGE, archive_existing_versions=True,
        )
        log.info("promoted %s v%s -> %s", MODEL_NAME, latest.version, MODEL_STAGE)
    else:
        log.info("%s v%s already in %s (no-op)", MODEL_NAME, latest.version, MODEL_STAGE)

    log.info("register_model OK: %s v%s in %s", MODEL_NAME, latest.version, MODEL_STAGE)
    return latest.version


def main() -> None:
    run()


if __name__ == "__main__":
    main()
