"""Model loading for the delivery-risk API.

Resolution order (never hard-fails): MLflow registry (``models:/{name}/{stage}``)
-> local joblib (``MODEL_PATH``) -> arithmetic baseline. Load is lock-protected and
runs in a background thread at startup so a slow/unreachable MLflow can't block Swagger.
"""
from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, List, Optional
from urllib.parse import urlparse

from . import config

log = logging.getLogger("model_loader")


@dataclass
class LoadedModelState:
    """Current model state. ``source`` is 'loading' | 'mlflow' | 'joblib' | 'baseline'."""

    source: str = "loading"
    model: Any = None
    name: str = config.BASELINE_VERSION
    version: Optional[str] = None
    stage: Optional[str] = None
    feature_names: Optional[List[str]] = None
    error: Optional[str] = None

    @property
    def is_real(self) -> bool:
        """True when a trained model is serving (not the arithmetic fallback)."""
        return self.model is not None

    @property
    def version_string(self) -> str:
        return f"{self.name}:{self.version}" if self.version else self.name

    def info(self) -> dict:
        return {
            "source": self.source,
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "n_features": len(self.feature_names) if self.feature_names else None,
            "is_real_model": self.is_real,
            "load_error": self.error,
        }


def _feature_names(model) -> Optional[List[str]]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    try:  # mlflow pyfunc: read the logged input schema if present
        schema = model.metadata.get_input_schema()
        if schema is not None:
            return list(schema.input_names())
    except Exception:  # noqa: BLE001 - schema is best-effort metadata
        pass
    return None


def _mlflow_reachable() -> bool:
    """Fast TCP probe so an unreachable tracking server can't stall startup."""
    parsed = urlparse(config.MLFLOW_TRACKING_URI)
    if parsed.scheme not in ("http", "https"):
        return True  # file:/local store — let MLflow handle it
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=config.MLFLOW_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


class ModelService:
    def __init__(self) -> None:
        self.state = LoadedModelState()
        self._load_lock = Lock()
        self._load_started = False

    # --- individual sources -------------------------------------------------
    def _try_mlflow(self) -> Optional[LoadedModelState]:
        if not _mlflow_reachable():
            log.info("MLflow unreachable at %s; skipping", config.MLFLOW_TRACKING_URI)
            return LoadedModelState(source="mlflow", error=f"unreachable: {config.MLFLOW_TRACKING_URI}")
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
            versions = client.get_latest_versions(config.MODEL_NAME, [config.MODEL_STAGE])
            if not versions:
                return LoadedModelState(source="mlflow", error=f"no '{config.MODEL_STAGE}' version for '{config.MODEL_NAME}'")
            mv = versions[0]
            model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}/{config.MODEL_STAGE}")
            log.info("Loaded MLflow model %s v%s (%s)", config.MODEL_NAME, mv.version, config.MODEL_STAGE)
            return LoadedModelState(
                source="mlflow", model=model, name=config.MODEL_NAME, version=str(mv.version),
                stage=config.MODEL_STAGE, feature_names=_feature_names(model),
            )
        except Exception as exc:  # noqa: BLE001 - any MLflow failure -> fall through
            log.warning("MLflow load failed: %s", exc)
            return LoadedModelState(source="mlflow", error=f"{type(exc).__name__}: {exc}")

    def _try_joblib(self, prev_error: Optional[str]) -> Optional[LoadedModelState]:
        if not config.MODEL_PATH:
            return None
        try:
            import os
            import joblib

            model = joblib.load(config.MODEL_PATH)
            log.info("Loaded local joblib model from %s", config.MODEL_PATH)
            return LoadedModelState(
                source="joblib", model=model, name=os.path.basename(config.MODEL_PATH),
                version="local", feature_names=_feature_names(model), error=prev_error,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("joblib load failed: %s", exc)
            return LoadedModelState(source="joblib", error=f"{prev_error or ''} | joblib: {exc}".strip(" |"))

    # --- orchestration ------------------------------------------------------
    def load(self) -> None:
        """Resolve the best available model; always leaves a usable state."""
        with self._load_lock:
            self._load_started = True
            mlflow_state = self._try_mlflow()
            if mlflow_state and mlflow_state.is_real:
                self.state = mlflow_state
                return

            carried = mlflow_state.error if mlflow_state else None
            joblib_state = self._try_joblib(carried)
            if joblib_state and joblib_state.is_real:
                self.state = joblib_state
                return

            error = joblib_state.error if joblib_state else carried
            log.info("Serving arithmetic baseline (no trained model available)")
            self.state = LoadedModelState(source="baseline", name=config.BASELINE_VERSION, error=error)

    def ensure_loaded(self) -> None:
        """Trigger a load if one was never started (e.g. startup load disabled).

        If a background load is already underway we don't kick off a second one;
        requests arriving mid-load are served by the baseline until it completes.
        """
        if not self._load_started:
            self.load()

    def model_info(self) -> dict:
        info = self.state.info()
        info["temporal_leakage_policy"] = "purchase-time features only"
        return info
