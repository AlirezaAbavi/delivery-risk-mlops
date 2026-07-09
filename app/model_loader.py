"""Model loading for the delivery-risk API.

The whole point of this module is a *resilient* model-resolution chain that never
hard-fails the service:

    MLflow registry (models:/{name}/{stage})   # the production path
        -> local joblib file (MODEL_PATH)       # offline / air-gapped fallback
            -> arithmetic baseline              # last resort, always available

Why this matters for MLOps: the API's job is to stay up and answer. If the tracking
server is down or empty, we would rather serve a clearly-labelled degraded answer
than refuse to start. Loading is guarded by a Lock (so two requests can't race a
double-load) and a fast TCP pre-check (so an unreachable MLflow can't stall startup).
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
    """An immutable-ish snapshot of "what model is serving right now".

    Bundling all model facts into one object (instead of loose globals) means the
    rest of the app reads a single, consistent view. ``source`` is the resolution
    outcome: 'loading' | 'mlflow' | 'joblib' | 'baseline'. When a load fails partway,
    we still return a state object carrying the ``error`` string so /health can
    explain what went wrong.
    """

    source: str = "loading"
    model: Any = None                             # the callable model object, or None for baseline
    name: str = config.BASELINE_VERSION
    version: Optional[str] = None
    stage: Optional[str] = None
    feature_names: Optional[List[str]] = None     # the model's own expected input columns, if known
    error: Optional[str] = None

    @property
    def is_real(self) -> bool:
        """True when a trained model is serving (not the arithmetic fallback).

        The whole service keys off this: predictor uses the model iff ``is_real``,
        and the delivery_model_loaded gauge is set from it.
        """
        return self.model is not None

    @property
    def version_string(self) -> str:
        """Human-readable id shown in responses/logs: 'name:version' or just 'name'."""
        return f"{self.name}:{self.version}" if self.version else self.name

    def info(self) -> dict:
        """Flatten the state into the dict /model-info serialises."""
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
    """Best-effort discovery of the model's expected input columns.

    Two sources, in order: sklearn estimators expose ``feature_names_in_`` after
    fitting on a named DataFrame; MLflow pyfunc models can carry a logged input
    schema. If neither is present we return None and the scorer falls back to the
    config feature contract. Wrapped in try/except because schema access is
    best-effort metadata that may be absent.
    """
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
    """Fast TCP probe so an unreachable tracking server can't stall startup.

    Rather than let MLflow's HTTP client discover the server is down (slow, retries),
    we open a raw socket to host:port with a short timeout first. If the connection
    can't be made we skip MLflow entirely. For a ``file:`` store there is no host to
    probe, so we optimistically return True and let MLflow handle it.
    """
    parsed = urlparse(config.MLFLOW_TRACKING_URI)
    if parsed.scheme not in ("http", "https"):
        return True  # file:/local store — let MLflow handle it
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        # ``create_connection`` succeeds only if something is actually listening.
        with socket.create_connection((host, port), timeout=config.MLFLOW_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


class ModelService:
    """Owns the current model state and the logic to (re)load it.

    A single instance lives for the process lifetime (created in main.py). It holds
    a Lock so concurrent requests calling ``ensure_loaded`` can't trigger overlapping
    loads, and a ``_load_started`` flag so we only auto-load once.
    """

    def __init__(self) -> None:
        self.state = LoadedModelState()  # starts in the 'loading' state
        self._load_lock = Lock()
        self._load_started = False

    # --- individual sources -------------------------------------------------
    def _try_mlflow(self) -> Optional[LoadedModelState]:
        """Attempt to load models:/{name}/{stage} from the MLflow registry.

        Returns a *real* state on success, or a state carrying only an ``error`` on
        any failure (unreachable / no such stage / load exception). Returning the
        error rather than raising lets the caller fall through to the next source
        while preserving the reason for /health.
        """
        # Cheap reachability gate first, so a down server fails in ~1.5s not ~30s.
        if not _mlflow_reachable():
            log.info("MLflow unreachable at %s; skipping", config.MLFLOW_TRACKING_URI)
            return LoadedModelState(source="mlflow", error=f"unreachable: {config.MLFLOW_TRACKING_URI}")
        try:
            # Import inside the function so the service doesn't hard-depend on mlflow
            # being importable just to start (keeps the import cost off the hot path).
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
            # Ask the registry for the version currently aliased to our stage (Staging).
            versions = client.get_latest_versions(config.MODEL_NAME, [config.MODEL_STAGE])
            if not versions:
                # The model exists but nothing is promoted to this stage yet.
                return LoadedModelState(source="mlflow", error=f"no '{config.MODEL_STAGE}' version for '{config.MODEL_NAME}'")
            mv = versions[0]
            # Load via the stage alias (not a pinned version) so promoting a new
            # version in the registry is picked up on the next restart with no code change.
            model = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}/{config.MODEL_STAGE}")
            log.info("Loaded MLflow model %s v%s (%s)", config.MODEL_NAME, mv.version, config.MODEL_STAGE)
            return LoadedModelState(
                source="mlflow", model=model, name=config.MODEL_NAME, version=str(mv.version),
                stage=config.MODEL_STAGE, feature_names=_feature_names(model),
            )
        except Exception as exc:  # noqa: BLE001 - any MLflow failure -> fall through
            # Catch broadly on purpose: whatever went wrong, we want to try joblib next
            # rather than crash. The exception text is preserved for observability.
            log.warning("MLflow load failed: %s", exc)
            return LoadedModelState(source="mlflow", error=f"{type(exc).__name__}: {exc}")

    def _try_joblib(self, prev_error: Optional[str]) -> Optional[LoadedModelState]:
        """Attempt to load a local joblib model from MODEL_PATH (the offline fallback).

        ``prev_error`` is threaded through so that if joblib *also* fails, /health can
        show both the MLflow and joblib reasons chained together. Returns None if no
        MODEL_PATH is configured (that fallback is simply disabled).
        """
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
            # Chain the previous (MLflow) error with this one; strip stray separators.
            log.warning("joblib load failed: %s", exc)
            return LoadedModelState(source="joblib", error=f"{prev_error or ''} | joblib: {exc}".strip(" |"))

    # --- orchestration ------------------------------------------------------
    def load(self) -> None:
        """Resolve the best available model; always leaves a usable state.

        Tries each source in priority order and stops at the first that yields a real
        model. If none do, it installs the arithmetic baseline (carrying whatever
        error explains why no real model loaded). The Lock makes the whole resolution
        atomic so two threads can't interleave and leave a half-built state.
        """
        with self._load_lock:
            self._load_started = True
            # 1) Preferred: the registry.
            mlflow_state = self._try_mlflow()
            if mlflow_state and mlflow_state.is_real:
                self.state = mlflow_state
                return

            # 2) Fallback: a local joblib file, carrying the MLflow error forward.
            carried = mlflow_state.error if mlflow_state else None
            joblib_state = self._try_joblib(carried)
            if joblib_state and joblib_state.is_real:
                self.state = joblib_state
                return

            # 3) Last resort: the always-available arithmetic baseline.
            error = joblib_state.error if joblib_state else carried
            log.info("Serving arithmetic baseline (no trained model available)")
            self.state = LoadedModelState(source="baseline", name=config.BASELINE_VERSION, error=error)

    def ensure_loaded(self) -> None:
        """Trigger a load if one was never started (e.g. startup load disabled).

        If a background load is already underway we don't kick off a second one;
        requests arriving mid-load are served by the baseline until it completes.
        This is what lets /predict work even when LOAD_MODEL_ON_STARTUP is off.
        """
        if not self._load_started:
            self.load()

    def model_info(self) -> dict:
        """State dict for /model-info, plus the advertised leakage policy string."""
        info = self.state.info()
        info["temporal_leakage_policy"] = "purchase-time features only"
        return info
