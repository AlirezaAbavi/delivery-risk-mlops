"""Train, compare, and register the late-delivery model.

Three credible candidates (LogisticRegression, RandomForest, HistGradientBoosting)
are evaluated on a strict *temporal* validation split (train the earliest 80% of
purchases, validate the newest 20%) so metrics respect the observed late-rate drift
and never see the future. All three are logged to MLflow; the best by PR-AUC
(average precision — the right metric for an 8% positive rate) is registered as
``delivery-risk`` and promoted to Staging.

The registered model is a pyfunc wrapper whose ``predict`` returns P(late) so the
API gets a probability regardless of load path. The same fitted pipeline is also
dumped to ``artifacts/model.joblib`` for the API's local-joblib fallback.

Usage:
    python -m pipeline.train
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Callable

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.config import FEATURE_COLUMNS
from .db import FEATURES_SCHEMA, get_engine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline.train")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5312")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "delivery-risk")
MODEL_NAME = os.getenv("MODEL_NAME", "delivery-risk")
FEATURESET_TABLE = os.getenv("FEATURESET_TABLE", "featureset_v1")
SEED = int(os.getenv("RANDOM_SEED", "42"))
TARGET = "is_late_delivery"

CATEGORICAL = ["payment_type_mode", "product_category_mode", "seller_state_mode"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]

# ColumnTransformer selects columns by *integer position* (into a frame ordered as
# FEATURE_COLUMNS), not by name. sklearn 1.8 has a regression where selecting by name
# makes fit read ``feature_names_in_`` before it is set, raising AttributeError; the
# integer path sidesteps it and behaves identically on the host's sklearn 1.5.
NUMERIC_IDX = [FEATURE_COLUMNS.index(c) for c in NUMERIC]
CATEGORICAL_IDX = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL]


# ── data / split ────────────────────────────────────────────────────────────
def load_featureset() -> pd.DataFrame:
    frame = pd.read_sql_table(FEATURESET_TABLE, get_engine(), schema=FEATURES_SCHEMA)
    frame["purchase_ts"] = pd.to_datetime(frame["purchase_ts"])
    # Keep is_multi_category boolean so the logged model signature matches the API's
    # bool field (schemas.PredictionInput); sklearn's numeric steps handle bool as 0/1.
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    setting = os.getenv("TEMPORAL_SPLIT_DATE", "auto").strip().lower()
    if setting in ("auto", ""):
        q = float(os.getenv("TEMPORAL_SPLIT_QUANTILE", "0.8"))
        split_ts = frame["purchase_ts"].quantile(q)
    else:
        split_ts = pd.Timestamp(setting)
    train = frame[frame["purchase_ts"] < split_ts]
    valid = frame[frame["purchase_ts"] >= split_ts]
    log.info("temporal split at %s -> train=%d valid=%d", split_ts, len(train), len(valid))
    return train, valid, split_ts


# ── model builders ──────────────────────────────────────────────────────────
def _logreg() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_IDX),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_IDX),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def _random_forest() -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_IDX),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_IDX),
    ])
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=20,
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _hist_gb() -> Pipeline:
    # Median-impute numerics (mirrors the RF path); class imbalance is handled via
    # per-sample weights at fit time. One-hot is dense because HistGB rejects sparse input.
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_IDX),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAL_IDX),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=None,
        l2_regularization=1.0, random_state=SEED,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


BUILDERS: dict[str, Callable[[], Pipeline]] = {
    "logistic_regression": _logreg,
    "random_forest": _random_forest,
    "hist_gradient_boosting": _hist_gb,
}


# ── evaluation ──────────────────────────────────────────────────────────────
def evaluate(pipe: Pipeline, X, y) -> dict[str, float]:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "val_positive_rate": float(y.mean()),
    }


class ProbaModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper: predict returns P(late) so the API always gets a probability."""

    def load_context(self, context):
        self._pipe = joblib.load(context.artifacts["sk_pipeline"])

    def predict(self, context, model_input):
        return self._pipe.predict_proba(model_input)[:, 1]


def _register_staging(pipe: Pipeline, X_example: pd.DataFrame, metrics: dict, name: str) -> str:
    """Log the winner as a probability pyfunc, register it, and promote to Staging."""
    with tempfile.TemporaryDirectory() as tmp:
        sk_path = os.path.join(tmp, "pipeline.joblib")
        joblib.dump(pipe, sk_path)
        signature = infer_signature(X_example, pipe.predict_proba(X_example)[:, 1])
        with mlflow.start_run(run_name="register_staging"):
            mlflow.log_metrics({f"winner_{k}": v for k, v in metrics.items()})
            mlflow.set_tag("winner_estimator", name)
            info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=ProbaModel(),
                artifacts={"sk_pipeline": sk_path},
                signature=signature,
                input_example=X_example.head(3),
                registered_model_name=MODEL_NAME,
            )
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version
    client.transition_model_version_stage(MODEL_NAME, version, stage="Staging", archive_existing_versions=True)
    log.info("registered %s v%s and promoted to Staging", MODEL_NAME, version)
    return version


# ── main ────────────────────────────────────────────────────────────────────
def run() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    frame = load_featureset()
    train, valid, split_ts = temporal_split(frame)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_valid, y_valid = valid[FEATURE_COLUMNS], valid[TARGET]

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, build in BUILDERS.items():
        with mlflow.start_run(run_name=name):
            pipe = build()
            # HistGB has no class_weight; balance it with per-sample weights.
            if name == "hist_gradient_boosting":
                sw = compute_sample_weight("balanced", y_train)
                pipe.fit(X_train, y_train, clf__sample_weight=sw)
            else:
                pipe.fit(X_train, y_train)

            metrics = evaluate(pipe, X_valid, y_valid)
            mlflow.log_params({"estimator": name, "n_train": len(train), "n_valid": len(valid),
                               "split_ts": str(split_ts), "seed": SEED})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="sklearn_model")
            results[name] = metrics
            fitted[name] = pipe
            log.info("%-24s pr_auc=%.4f roc_auc=%.4f brier=%.4f",
                     name, metrics["pr_auc"], metrics["roc_auc"], metrics["brier"])

    winner = max(results, key=lambda k: results[k]["pr_auc"])
    log.info("WINNER by PR-AUC: %s (%.4f)", winner, results[winner]["pr_auc"])

    version = _register_staging(fitted[winner], X_valid, results[winner], winner)

    # Local joblib fallback for the API (raw pipeline exposes predict_proba).
    fallback = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    os.makedirs(os.path.dirname(fallback) or ".", exist_ok=True)
    joblib.dump(fitted[winner], fallback)
    log.info("wrote local fallback %s (winner=%s, staging v%s)", fallback, winner, version)


if __name__ == "__main__":
    run()
