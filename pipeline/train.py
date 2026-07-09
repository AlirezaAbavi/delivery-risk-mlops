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

# Three of the features are categorical (text codes); the rest are numeric. They need
# different preprocessing (one-hot vs. scale/impute), so we split the column list here.
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
    """Read the materialised featureset table back into a DataFrame for training."""
    frame = pd.read_sql_table(FEATURESET_TABLE, get_engine(), schema=FEATURES_SCHEMA)
    # purchase_ts arrives as text/naive; parse it so the temporal split can compare dates.
    frame["purchase_ts"] = pd.to_datetime(frame["purchase_ts"])
    # Keep is_multi_category boolean so the logged model signature matches the API's
    # bool field (schemas.PredictionInput); sklearn's numeric steps handle bool as 0/1.
    return frame


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by TIME, not randomly: earliest 80% of purchases train, newest 20% validate.

    Why this matters (a core temporal-validity idea): the late-delivery rate drifts over
    time. A random split would let the model peek at future patterns while validating on
    the past — an optimistic, dishonest estimate. Splitting on the purchase timestamp
    mimics production ("train on what we knew, predict what came next"), so the reported
    metrics are trustworthy. The split point is the 80th-percentile timestamp by default,
    or a fixed date if TEMPORAL_SPLIT_DATE is set.
    """
    setting = os.getenv("TEMPORAL_SPLIT_DATE", "auto").strip().lower()
    if setting in ("auto", ""):
        q = float(os.getenv("TEMPORAL_SPLIT_QUANTILE", "0.8"))
        split_ts = frame["purchase_ts"].quantile(q)  # the timestamp below which 80% of orders fall
    else:
        split_ts = pd.Timestamp(setting)
    train = frame[frame["purchase_ts"] < split_ts]   # strictly before the cut => the past
    valid = frame[frame["purchase_ts"] >= split_ts]  # on/after the cut => the "future"
    log.info("temporal split at %s -> train=%d valid=%d", split_ts, len(train), len(valid))
    return train, valid, split_ts


# ── model builders ──────────────────────────────────────────────────────────
# Each builder returns a full sklearn Pipeline = preprocessing + estimator in one object.
# Bundling preprocessing INTO the model is deliberate: the exact same imputation/encoding
# is applied at train time and at serve time, so there's no train/serve skew, and the API
# can hand a raw feature row straight to the pipeline.
def _logreg() -> Pipeline:
    """Candidate 1: Logistic Regression — a fast, linear, highly interpretable baseline.

    Numerics are median-imputed then standardised (linear models need comparable scales);
    categoricals are one-hot encoded. ``class_weight="balanced"`` up-weights the rare
    late class so the model doesn't just predict "on time" for everyone.
    """
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_IDX),
        # min_frequency=20 folds rare categories into an "infrequent" bucket to avoid a
        # blow-up of one-hot columns; handle_unknown="ignore" tolerates unseen categories
        # at serve time (an unfamiliar seller state won't crash scoring).
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_IDX),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def _random_forest() -> Pipeline:
    """Candidate 2: Random Forest — a strong non-linear bagging ensemble.

    Trees don't need feature scaling, so numerics are only imputed (no StandardScaler).
    ``min_samples_leaf=20`` regularises (prevents tiny overfit leaves);
    ``balanced_subsample`` rebalances the class weights within each bootstrap sample.
    """
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
    """Candidate 3: Histogram Gradient Boosting — the eventual winner.

    Boosting builds trees sequentially, each correcting the last; it's typically the
    strongest tabular model. It has no ``class_weight`` param, so imbalance is handled by
    per-sample weights passed at fit time (see ``run``). One-hot output must be dense here
    because HistGB rejects sparse matrices.
    """
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
    """Score a fitted pipeline on the validation set and return a metric dict.

    Metric choices (this is the "which metrics for model selection" story):
      - pr_auc (average precision): the PRIMARY selection metric. With only ~8% positives,
        ROC-AUC can look flatteringly high; PR-AUC focuses on how well we rank the rare
        late orders and is far more honest for imbalanced problems.
      - roc_auc: reported for comparability / intuition, but NOT the decider.
      - brier: calibration — are the probabilities themselves trustworthy (lower better)?
      - f1_at_0.5: a single point on the precision/recall trade-off at the 0.5 threshold.
      - val_positive_rate: the late rate in the validation window (drift context).
    ``[:, 1]`` selects the positive-class (late) probability column.
    """
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
    """A thin MLflow ``pyfunc`` wrapper so ``predict`` returns P(late), not a 0/1 label.

    Why wrap at all? A raw sklearn classifier's ``predict`` returns the hard class, but the
    API contract is a *probability*. By wrapping the fitted pipeline and overriding
    ``predict`` to call ``predict_proba(...)[:, 1]``, we guarantee that however the model is
    loaded (MLflow pyfunc OR the joblib fallback), the service receives a probability. This
    is the "serve a stable interface, hide the implementation" idea.
    """

    def load_context(self, context):
        # Called once when MLflow loads the model: restore the fitted sklearn pipeline
        # from the artifact that was logged alongside this wrapper.
        self._pipe = joblib.load(context.artifacts["sk_pipeline"])

    def predict(self, context, model_input):
        # The public inference method: always hand back the positive-class probability.
        return self._pipe.predict_proba(model_input)[:, 1]


def _register_staging(pipe: Pipeline, X_example: pd.DataFrame, metrics: dict, name: str) -> str:
    """Log the winning pipeline as a probability pyfunc, register it, and promote to Staging.

    This is the "model handoff" from experimentation to serving:
      1. Dump the fitted pipeline to a temp file and log it as an artifact of a pyfunc model.
      2. ``infer_signature`` records the expected input/output schema so the registry can
         validate future inputs and the API can read ``feature_names``.
      3. ``registered_model_name`` creates/appends a new *version* under that name — this is
         why MLflow accumulates many versions over time (each retrain adds one).
      4. Transition the new version to the ``Staging`` stage and archive the previous one, so
         the API (which loads ``models:/{name}/Staging``) picks it up on next restart.
    Returns the new version number.
    """
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
    """Full training run: load -> temporal split -> train 3 candidates -> pick winner by
    PR-AUC -> register+promote to Staging -> write the joblib fallback.

    Every candidate is logged as its own MLflow run under the experiment, so the registry
    keeps a full, comparable record of what was tried (not just the winner). This is what
    makes the model choice *defensible* — the evidence lives in MLflow.
    """
    # Point MLflow at our tracking server and experiment (creates it if new).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    frame = load_featureset()
    train, valid, split_ts = temporal_split(frame)
    # X = features (contract-ordered), y = the target column, for each split.
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_valid, y_valid = valid[FEATURE_COLUMNS], valid[TARGET]

    results: dict[str, dict] = {}   # name -> validation metrics
    fitted: dict[str, Pipeline] = {}  # name -> the trained pipeline (kept for the winner)
    for name, build in BUILDERS.items():
        # One MLflow run per candidate: params + metrics + the model artifact are grouped
        # under this run so they can be compared side-by-side in the MLflow UI.
        with mlflow.start_run(run_name=name):
            pipe = build()
            # HistGB has no class_weight param, so we balance the rare late class by passing
            # per-sample weights straight into its fit step (``clf__`` targets the pipeline's
            # clf stage). The other two models balance via their class_weight setting.
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

    # Model selection: highest PR-AUC wins (the imbalance-aware metric chosen in evaluate).
    winner = max(results, key=lambda k: results[k]["pr_auc"])
    log.info("WINNER by PR-AUC: %s (%.4f)", winner, results[winner]["pr_auc"])

    # Promote the winner into the registry's Staging stage (a 4th MLflow run).
    version = _register_staging(fitted[winner], X_valid, results[winner], winner)

    # Also dump the raw winning pipeline to disk. The API's local-joblib fallback loads
    # this if MLflow is ever unreachable — same model, no registry dependency. The raw
    # pipeline exposes predict_proba directly, so the API's scorer handles it natively.
    fallback = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    os.makedirs(os.path.dirname(fallback) or ".", exist_ok=True)
    joblib.dump(fitted[winner], fallback)
    log.info("wrote local fallback %s (winner=%s, staging v%s)", fallback, winner, version)


if __name__ == "__main__":
    run()
