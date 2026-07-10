"""Delivery-risk MLOps pipeline (local Airflow in the compose stack).

Every step runs a pipeline module directly in the Airflow worker as
``python -m pipeline.X``. The worker has the repo mounted at ``/opt/project``
(``PYTHONPATH=/opt/project``) and reads its DB / MLflow / data settings from the
environment supplied by ``docker-compose.yaml`` — the same config the CLI and API use.

Chain (the full ops cycle):
    load_raw_data -> build_features -> train_model -> register_model
      -> api_smoke_test -> batch_predict -> monitor
      -> decide_retrain -> [flag_retrain | no_retrain]
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator

# The repo is mounted here in the Airflow containers (see docker-compose.yaml).
PROJECT_DIR = "/opt/project"
# The pipeline runs in its own virtualenv (isolated from Airflow's own dependency
# pins — notably SQLAlchemy). docker-compose.yaml points PIPELINE_PYTHON at it.
PIPELINE_PYTHON = os.getenv("PIPELINE_PYTHON", "python")


def step(module: str) -> str:
    """Bash command that runs a pipeline module in the project directory."""
    return f"{PIPELINE_PYTHON} -m {module}"


def _decide_retrain(**context) -> str:
    """Branch on the monitor step's retrain flag (its last stdout line -> XCom).

    Defaults to 'no_retrain' if the flag is missing/unreadable, so an XCom hiccup
    never forces an unwanted retrain path.
    """
    flag = (context["ti"].xcom_pull(task_ids="monitor") or "").strip().lower()
    return "flag_retrain" if flag.endswith("true") else "no_retrain"


# Defaults applied to every task unless overridden. retries=0 keeps a failing step
# failing fast and visibly rather than masking a real problem behind auto-retries;
# retry_delay is set anyway so it's easy to flip retries on.
default_args = {
    "owner": "delivery-risk",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

# schedule=None + catchup=False => this DAG only runs when triggered manually (from the
# UI or CLI). It's an on-demand operational pipeline, not a cron job, so we don't want
# Airflow backfilling missed runs from start_date to now.
with DAG(
    dag_id="delivery_risk_pipeline",
    description="Olist late-delivery MLOps pipeline: load -> features -> train -> register -> predict -> monitor.",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["delivery-risk", "mlops"],
) as dag:

    # Each step runs `python -m pipeline.X` in the mounted project directory. The worker
    # does the data work in-process; there is no remote bridge.
    load_raw_data = BashOperator(
        task_id="load_raw_data",
        bash_command=step("pipeline.load_raw"),
        cwd=PROJECT_DIR,
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=step("pipeline.features"),
        cwd=PROJECT_DIR,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=step("pipeline.train"),
        cwd=PROJECT_DIR,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=step("pipeline.register"),
        cwd=PROJECT_DIR,
    )

    api_smoke_test = BashOperator(
        task_id="api_smoke_test",
        bash_command=step("pipeline.smoke_test"),
        cwd=PROJECT_DIR,
    )

    batch_predict = BashOperator(
        task_id="batch_predict",
        bash_command=step("pipeline.batch_predict"),
        cwd=PROJECT_DIR,
    )

    # monitor prints the retrain flag as its last stdout line -> captured as XCom.
    monitor = BashOperator(
        task_id="monitor",
        bash_command=step("pipeline.monitor"),
        cwd=PROJECT_DIR,
        do_xcom_push=True,
    )

    # A branch operator returns the task_id of whichever downstream branch to run; the
    # other branch is skipped. This is how the monitor verdict controls the DAG's shape.
    decide_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=_decide_retrain,
    )

    # The two branches are deliberately just echoes: this pipeline *surfaces* the retrain
    # decision, it doesn't auto-retrain (that would retrain on the data we just trained on).
    # Retraining stays a human-initiated action; these tasks make the verdict visible.
    flag_retrain = BashOperator(
        task_id="flag_retrain",
        bash_command="echo 'drift/decay detected -> retrain recommended'",
    )

    no_retrain = BashOperator(
        task_id="no_retrain",
        bash_command="echo 'within thresholds -> no retrain needed'",
    )

    # The `>>` operator wires task dependencies (left runs before right). The final
    # `[flag_retrain, no_retrain]` is the fan-out the branch operator chooses between.
    (
        load_raw_data
        >> build_features
        >> train_model
        >> register_model
        >> api_smoke_test
        >> batch_predict
        >> monitor
        >> decide_retrain
        >> [flag_retrain, no_retrain]
    )
