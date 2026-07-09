"""MLOps Delivery — delivery-risk MLOps pipeline (course Airflow).

Deployed by git-sync: the course Airflow pulls this repo's ``airflow/dags/*.py``
into ``/opt/airflow/dags/delivery-capstone/delivery/``. Because the Airflow worker has
neither our code, the Olist CSVs, nor our MLflow, every real step executes ON the
group VM through the course SSH bridge (``delivery_group_ssh.sh``); the worker only
orchestrates and branches. This mirrors the proven benchmark-team pattern.

Isolation/security (graded): NO credentials in this file. Steps read the VM's
``~/project/.env`` (DB, schema layout, MLflow) via ``load_dotenv`` — the same
config the CLI and API use. The VM ``.env`` must set the medallion schemas:
    RAW_SCHEMA=raw  STAGING_SCHEMA=staging  FEATURES_SCHEMA=features
    PREDICTIONS_SCHEMA=predictions  MONITORING_SCHEMA=monitoring
    RAW_DATA_DIR=/home/delivery/olist_data

Chain (covers the brief's required tasks + the full ops cycle):
    load_raw_data -> build_features -> train_model -> register_model
      -> api_smoke_test -> batch_predict -> monitor
      -> decide_retrain -> [flag_retrain | no_retrain]
"""
from __future__ import annotations

import shlex
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator

# --- server deployment settings (no secrets) ------------------------------
GROUP = "delivery"
PROJECT_DIR = f"/home/{GROUP}/project"
PYTHON = f"{PROJECT_DIR}/.venv/bin/python"
# SSH bridge mounted in the Airflow container; forwards a command to the VM host.
BRIDGE = "/opt/airflow/plugins/delivery_group_ssh.sh"


def bridge(command: str) -> str:
    """Wrap a shell command so it runs on the group VM via the SSH bridge."""
    inner = f"cd {PROJECT_DIR} && {command}"
    return f"{BRIDGE} {shlex.quote(GROUP)} {shlex.quote(inner)}"


def step(module: str) -> str:
    """Bridge command that runs a pipeline module on the VM's project venv."""
    return bridge(f"{PYTHON} -m {module}")


def _decide_retrain(**context) -> str:
    """Branch on the monitor step's retrain flag (its last stdout line -> XCom).

    Defaults to 'no_retrain' if the flag is missing/unreadable, so an XCom hiccup
    never forces an unwanted retrain path.
    """
    flag = (context["ti"].xcom_pull(task_ids="monitor") or "").strip().lower()
    return "flag_retrain" if flag.endswith("true") else "no_retrain"


# Defaults applied to every task unless overridden. retries=0 keeps a failing step
# failing fast and visibly (rather than masking a real problem behind auto-retries)
# during a graded demo; retry_delay is set anyway so it's easy to flip retries on.
default_args = {
    "owner": GROUP,
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

# schedule=None + catchup=False => this DAG only runs when triggered manually (from the
# UI or CLI). It's an on-demand operational pipeline for the demo, not a cron job, so we
# don't want Airflow backfilling missed runs from start_date to now.
with DAG(
    dag_id="delivery_capstone_workflow",
    description="Delivery Olist late-delivery MLOps pipeline (SSH bridge to VM).",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["delivery", "delivery", "mlops", "delivery-risk"],
) as dag:

    # Each step is a BashOperator whose command is a `step("pipeline.X")` string — i.e.
    # "run `python -m pipeline.X` on the group VM via the SSH bridge". The worker itself
    # does no data work; it just fires these remote commands and reacts to their exit code.
    load_raw_data = BashOperator(
        task_id="load_raw_data",
        bash_command=step("pipeline.load_raw"),
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=step("pipeline.features"),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=step("pipeline.train"),
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=step("pipeline.register"),
    )

    api_smoke_test = BashOperator(
        task_id="api_smoke_test",
        bash_command=step("pipeline.smoke_test"),
    )

    batch_predict = BashOperator(
        task_id="batch_predict",
        bash_command=step("pipeline.batch_predict"),
    )

    # monitor prints the retrain flag as its last stdout line -> captured as XCom.
    monitor = BashOperator(
        task_id="monitor",
        bash_command=step("pipeline.monitor"),
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
