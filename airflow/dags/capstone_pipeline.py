from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def phase(name: str) -> None:
    print(f"delivery reference workflow phase: {name}")


with DAG(
    "delivery_capstone_workflow",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["delivery", "delivery"],
) as dag:
    load = PythonOperator(task_id="load_data", python_callable=phase, op_args=["load"])
    features = PythonOperator(task_id="build_features", python_callable=phase, op_args=["features"])
    train = PythonOperator(task_id="train_and_register", python_callable=phase, op_args=["train"])
    batch = PythonOperator(task_id="batch_predict", python_callable=phase, op_args=["batch_predict"])
    monitor = PythonOperator(task_id="monitor_and_retrain", python_callable=phase, op_args=["monitor"])

    load >> features >> train >> batch >> monitor
