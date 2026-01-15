from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"


def _choose_branch() -> str:
    drift_path = LOGS_DIR / "drift.json"
    if not drift_path.exists():
        return "skip_retrain"
    data = json.loads(drift_path.read_text(encoding="utf-8"))
    return "train_model" if data.get("drift") else "skip_retrain"


with DAG(
    dag_id="drift_retrain_register",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    description="Check drift, retrain with PyCaret, register in MLflow",
) as dag:
    make_data = BashOperator(
        task_id="make_data",
        bash_command=f"python {SCRIPTS_DIR / 'make_data.py'}",
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command=f"python {SCRIPTS_DIR / 'drift_check.py'}",
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_choose_branch,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"python {SCRIPTS_DIR / 'train_pycaret.py'}",
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"python {SCRIPTS_DIR / 'register_mlflow.py'}",
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    make_data >> drift_check >> branch
    branch >> skip_retrain
    branch >> train_model >> register_model
