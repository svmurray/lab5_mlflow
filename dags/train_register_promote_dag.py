from __future__ import annotations
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import ExternalPythonOperator
from airflow.operators.bash import BashOperator

# ---- All “environment” is read from Airflow Variables (set once via Makefile) ----
APP_PY = Variable.get("APP_PY")
REPO_DIR = Variable.get("REPO_DIR")
ART_DIR = Path(Variable.get("ART_DIR"))
TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
REGISTRY_URI = Variable.get("MLFLOW_REGISTRY_URI")
SERVICE_RELOAD_URL = Variable.get("SERVICE_RELOAD_URL")

BEST_JSON = (ART_DIR / "best_run.json").as_posix()

# ----- Callables executed inside the external interpreter (app_venv) -----
def _fetch(repo_dir: str):
    import sys
    sys.path.append(repo_dir)
    from src.data_fetch import fetch
    fetch()

def _train(repo_dir: str, tracking_uri: str, registry_uri: str, best_json: str):
    import os, sys, json
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    sys.path.append(repo_dir)
    from src.train import run_train
    run_id, metrics = run_train()
    with open(best_json, "w") as f:
        json.dump({"run_id": run_id, "metrics": metrics}, f)

def _register(repo_dir: str, tracking_uri: str, registry_uri: str, best_json: str):
    import os, sys, json
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    sys.path.append(repo_dir)
    from src.evaluate_register import promote_if_better
    with open(best_json) as f:
        payload = json.load(f)
    res = promote_if_better(payload["run_id"], payload["metrics"])
    print("register/promote:", res)

# ----- DAG definition (no env specifics; all values come from Variables) -----
with DAG(
    dag_id="train_register_promote",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description="Train → register → conditional promote → reload API",
) as dag:

    fetch_data = ExternalPythonOperator(
        task_id="fetch_data",
        python=APP_PY,
        python_callable=_fetch,
        op_kwargs={"repo_dir": REPO_DIR},
    )

    train_model = ExternalPythonOperator(
        task_id="train_model",
        python=APP_PY,
        python_callable=_train,
        op_kwargs={
            "repo_dir": REPO_DIR,
            "tracking_uri": TRACKING_URI,
            "registry_uri": REGISTRY_URI,
            "best_json": BEST_JSON,
        },
    )

    register_and_promote = ExternalPythonOperator(
        task_id="register_and_promote",
        python=APP_PY,
        python_callable=_register,
        op_kwargs={
            "repo_dir": REPO_DIR,
            "tracking_uri": TRACKING_URI,
            "registry_uri": REGISTRY_URI,
            "best_json": BEST_JSON,
        },
    )

    reload_service = BashOperator(
        task_id="reload_service",
        bash_command=f'curl -s -X POST "{SERVICE_RELOAD_URL}" || true && echo "reload attempted"',
    )

    fetch_data >> train_model >> register_and_promote >> reload_service
