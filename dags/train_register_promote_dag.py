from __future__ import annotations

from pathlib import Path
from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import ExternalPythonOperator


def _get_required_var(key: str) -> str:
    val = Variable.get(key, default_var=None)
    if not val:
        raise AirflowFailException(
            f"Airflow Variable '{key}' is not set. "
            f"Run 'make airflow-setvars' or set it in the UI."
        )
    return val


# ---------- TOP-LEVEL FUNCTIONS (REQUIRED) ----------

def _train(repo_dir: str, tracking_uri: str, registry_uri: str, artifact_uri: str, best_json: str):
    # Runs in app_venv. Do NOT import airflow here.
    import os, sys, json
    os.makedirs(os.path.dirname(best_json), exist_ok=True)  # <- fixed
    os.chdir(repo_dir)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    os.environ["MLFLOW_ARTIFACT_URI"] = artifact_uri
    sys.path.append(repo_dir)
    from src.train import run_train  # import inside app venv
    run_id, metrics = run_train()
    with open(best_json, "w") as f:
        json.dump({"run_id": run_id, "metrics": metrics}, f)


def _register_wrapper(
    repo_dir: str,
    tracking_uri: str,
    registry_uri: str,
    best_json: str,
    model_name: str,
    stage: str,
):
    """
    Runs in app_venv. Do NOT import airflow here.
    """
    import os, sys
    os.chdir(repo_dir)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    sys.path.append(repo_dir)
    # Import inside app venv
    from src.register_promote import register_and_promote
    return register_and_promote(
        repo_dir=repo_dir,
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        best_json=best_json,
        model_name=model_name,
        stage=stage,
        archive_existing=True,
    )


# ---------- DAG FACTORY ----------

def build_dag() -> DAG:
    APP_PY   = _get_required_var("APP_PY")               # <repo>/app_venv/bin/python
    REPO_DIR = _get_required_var("REPO_DIR")             # <repo>
    TRACKING = _get_required_var("MLFLOW_TRACKING_URI")  # sqlite:////<repo>/mlflow.db
    REGISTRY = _get_required_var("MLFLOW_REGISTRY_URI")  # sqlite:////<repo>/mlflow.db
    ART_DIR  = _get_required_var("ART_DIR")              # <repo>/.airflow/artifacts
    ARTIFACT = _get_required_var("MLFLOW_ARTIFACT_URI")  # file://<repo>/mlruns

    BEST_JSON = str(Path(ART_DIR) / "best.json")

    with DAG(
        dag_id="train_register_promote",
        start_date=datetime(2024, 1, 1),
        schedule=None,
        catchup=False,
        tags=["mlflow", "spam"],
    ) as dag:

        make_artifacts_dir = BashOperator(
            task_id="make_artifacts_dir",
            bash_command=f"mkdir -p {ART_DIR}",
        )

        train_model = ExternalPythonOperator(
            task_id="train_model",
            python=str(APP_PY),
            python_callable=_train,                 # ✅ top-level function
            op_kwargs={
                "repo_dir": REPO_DIR,
                "tracking_uri": TRACKING,
                "registry_uri": REGISTRY,
                "artifact_uri": ARTIFACT,
                "best_json": BEST_JSON,
            },
        )

        register_and_promote = ExternalPythonOperator(
            task_id="register_and_promote",
            python=str(APP_PY),
            python_callable=_register_wrapper,      # ✅ top-level function
            op_kwargs={
                "repo_dir": REPO_DIR,
                "tracking_uri": TRACKING,
                "registry_uri": REGISTRY,
                "best_json": BEST_JSON,
                "model_name": "SpamClassifier",
                "stage": "Staging",
            },
        )

        make_artifacts_dir >> train_model >> register_and_promote
        return dag


# Airflow discovers this
dag = build_dag()