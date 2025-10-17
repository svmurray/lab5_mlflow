from pathlib import Path
from datetime import datetime

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import ExternalPythonOperator


# -------- Resolve Airflow Variables once (Airflow venv context) --------
APP_PY   = Variable.get("APP_PY")                      # <repo>/app_venv/bin/python
REPO_DIR = Variable.get("REPO_DIR")                    # <repo>
TRACKING = Variable.get("MLFLOW_TRACKING_URI")         # sqlite:////<repo>/mlflow.db
REGISTRY = Variable.get("MLFLOW_REGISTRY_URI")         # sqlite:////<repo>/mlflow.db
ART_DIR  = Variable.get("ART_DIR")                     # <repo>/.airflow/artifacts
ARTIFACT = Variable.get("MLFLOW_ARTIFACT_URI")         # file://<repo>/mlruns

BEST_JSON = str(Path(ART_DIR) / "best.json")


# -------- External callable (runs in app_venv) --------
def _train(repo_dir: str, tracking_uri: str, registry_uri: str, artifact_uri: str, best_json: str):
    # DO NOT import airflow in here — this runs in the app venv.
    import os, sys, json
    os.makedirs(os.path.dirname(best_json), exist_ok=True)  # safety: ensure ART_DIR exists
    os.chdir(repo_dir)                                      # ensure relative paths are repo-rooted
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_REGISTRY_URI"] = registry_uri
    os.environ["MLFLOW_ARTIFACT_URI"] = artifact_uri
    sys.path.append(repo_dir)

    from src.train import run_train
    run_id, metrics = run_train()
    with open(best_json, "w") as f:
        json.dump({"run_id": run_id, "metrics": metrics}, f)


# -------- DAG --------
with DAG(
    dag_id="train_register_promote",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow", "spam"],
) as dag:

    # 1) Make sure our Airflow-side artifacts directory exists
    make_artifacts_dir = BashOperator(
        task_id="make_artifacts_dir",
        bash_command=f"mkdir -p {ART_DIR}",
    )

    # 2) Train in the external (app) venv
    train_model = ExternalPythonOperator(
        task_id="train_model",
        python=str(APP_PY),           # run in app_venv
        python_callable=_train,       # no airflow imports inside
        op_kwargs={
            "repo_dir": REPO_DIR,
            "tracking_uri": TRACKING,
            "registry_uri": REGISTRY,
            "artifact_uri": ARTIFACT,
            "best_json": BEST_JSON,
        },
    )

    # Order: create dir -> train
    make_artifacts_dir >> train_model
