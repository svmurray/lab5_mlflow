from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator

# Minimal training wrapper that sets MLFLOW_TRACKING_URI env and runs train.py
def run_training():
    import os, sys, subprocess, pathlib, shlex

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    train_py = repo_root / "src" / "train.py"

    # Environment: write runs to repo_root/mlruns, ensure src/ imports work if needed
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env.get("PYTHONPATH","") if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, str(train_py)]
    print(f"[runner] exec: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[runner] cwd:  {repo_root}")
    print(f"[runner] MLFLOW_TRACKING_URI={env['MLFLOW_TRACKING_URI']}")

    # Capture output so Airflow logs show the real error
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        # keep stderr visible in Airflow logs
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"train.py failed with exit code {proc.returncode}")
default_args = {
    "owner": "you",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="train_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual only
    catchup=False,
    default_args=default_args,
    description="Train a tiny model and log to MLflow",
) as dag:
    train = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
    )

    train