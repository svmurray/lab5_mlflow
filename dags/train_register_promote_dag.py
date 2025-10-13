import os, json
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

from src.data_fetch import fetch
from src.train import run_train
from src.evaluate_register import promote_if_better
import requests

def _do_train_push(**ctx):
    run_id, metrics = run_train()
    ctx["ti"].xcom_push(key="run_id", value=run_id)
    ctx["ti"].xcom_push(key="metrics", value=json.dumps(metrics))

def _promote_branch(**ctx):
    run_id = ctx["ti"].xcom_pull(key="run_id")
    metrics = json.loads(ctx["ti"].xcom_pull(key="metrics"))
    res = promote_if_better(run_id, metrics)
    ctx["ti"].xcom_push(key="promote_result", value=json.dumps(res))
    if "promoted" in res["status"]:
        return "reload_service"
    return "skip_reload"

def _reload():
    try:
        requests.post("http://127.0.0.1:8000/reload", timeout=5)
    except Exception as e:
        print("Reload failed:", e)


with DAG(
    dag_id="train_register_promote",
    start_date=datetime(2024,1,1),
    schedule_interval=None,
    catchup=False
) as dag:

    fetch_data = PythonOperator(task_id="fetch_data", python_callable=fetch)
    train = PythonOperator(task_id="train_model", python_callable=_do_train_push, provide_context=True)
    decision = BranchPythonOperator(task_id="decide_promotion", python_callable=_promote_branch, provide_context=True)
    reload_svc = PythonOperator(task_id="reload_service", python_callable=_reload, trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    skip_reload = PythonOperator(task_id="skip_reload", python_callable=lambda: print("no promotion"))

    fetch_data >> train >> decision
    decision >> [reload_svc, skip_reload]
