import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5001")
MODEL_NAME = os.getenv("MODEL_NAME","SpamClassifier")
PRIMARY    = os.getenv("PRIMARY_METRIC","f1_valid")
EPS        = float(os.getenv("PROMOTE_EPSILON","0.002"))

def _get_metric(client: MlflowClient, run_id: str, metric: str) -> float:
    r = client.get_run(run_id)
    hist = [m.value for m in r.data.metrics.items() if False]  # n/a
    # Current mlflow client exposes last logged value under run.data.metrics
    return float(r.data.metrics.get(metric, float("nan")))

def _staging_metric(client: MlflowClient, metric: str) -> float:
    try:
        latest = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not latest:
            return float("nan")
        r = client.get_run(latest[0].run_id)
        return float(r.data.metrics.get(metric, float("nan")))
    except Exception:
        return float("nan")

def promote_if_better(run_id: str, new_metrics: dict):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    current = _staging_metric(client, PRIMARY)
    new = float(new_metrics.get(PRIMARY, float("nan")))
    better = (current != current) or (new > current + EPS)  # NaN check

    # Ensure RM exists
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    # Create model version from run artifact
    run = client.get_run(run_id)
    src = f"{run.info.artifact_uri}/model"
    mv = client.create_model_version(MODEL_NAME, src, run_id)

    # Tag useful metadata
    for k,v in new_metrics.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
    client.set_model_version_tag(MODEL_NAME, mv.version, "data_source", os.getenv("DATA_SOURCE","sms_spam"))
    client.set_model_version_tag(MODEL_NAME, mv.version, "data_version", os.getenv("DATA_VERSION","v1"))

    if better:
        client.transition_model_version_stage(MODEL_NAME, mv.version, stage="Staging", archive_existing_versions=True)
        status = f"promoted to Staging (prev={current}, new={new})"
    else:
        status = f"kept current Staging (prev={current}, new={new})"
    return {"version": mv.version, "status": status, "new_metric": new, "prev_metric": current}

if __name__ == "__main__":
    # Manual test:
    print("Use from DAG with run_id + metrics dict.")
