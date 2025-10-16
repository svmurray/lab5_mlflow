# src/evaluate_register.py
import os, math
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "SpamClassifier")
PRIMARY    = os.getenv("PRIMARY_METRIC", "f1_valid")
EPS        = float(os.getenv("PROMOTE_EPSILON", "0.002"))

def _has_mlmodel(client: MlflowClient, run_id: str) -> str | None:
    """
    Return artifact subpath containing 'MLmodel' for this run, or None.
    Prefers 'model/' but will accept any top-level subdir that contains MLmodel.
    """
    for top in client.list_artifacts(run_id, path=""):
        if not top.is_dir:
            continue
        for child in client.list_artifacts(run_id, path=top.path):
            if child.path.endswith("MLmodel"):
                # return 'model' or whatever the top dir is
                return top.path.rstrip("/")
    return None

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

    # Ensure RM exists
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    # Validate the run actually has a logged MLflow model
    subdir = _has_mlmodel(client, run_id)
    if not subdir:
        return {
            "version": None,
            "status": "skipped",
            "reason": "run has no MLmodel artifact",
            "run_id": run_id,
        }

    # Compare against current Staging
    prev = _staging_metric(client, PRIMARY)
    new  = float(new_metrics.get(PRIMARY, float("nan")))
    better = math.isnan(prev) or (new > prev + EPS)

    # Register this run’s model directory as a new version
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/{subdir}"
    mv = client.create_model_version(MODEL_NAME, source, run_id)

    # Optional: tag metrics on the new version
    for k, v in new_metrics.items():
        client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))

    # Promote if better
    if better:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=mv.version,
            stage="Staging", archive_existing_versions=True
        )
        status = f"promoted to Staging (prev={prev}, new={new})"
    else:
        status = f"kept current Staging (prev={prev}, new={new})"

    return {"version": mv.version, "status": status, "new_metric": new, "prev_metric": prev}
