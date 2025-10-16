import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from mlflow.exceptions import MlflowException

# ---- Config ----
MODEL_NAME = os.getenv("MODEL_NAME", "SpamClassifier")
TRACKING   = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

from pathlib import Path
REPO_DIR = Path(__file__).resolve().parents[1]
ABS_DB = f"sqlite:////{REPO_DIR / 'mlflow.db'}"
os.environ.setdefault("MLFLOW_TRACKING_URI", ABS_DB)
os.environ.setdefault("MLFLOW_REGISTRY_URI", ABS_DB)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# ---- App state ----
app = FastAPI(title="SpamClassifier API")
_model = None
_version = None
_last_error = None  # str | None

def _try_load_staging():
    """Attempt to load models:/<MODEL_NAME>/Staging. Set globals; never raise."""
    global _model, _version, _last_error
    _last_error = None
    _version = None
    _model = None
    try:
        mlflow.set_tracking_uri(TRACKING)
        m = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Staging")
        _model = m
        # Try to extract something version-like for /healthz
        meta = getattr(m, "metadata", None)
        _version = getattr(meta, "model_uuid", None) or getattr(meta, "run_id", None) or "unknown"
    except Exception as e:
        # Common cases:
        # - Registered model doesn't exist yet
        # - Staging has no versions yet
        # - Tracking URI misconfigured
        _last_error = f"{type(e).__name__}: {e}"

class PredictReq(BaseModel):
    texts: List[str]
    explain: Optional[bool] = False
    k: Optional[int] = 3

# NOTE: no eager load here — let the service boot even with no model
# and load on first /reload or first /predict.

@app.get("/healthz")
def healthz():
    ready = _model is not None
    return {
        "model_name": MODEL_NAME,
        "tracking_uri": TRACKING,
        "ready": ready,
        "version": _version if ready else None,
        "last_error": _last_error,
        "hint": (
            None if ready else
            "No Staging model loaded. Train & register a model, promote to Staging, "
            "then call /reload. Example: run the Airflow DAG or src/train.py + evaluate_register."
        ),
    }

@app.post("/reload")
def reload():
    _try_load_staging()
    if _model is None:
        # Stay 200 but report not ready, so scripts don't fail hard.
        return {"status": "not_ready", "reason": _last_error}
    return {"status": "reloaded", "version": _version}

# Optional lightweight explain helper (works only if model is loaded)
def _top_tokens_for_text(pipe, text: str, k: int = 3):
    try:
        from sklearn.pipeline import Pipeline
        from numpy import argsort, abs as np_abs
        tfidf = pipe.named_steps.get("tfidf")
        clf   = pipe.named_steps.get("clf")
        if tfidf is None or clf is None:
            return []
        feats = tfidf.get_feature_names_out()
        X = tfidf.transform([text]).toarray().ravel()
        coefs = getattr(clf, "coef_", None)
        if coefs is None:
            return []
        contrib = X * coefs.ravel()
        idx = argsort(-np_abs(contrib))[:k]
        return [(feats[i], float(contrib[i])) for i in idx]
    except Exception:
        return []

@app.post("/predict")
def predict(req: PredictReq):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "MODEL_NOT_READY",
                "message": "No Staging model is loaded yet.",
                "hint": "Run training/registration to Staging, then POST /reload.",
                "tracking_uri": TRACKING,
                "model_name": MODEL_NAME,
                "last_error": _last_error,
            },
        )
    pipe = _model
    try:
        proba = pipe.predict_proba(req.texts)[:, 1]
        pred = (proba >= 0.5).astype(int).tolist()
        out = {"predictions": pred, "probabilities": proba.tolist()}
        if req.explain:
            out["explanations"] = [_top_tokens_for_text(pipe, t, req.k) for t in req.texts]
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
