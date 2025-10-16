import os
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
)
import matplotlib.pyplot as plt


# ---- Config (env-driven, with single-box defaults) --------------------------------
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "sms-exp")
MODEL_NAME   = os.getenv("MODEL_NAME", "SpamClassifier")
C_GRID       = [float(x) for x in os.getenv("C_GRID",  "0.1,1,10").split(",")]
NGRAMS       = [tuple(map(int, p.split("-"))) for p in os.getenv("NGRAMS", "1-1,1-2").split(",")]


# ---- Data -------------------------------------------------------------------------
def _read_split(name: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(Path("data") / f"{name}.csv")
    return df["text"].tolist(), df["label"].tolist()


def _labels_to01(y: Iterable[Any]) -> np.ndarray:
    if not y:
        return np.array([], dtype=int)
    first = next(iter(y))
    if isinstance(first, str):
        return np.array([1 if str(v).strip().lower() == "spam" else 0 for v in y], dtype=int)
    return np.asarray(y, dtype=int)


# ---- Metrics & plots --------------------------------------------------------------
def _compute_metrics(y_true01: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    f1 = f1_score(y_true01, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true01, y_prob) if len(np.unique(y_true01)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    prec, recall, _, _ = precision_recall_fscore_support(
        y_true01, y_pred, average="binary", zero_division=0
    )
    return {
        "f1_valid": float(f1),
        "roc_auc_valid": float(auc),
        "precision_valid": float(prec),
        "recall_valid": float(recall),
    }


def _log_confusion_png(y_true01: np.ndarray, y_pred: np.ndarray, out_png: str = "confusion_valid.png") -> None:
    cm = confusion_matrix(y_true01, y_pred, labels=[0, 1])
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (valid)")
    plt.xlabel("Predicted (0=ham,1=spam)")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    mlflow.log_artifact(out_png)


# ---- Model building ---------------------------------------------------------------
def _make_pipeline(C: float, ngram: Tuple[int, int]) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=1, max_features=20_000, ngram_range=ngram)),
        ("clf",   LogisticRegression(max_iter=500, class_weight="balanced", C=C)),
    ])


def _train_once(C: float, ngram: Tuple[int, int],
                Xtr: List[str], ytr01: np.ndarray,
                Xva: List[str], yva01: np.ndarray) -> Dict[str, Any]:
    """Train one candidate, log params/metrics/artifacts/model in a nested run, return summary."""
    with mlflow.start_run(run_name=f"C={C}_ngr={ngram}", nested=True):
        pipe = _make_pipeline(C, ngram)
        pipe.fit(Xtr, ytr01)

        prob1 = pipe.predict_proba(Xva)[:, 1]
        pred  = (prob1 >= 0.5).astype(int)
        metrics = _compute_metrics(yva01, prob1, pred)

        mlflow.log_params({"C": C, "ngram_min": ngram[0], "ngram_max": ngram[1]})
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=1)  # step=1 avoids SQLite unique collisions

        # Best-effort plot (don’t fail training if plotting breaks)
        try:
            _log_confusion_png(yva01, pred)
        except Exception:
            pass

        # Log a real MLflow model under artifacts/model/
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            input_example={"texts": ["free prize!!!", "meeting moved to 3pm"]},
        )

        run_id = mlflow.active_run().info.run_id
        return {"run_id": run_id, "metrics": metrics, "params": {"C": C, "ngram": ngram}}


# ---- Orchestration ----------------------------------------------------------------
def run_train() -> Tuple[str, Dict[str, float]]:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    Xtr, ytr = _read_split("train")
    Xva, yva = _read_split("valid")
    ytr01, yva01 = _labels_to01(ytr), _labels_to01(yva)

    best: Dict[str, Any] | None = None

    with mlflow.start_run(run_name="sweep"):
        for C in C_GRID:
            for ngr in NGRAMS:
                result = _train_once(C, ngr, Xtr, ytr01, Xva, yva01)
                if (best is None) or (result["metrics"]["f1_valid"] > best["metrics"]["f1_valid"]):
                    best = result

        # persist the selection summary on the parent run
        mlflow.log_dict(best, "best_selection.json")

    assert best is not None, "No candidates were trained"
    return best["run_id"], best["metrics"]


if __name__ == "__main__":
    run_train()
