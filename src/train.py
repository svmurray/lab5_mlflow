import os, json
import mlflow, mlflow.sklearn
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI","http://127.0.0.1:5001")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT","sms-exp")
MODEL_NAME = os.getenv("MODEL_NAME","SpamClassifier")

def _read_split(name: str):
    df = pd.read_csv(Path("data")/f"{name}.csv")
    return df["text"].tolist(), df["label"].tolist()

def _metric_pack(y_true, y_prob, y_pred):
    # map labels to 0/1 with spam=1
    y_true01 = np.array([1 if y=="spam" else 0 for y in y_true])
    f1 = f1_score(y_true01, y_pred)
    try:
        auc = roc_auc_score(y_true01, y_prob)
    except Exception:
        auc = float("nan")
    prec, recall, _, _ = precision_recall_fscore_support(y_true01, y_pred, average="binary", zero_division=0)
    return {"f1_valid": f1, "roc_auc_valid": float(auc), "precision_valid": float(prec), "recall_valid": float(recall)}

def _plot_confusion(y_true, y_pred, out_png: str):
    y_true01 = np.array([1 if y=="spam" else 0 for y in y_true])
    cm = confusion_matrix(y_true01, y_pred, labels=[0,1])
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (valid)")
    plt.xlabel("Predicted (0=ham,1=spam)")
    plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def run_train():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    Xtr, ytr = _read_split("train")
    Xva, yva = _read_split("valid")

    # sweep grids from env
    C_GRID = [float(x) for x in os.getenv("C_GRID","0.1,1,10").split(",")]
    NGRAMS = []
    for n in os.getenv("NGRAMS","1-1,1-2").split(","):
        a,b = n.split("-"); NGRAMS.append((int(a),int(b)))

    best = None

    with mlflow.start_run(run_name="sweep") as parent:
        for C in C_GRID:
            for ngr in NGRAMS:
                with mlflow.start_run(run_name=f"C={C}_ngr={ngr}", nested=True) as child:
                    pipe = Pipeline([
                        ("tfidf", TfidfVectorizer(min_df=1, max_features=20000, ngram_range=ngr)),
                        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=C))
                    ])
                    pipe.fit(Xtr, ytr)
                    # Valid predictions
                    prob1 = pipe.predict_proba(Xva)[:,1]
                    pred  = (prob1 >= 0.5).astype(int)
                    metrics = _metric_pack(yva, prob1, pred)
                    mlflow.log_params({"C":C, "ngram_min":ngr[0], "ngram_max":ngr[1]})
                    for k,v in metrics.items():
                        mlflow.log_metric(k, v)
                    # Confusion plot
                    _plot_confusion(yva, pred, "confusion_valid.png")
                    mlflow.log_artifact("confusion_valid.png")
                    # Log model (unregistered) for this child
                    mlflow.sklearn.log_model(pipe, artifact_path="model")
                    # Track best
                    if (best is None) or (metrics["f1_valid"] > best["f1_valid"]):
                        best = {"metrics":metrics, "params":{"C":C, "ngram":ngr}, "run_id": mlflow.active_run().info.run_id}
        # Record best in parent run
        mlflow.log_dict(best, "best_selection.json")

    return best["run_id"], best["metrics"]
    
if __name__ == "__main__":
    run_train()
