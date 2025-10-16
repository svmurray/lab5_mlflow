# Lab: MLflow + Airflow (Spam Classifier)

This lab trains a simple spam classifier, tracks runs in **MLflow**, and uses **Airflow** to orchestrate the flow (fetch → train → register → promote). The goal: students can see *real runs* with artifacts and a **Model Registry** entry set to **Staging**.

> ✅ You should **see a `model/MLmodel` file** under each winning run’s **Artifacts**, and a **Model Registry** entry (`SpamClassifier`) where **Staging** points to a valid `mlruns/<exp>/<run>/artifacts/model` source.

---

## Prereqs

- macOS / Linux
- Python **3.11**
- `make`
- Ports free: **8081** (Airflow), **5001** (MLflow UI)

---

## 1) Bootstrap (two venvs)

```bash
# from repo root
make bootstrap
```

This creates:
- `app_venv` → MLflow + sklearn stack
- `airflow_venv` → Airflow with its own constraints

---

## 2) Initialize Airflow & set Variables

```bash
# initialize the metadata DB and admin user
make airflow-init

# set environment via Airflow Variables (uses absolute paths)
make airflow-setvars
```

What `airflow-setvars` does (for clarity):

- `APP_PY` → `<repo>/app_venv/bin/python`
- `REPO_DIR` → `<repo>`
- `ART_DIR` → `<repo>/.airflow/artifacts`
- `MLFLOW_TRACKING_URI` → `sqlite:////<repo>/mlflow.db`  (**absolute** path)
- `MLFLOW_REGISTRY_URI` → `sqlite:////<repo>/mlflow.db`
- `MLFLOW_ARTIFACT_URI` → `file://<repo>/mlruns` (artifact root)
- `SERVICE_RELOAD_URL` → (unused in this lab)

> Having **absolute** SQLite URIs ensures every process uses the same `mlflow.db`.

---

## 3) Start Airflow

Open two terminals:

**Terminal A – Scheduler**
```bash
make airflow-scheduler
```

**Terminal B – Webserver**
```bash
make airflow-webserver
```

Visit Airflow UI: http://127.0.0.1:8081  
Login: `admin` / `admin` (from `airflow-init` target)

You should see a DAG named **`train_register_promote`**.

---

## 4) Run the pipeline

From the Airflow UI:
1. Click **`train_register_promote`**
2. **Trigger DAG**
3. Watch tasks run:  
   - `fetch_data` → prepares `data/train.csv`, `data/valid.csv`  
   - `train_model` → sweeps hyperparams, logs runs to MLflow  
   - `register_and_promote` → registers best run, promotes to **Staging**

---

## 5) Open MLflow UI

```bash
make mlflow-ui
```

Visit MLflow: http://127.0.0.1:5001

### What to look for (important!)

#### A) **Experiment**  
- Left pane → select experiment (default `sms-exp`).
- Click the most recent **child run** (e.g., `C=1.0_ngr=(1,2)`).

#### B) **Artifacts** tab of a run  
- You **must** see a folder:  
  `Artifacts → model → MLmodel`  
- Path preview should look like:  
  `…/mlruns/<exp_id>/<run_id>/artifacts/model/MLmodel`

> If you only see a PNG (e.g., `confusion_valid.png`) and **no** `model/MLmodel`, that run is not a valid candidate for registration.

#### C) **Model Registry**  
- Top nav → **Models** → `SpamClassifier`
- There should be at least one **Version** (e.g., `v1`)
- **Stage** should be **Staging**
- **Source** should look like:  
  `mlruns/<exp_id>/<run_id>/artifacts/model`  
  (Not `mlruns/models/...` and not missing!)

If the Source points to a run whose `Artifacts` tab doesn’t contain `model/MLmodel`, it’s a broken version — see Troubleshooting.

---

## 6) What students should deliver / verify

- A screenshot of the **winning run’s Artifacts** showing `model/MLmodel`.
- A screenshot of **Model Registry** (`SpamClassifier`) with **Staging** pointing to `…/mlruns/<exp>/<run>/artifacts/model`.
- (Optional) A brief note of the best run’s metrics (e.g., `f1_valid`).

---

## Troubleshooting (quick)

### “No runs with a logged MLflow model were found”
- You looked at a fresh/empty DB. Use **absolute** URIs:
  - `MLFLOW_TRACKING_URI = sqlite:////<repo>/mlflow.db`
  - `MLFLOW_REGISTRY_URI = sqlite:////<repo>/mlflow.db`
- Re-run **`make airflow-setvars`** and trigger the DAG again.

### Runs exist, but **Artifacts** folder is empty (no `model/MLmodel`)
- Training didn’t log the model artifact (or wrote to the wrong place).  
  In `src/train.py` ensure:
  ```python
  mlflow.sklearn.log_model(pipe, artifact_path="model")
  ```
- The DAG already enforces a consistent working directory and artifact root via Variables.

### Model Registry **Source** points somewhere that doesn’t exist
- You registered a run without a model. Re-run the DAG; it will:
  - Find the best run **that has** `model/MLmodel`
  - Register a new version
  - Promote to **Staging**
- In MLflow → Model Registry, archive the broken version (optional tidy-up).

### Airflow doesn’t see the DAG
- Use the provided Makefile targets; they export:
  - `AIRFLOW__CORE__DAGS_FOLDER=<repo>/dags`
  - `PYTHONPATH=<repo>`
- Check:
  ```bash
  make airflow-import-errors
  make airflow-list
  ```

---

## Command cheatsheet (for TAs)

```bash
# fresh reset if needed (one-time quick repair)
rm -rf mlruns mlflow.db && mkdir mlruns
make airflow-setvars
make airflow-scheduler
make airflow-webserver
make mlflow-ui
# then trigger the DAG in the UI
```

```bash
# Inspect experiment & runs quickly
./app_venv/bin/python - <<'PY'
import os, mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","sqlite:////%s/mlflow.db" % os.getcwd()))
c = MlflowClient()
exp = c.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT","sms-exp"))
print("EXP:", exp.experiment_id, exp.name, exp.artifact_location)
runs = c.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=3)
for r in runs:
    print("RUN:", r.info.run_id, "artifact_uri:", r.info.artifact_uri)
PY
```

Expected:
- `artifact_location` ends with `…/mlruns`
- Each run’s `artifact_uri` ends with `…/mlruns/<exp_id>/<run_id>/artifacts`

---

## File layout (reference)

```
repo/
  ├─ app_venv/                 # (created by make)
  ├─ airflow_venv/             # (created by make)
  ├─ dags/
  │   └─ train_register_promote_dag.py
  ├─ src/
  │   ├─ data_fetch.py
  │   ├─ train.py              # logs model → artifacts/model/MLmodel
  │   └─ evaluate_register.py  # registers only runs containing MLmodel
  ├─ mlruns/                   # artifacts root
  ├─ mlflow.db                 # tracking DB (SQLite)
  ├─ requirements-mlflow.txt
  ├─ requirements-airflow.txt
  └─ Makefile
```

---

## Notes for instructors

- We intentionally use **two venvs** to avoid dependency pin conflicts (Airflow vs MLflow).
- The DAG uses `ExternalPythonOperator` to run the app code in `app_venv`; Airflow itself does **not** need MLflow installed.
- Metrics are logged at `step=1` to avoid a known SQLite uniqueness collision.
- If students get stuck, have them:
  1) Confirm **absolute** SQLite URIs in Airflow Variables
  2) Confirm `Artifacts → model → MLmodel` exists for the chosen run
  3) Re-run the DAG to produce a valid **Staging** version

---

That’s it. Students should be able to follow this and verify their work entirely in **MLflow** (experiments + registry) without touching the API.
