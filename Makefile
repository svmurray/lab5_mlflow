# -------- Python --------
PY := python3.11

# -------- Venvs --------
APP_VENV := app_venv
AF_VENV  := airflow_venv

APP_PY := $(APP_VENV)/bin/python
APP_PIP := $(APP_VENV)/bin/pip

AF_PY := $(AF_VENV)/bin/python
AF_PIP := $(AF_VENV)/bin/pip
AF_BIN := $(AF_VENV)/bin/airflow

# -------- Airflow env (point UI/scheduler at your repo) --------
AF_ENV = AIRFLOW_HOME=$$(pwd)/.airflow \
         AIRFLOW__CORE__DAGS_FOLDER=$$(pwd)/dags \
         PYTHONPATH=$$(pwd)

# ===== Bootstrap targets =====
bootstrap-mlflow:
	rm -rf $(APP_VENV)
	$(PY) -m venv $(APP_VENV)
	$(APP_PY) -m pip install --upgrade pip wheel
	$(APP_PIP) install -r requirements-mlflow.txt

bootstrap-airflow:
	rm -rf $(AF_VENV)
	$(PY) -m venv $(AF_VENV)
	$(AF_PY) -m pip install --upgrade pip wheel
	$(AF_PIP) install -r requirements-airflow.txt

bootstrap: bootstrap-mlflow bootstrap-airflow

# ===== MLflow UI (no gunicorn) =====
mlflow-ui:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	MLFLOW_REGISTRY_URI=sqlite:///mlflow.db \
	$(APP_PY) -m mlflow ui \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1 --port 5001

# ===== FastAPI =====
api:
	$(APP_PY) -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

# ===== Airflow (uses airflow_venv) =====
airflow-init:
	$(AF_ENV) $(AF_PY) -m airflow db init && \
	$(AF_ENV) $(AF_PY) -m airflow users create \
	  --username admin --firstname admin --lastname user \
	  --role Admin --email admin@example.com --password admin || true

airflow-webserver:
	$(AF_ENV) $(AF_PY) -m airflow webserver --port 8081

airflow-scheduler:
	$(AF_ENV) $(AF_PY) -m airflow scheduler

# ===== Debug helpers =====
airflow-list:
	$(AF_ENV) $(AF_PY) -m airflow dags list -v

airflow-import-errors:
	$(AF_ENV) $(AF_PY) -m airflow dags list-import-errors

airflow-conf:
	AIRFLOW_HOME=$$(pwd)/.airflow $(AF_PY) -m airflow config get-value core dags_folder

airflow-setvars:
	$(AF_ENV) $(AF_PY) -m airflow variables set APP_PY "$$(pwd)/app_venv/bin/python"
	$(AF_ENV) $(AF_PY) -m airflow variables set REPO_DIR "$$(pwd)"
	$(AF_ENV) $(AF_PY) -m airflow variables set ART_DIR  "$$(pwd)/.airflow/artifacts"
	# ABSOLUTE tracking DB (note FOUR slashes for sqlite absolute path)
	$(AF_ENV) $(AF_PY) -m airflow variables set MLFLOW_TRACKING_URI "sqlite:////$$(pwd)/mlflow.db"
	$(AF_ENV) $(AF_PY) -m airflow variables set MLFLOW_REGISTRY_URI "sqlite:////$$(pwd)/mlflow.db"
	# Explicit artifact root (repo-local mlruns)
	$(AF_ENV) $(AF_PY) -m airflow variables set MLFLOW_ARTIFACT_URI "file://$$(pwd)/mlruns"
	$(AF_ENV) $(AF_PY) -m airflow variables set SERVICE_RELOAD_URL "http://127.0.0.1:8000/reload"