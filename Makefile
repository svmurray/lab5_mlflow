# -------- Python --------
PY := python3.11

# -------- Venvs --------
APP_VENV := app_venv
AF_VENV  := airflow_venv

APP_PY := $(APP_VENV)/bin/python
APP_PIP := $(APP_VENV)/bin/pip

AF_PY := $(AF_VENV)/bin/python
AF_PIP := $(AF_VENV)/bin/pip

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

mlflow-ui:
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	MLFLOW_REGISTRY_URI=sqlite:///mlflow.db \
	./app_venv/bin/mlflow ui \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1 --port 5001

api:
	./app_venv/bin/python3.11 -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

airflow-init:
	AIRFLOW_HOME=$$(pwd)/.airflow ./airflow_venv/bin/airflow db init && \
	AIRFLOW_HOME=$$(pwd)/.airflow ./airflow_venv/bin/airflow users create \
	  --username admin --firstname admin --lastname user \
	  --role Admin --email admin@example.com --password admin || true

airflow-webserver:
	AIRFLOW_HOME=$$(pwd)/.airflow ./airflow_venv/bin/airflow webserver --port 8081

airflow-scheduler:
	AIRFLOW_HOME=$$(pwd)/.airflow ./airflow_venv/bin/airflow scheduler