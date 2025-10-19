# Lab: Lightweight MLOps with Airflow, MLflow, and FastAPI

This lab demonstrates a **minimal end-to-end MLOps pipeline** using a single virtual environment.  
You will:
1. Train an Iris classifier via **Airflow**  
2. Track and register models in **MLflow**  
3. Serve predictions through a **FastAPI** endpoint

---

## 🧱 1. Ensure python3.11 is default
run python3 and see which version is running. 
If not python3.11 then you need to upgrade and set as default. 

```bash
sudo yum install python3.11
sudo ln -s /usr/bin/python3.11 /usr/local/bin/python3
python3 --version
```


## ⚙️ 2. Environment setup

From your repo root:

```bash
# create and activate the virtual environment
bash scripts/setup.sh
source .venv/bin/activate
```

This installs:
- Apache Airflow (SequentialExecutor, SQLite)
- MLflow 3.4
- Scikit-learn, pandas, numpy
- FastAPI + Uvicorn

If you ever need to start clean:
```bash
rm -rf .venv .airflow mlruns mlartifacts mlflow.db
```

---

## 🚀 3. Initialize Airflow

```bash
bash scripts/airflow_init.sh
```

This sets up Airflow’s SQLite metadata DB and an admin user (`admin` / `admin`).

---

## 🏃 4. Start Airflow

In two seperate terminals run airflow webserver and airflow scheduler

Terminal 1
```bash
bash scripts/airflow_webserver.sh
```

Terminal 2
```bash
bash scripts/airflow_scheduler.sh
```


Airflow web UI → http://localhost:8080  
- Log in with **admin / admin**
- Enable and trigger the DAG **`simple_train`**

The DAG runs `src/train.py`, trains a small Iris classifier, and logs results to MLflow.


---

## 📊 5. Start MLflow tracking server

```bash
bash scripts/mlflow_server.sh
```

MLflow UI → http://localhost:5000  
You’ll see:
- The *iris-logreg* run
- Parameters, metrics, and logged model
- You can **register** the model as `iris-classifier` and set its stage to **Production**

---

## 🌐 6. Serve predictions via FastAPI

After a model is registered and promoted:

```bash
bash scripts/fastapi_up.sh
```

FastAPI UI → http://localhost:8000/docs  
Try the **POST /predict** endpoint using the built-in example:

```json
{
  "samples": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5}
  ]
}
```

The API returns predicted Iris species with both class IDs and labels.

---

## ✅ 7. Summary

| Component | Purpose | UI Port |
|------------|----------|---------|
| **Airflow** | Orchestrates training tasks | 8080 |
| **MLflow Server** | Tracks runs and hosts registry | 5000 |
| **FastAPI** | Serves model predictions | 8000 |

All three share the same Python virtual environment for simplicity.  
This lab mirrors a lightweight MLOps pipeline: **Train → Track → Register → Serve.**
