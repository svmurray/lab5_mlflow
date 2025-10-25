# app/server.py
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
MODEL_VERSION       = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(MODEL_URI)

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }
    
class VersionRequest(BaseModel):
    version: str

    model_config = {
        "json_schema_extra": {
            "examples": [ {"version": "1"}
            ]
        }
    }
    
class VersionResponse(BaseModel):
    version: str #Model version cast as string to protect against non-int types
    status: str  #status message regarding the Model version

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.get(
    "/version", 
    response_model=VersionResponse,
    tags=["model"],
    summary = 'Retrieve current model version',
    description = 'Returns the current version of the model being served.')
def version():
    print(MODEL_VERSION)
    return VersionResponse(
        version=MODEL_VERSION,
        status='Currently being served')
    
@app.post(
    "/set_version", 
    response_model=VersionResponse,
    tags=["model"],
    summary = 'Set current model version',
    description = 'Attempts to set the Model Version to the input parameter.')
def set_version(req: VersionRequest) -> VersionRequest:
    status_string = 'Attempted to set'
    try:
        global MODEL_VERSION
        global MODEL_URI
        global model
        MODEL_VERSION = req.version
        MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        model = mlflow.pyfunc.load_model(MODEL_URI)
        status_string = 'Model Version successfully updated.'
    except Exception as e:
        status_string = 'Failed to set Model Version. Please ensure that it exists.'
    return VersionResponse(
        version=MODEL_VERSION,
        status=status_string)

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
)
def predict(req: PredictRequest) -> PredictResponse:
    ids = []
    labels = []
    for s in req.samples:
        res = model.predict([[item[1] for item in s]])
        ids += [res[0]]
        labels += [IRIS_LABELS[res[0]]]
    return PredictResponse(
        class_id=ids,
        class_label=labels
    )
    
# TODO Add endpoint to get the current model serving version
# TODO Add endpoint to update the serving version
# TODO Predict using the correct served version
