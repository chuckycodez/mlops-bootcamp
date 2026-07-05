from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"

app = FastAPI(title="Iris ML Model API")

model = joblib.load(MODEL_PATH)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"status": "running", "message": "Iris model API is live"}


@app.post("/predict")
def predict(features: IrisFeatures):
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    prediction = model.predict(input_data)[0]

    return {"prediction": int(prediction)}