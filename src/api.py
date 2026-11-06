from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer

MODEL_PATH = Path("artifacts/best_model.joblib")

app = FastAPI(
    title="Student MLOps Classification API",
    description="Educational demo only; not for medical decisions."
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists()
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run python -m src.train"
        )

    model = joblib.load(MODEL_PATH)
    expected = len(load_breast_cancer().feature_names)

    if len(request.features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected} features"
        )

    prediction = int(
        model.predict(
            np.array(request.features).reshape(1, -1)
        )[0]
    )

    label = "malignant" if prediction == 0 else "benign"

    return {
        "prediction_code": prediction,
        "predicted_label": label,
        "notice": "Educational dataset output only; not a medical diagnosis."
    }