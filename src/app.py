"""
FastAPI inference service for Heart Disease prediction.
Endpoints:
  GET  /           → health check
  GET  /health     → health check
  POST /predict    → prediction + confidence
  GET  /metrics    → Prometheus metrics (via instrumentator)
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment 01 – AIMLCZG523, BITS Pilani WILP",
    version="1.0.0",
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "../models/best_model.joblib"))
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Run src/train.py first.")


# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age:      float = Field(..., example=54, description="Age in years")
    sex:      float = Field(..., example=1,  description="1=male, 0=female")
    cp:       float = Field(..., example=0,  description="Chest pain type (0-3)")
    trestbps: float = Field(..., example=130, description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., example=245, description="Serum cholesterol (mg/dl)")
    fbs:      float = Field(..., example=0,  description="Fasting blood sugar >120mg/dl (1=true)")
    restecg:  float = Field(..., example=0,  description="Resting ECG results (0-2)")
    thalach:  float = Field(..., example=150, description="Maximum heart rate achieved")
    exang:    float = Field(..., example=0,  description="Exercise-induced angina (1=yes)")
    oldpeak:  float = Field(..., example=1.4, description="ST depression induced by exercise")
    slope:    float = Field(..., example=1,  description="Slope of peak exercise ST segment (0-2)")
    ca:       float = Field(..., example=0,  description="Number of major vessels coloured by fluoroscopy (0-3)")
    thal:     float = Field(..., example=2,  description="Thal: 1=normal, 2=fixed defect, 3=reversible defect")


class PredictResponse(BaseModel):
    prediction:   int
    label:        str
    confidence:   float
    model_version: str = "1.0.0"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status, "service": "heart-disease-prediction-api"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([request.model_dump()])

    try:
        prediction  = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    logger.info(f"Prediction={prediction} | Confidence={probability:.4f} | Input={request.model_dump()}")

    return PredictResponse(
        prediction=prediction,
        label=label,
        confidence=round(probability, 4),
    )
