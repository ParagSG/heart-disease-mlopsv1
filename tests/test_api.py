"""
Unit tests — FastAPI /predict endpoint.
Mocks the model so tests run without a trained model file.
Run: pytest tests/
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# Patch model loading before importing app
mock_model = MagicMock()
mock_model.predict.return_value       = np.array([1])
mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])

with patch("app.joblib.load", return_value=mock_model):
    from app import app, model
    import app as app_module
    app_module.model = mock_model

client = TestClient(app)

VALID_PAYLOAD = {
    "age":      54,
    "sex":      1,
    "cp":       0,
    "trestbps": 130,
    "chol":     245,
    "fbs":      0,
    "restecg":  0,
    "thalach":  150,
    "exang":    0,
    "oldpeak":  1.4,
    "slope":    1,
    "ca":       0,
    "thal":     2,
}


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200


# ── Predict ────────────────────────────────────────────────────────────────────

def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction"  in body
    assert "confidence"  in body
    assert "label"       in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_returns_label():
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert body["label"] in ["Heart Disease Detected", "No Heart Disease"]


def test_predict_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_wrong_type():
    payload = dict(VALID_PAYLOAD)
    payload["age"] = "fifty-four"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_confidence_range():
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_model.predict.return_value       = np.array([0])
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert 0.0 <= response.json()["confidence"] <= 1.0
