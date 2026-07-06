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


# ── Corner Cases ───────────────────────────────────────────────────────────────

def test_predict_empty_body():
    """Empty JSON body should return 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_extra_fields_ignored():
    """Extra unknown fields should not break the API."""
    payload = dict(VALID_PAYLOAD)
    payload["unknown_field"] = 999
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_negative_age():
    """Negative age is biologically invalid — API should reject it (422)."""
    payload = dict(VALID_PAYLOAD)
    payload["age"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_extreme_values():
    """Extreme but type-valid values should still return a prediction."""
    mock_model.predict.return_value       = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    payload = dict(VALID_PAYLOAD)
    payload["age"]     = 120
    payload["chol"]    = 999
    payload["thalach"] = 250
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_predict_label_matches_prediction():
    """Label must correspond correctly to prediction value."""
    mock_model.predict.return_value       = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    if body["prediction"] == 1:
        assert body["label"] == "Heart Disease Detected"
    else:
        assert body["label"] == "No Heart Disease"


def test_metrics_endpoint():
    """Prometheus /metrics endpoint must be reachable."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_predict_null_field():
    """Null value for a required field should return 422."""
    payload = dict(VALID_PAYLOAD)
    payload["age"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_model_not_loaded():
    """If model is None, /predict must return 503 not 500."""
    original = app_module.model
    app_module.model = None
    response = client.post("/predict", json=VALID_PAYLOAD)
    app_module.model = original
    assert response.status_code == 503


def test_health_reflects_model_not_loaded():
    """Health endpoint must report 'model_not_loaded' when model is None."""
    original = app_module.model
    app_module.model = None
    response = client.get("/health")
    app_module.model = original
    assert response.json()["status"] == "model_not_loaded"


def test_predict_response_has_model_version():
    """Response must include model_version field."""
    mock_model.predict.return_value       = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert "model_version" in response.json()


def test_predict_confidence_is_positive_class_prob():
    """Confidence must be the probability of class 1 (disease), not class 0."""
    mock_model.predict.return_value       = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert abs(response.json()["confidence"] - 0.7) < 0.001


def test_predict_get_method_not_allowed():
    """GET on /predict must return 405 Method Not Allowed."""
    response = client.get("/predict")
    assert response.status_code == 405


def test_predict_float_inputs_accepted():
    """All fields are typed as float — float inputs must work fine."""
    payload = dict(VALID_PAYLOAD)
    payload["age"]     = 54.5
    payload["oldpeak"] = 1.45
    mock_model.predict.return_value       = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.65, 0.35]])
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
