# Heart Disease MLOps – AIMLCZG523 Assignment 01

**BITS Pilani WILP | End-to-End MLOps Pipeline**

Binary classification — predicts presence/absence of heart disease  
from the UCI Heart Disease dataset, deployed as a monitored REST API on AKS.

---

## Project Structure

```
heart-disease-mlops/
├── data/
│   ├── download_data.py        # Dataset acquisition script
│   └── heart.csv               # Generated after download (gitignored)
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── preprocess.py           # Preprocessing pipeline (sklearn)
│   ├── train.py                # Training + MLflow experiment tracking
│   └── app.py                  # FastAPI inference service
├── tests/
│   ├── test_preprocess.py      # Unit tests – data pipeline
│   └── test_api.py             # Unit tests – API endpoints
├── models/                     # Saved model (gitignored, produced by train.py)
├── k8s/
│   ├── deployment.yaml         # AKS Deployment manifest
│   ├── service.yaml            # LoadBalancer Service
│   └── prometheus-configmap.yaml
├── screenshots/                # Submission evidence
├── .github/workflows/ci.yml    # GitHub Actions CI/CD pipeline
├── Dockerfile                  # Multi-stage Docker build
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset
```bash
python data/download_data.py
```

### 3. Run EDA notebook
Open `notebooks/01_eda.ipynb` in Jupyter or VS Code.

### 4. Train models + track with MLflow
```bash
cd src
python train.py
```
View MLflow UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### 5. Run unit tests
```bash
pytest tests/ -v
```

### 6. Run API locally
```bash
cd src
uvicorn app:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 7. Build and run Docker container
```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest

# Test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"cp":0,"trestbps":130,"chol":245,"fbs":0,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":1.4,"slope":1,"ca":0,"thal":2}'
```

---

## Azure AKS Deployment

### Prerequisites
- Azure CLI installed and logged in
- Docker Desktop with AKS credentials configured

### Steps

```bash
# 1. Create resource group and ACR
az group create --name mlops-rg --location eastus
az acr create --resource-group mlops-rg --name <YOUR_ACR_NAME> --sku Basic

# 2. Build and push image to ACR
az acr build --registry <YOUR_ACR_NAME> --image heart-disease-api:latest .

# 3. Create AKS cluster
az aks create --resource-group mlops-rg --name mlops-aks \
  --node-count 2 --generate-ssh-keys \
  --attach-acr <YOUR_ACR_NAME>

# 4. Get credentials
az aks get-credentials --resource-group mlops-rg --name mlops-aks

# 5. Update k8s/deployment.yaml — replace <ACR_LOGIN_SERVER> with your ACR URL
# Then deploy:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 6. Get public IP (wait ~2 minutes)
kubectl get service heart-disease-api-svc
```

---

## GitHub Actions Secrets Required

| Secret | Description |
|--------|-------------|
| `ACR_LOGIN_SERVER` | e.g. `myacr.azurecr.io` |
| `ACR_USERNAME`     | ACR admin username |
| `ACR_PASSWORD`     | ACR admin password |

---

## API Reference

### POST /predict
```json
{
  "age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 245,
  "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 2
}
```
Response:
```json
{
  "prediction": 1,
  "label": "Heart Disease Detected",
  "confidence": 0.82,
  "model_version": "1.0.0"
}
```

### GET /health
```json
{ "status": "ok", "service": "heart-disease-prediction-api" }
```

### GET /metrics
Prometheus metrics endpoint.
