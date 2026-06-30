"""
Model training script with MLflow experiment tracking.
Trains Logistic Regression and Random Forest, logs all runs,
and saves the best model + preprocessing pipeline.

Run: python src/train.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

from preprocess import load_data, get_feature_target, build_preprocessing_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/heart.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

EXPERIMENT_NAME = "heart-disease-classification"

# ── Helpers ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def log_confusion_matrix(y_true, y_pred, run_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix – {run_name}")
    path = f"/tmp/cm_{run_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


def log_roc_curve(y_true, y_prob, run_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=run_name)
    ax.set_title(f"ROC Curve – {run_name}")
    path = f"/tmp/roc_{run_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")


# ── Training ───────────────────────────────────────────────────────────────────
def train_model(name, estimator, param_grid, X_train, y_train, X_test, y_test):
    """
    Runs GridSearchCV, logs everything to MLflow, returns (best_pipeline, metrics).
    """
    preprocessor = build_preprocessing_pipeline()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   estimator),
    ])

    # Prefix param keys for pipeline step
    prefixed_grid = {f"classifier__{k}": v for k, v in param_grid.items()}

    grid = GridSearchCV(pipeline, prefixed_grid, cv=5,
                        scoring="roc_auc", n_jobs=-1, verbose=0)

    with mlflow.start_run(run_name=name):
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # CV score
        cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="roc_auc")

        # Log params
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("model_type", name)

        # Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std",  cv_scores.std())

        # Log artifacts
        log_confusion_matrix(y_test, y_pred, name)
        log_roc_curve(y_test, y_prob, name)

        # Log model
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 registered_model_name=f"HeartDisease-{name}")

        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  Best params : {grid.best_params_}")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print(f"  CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return best, metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Class balance (train): {y_train.value_counts().to_dict()}")

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    lr_params = {
        "C":           [0.01, 0.1, 1, 10],
        "max_iter":    [500],
        "solver":      ["lbfgs"],
        "class_weight": ["balanced", None],
    }
    lr_best, lr_metrics = train_model(
        "LogisticRegression",
        LogisticRegression(random_state=42),
        lr_params,
        X_train, y_train, X_test, y_test,
    )

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    rf_params = {
        "n_estimators":  [100, 200],
        "max_depth":     [None, 5, 10],
        "min_samples_split": [2, 5],
        "class_weight":  ["balanced", None],
    }
    rf_best, rf_metrics = train_model(
        "RandomForest",
        RandomForestClassifier(random_state=42),
        rf_params,
        X_train, y_train, X_test, y_test,
    )

    # ── Save best model ───────────────────────────────────────────────────────
    best_model  = rf_best  if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else lr_best
    best_name   = "RandomForest" if rf_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "LogisticRegression"

    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nBest model: {best_name}  |  ROC-AUC: {max(rf_metrics['roc_auc'], lr_metrics['roc_auc']):.4f}")
    print(f"Saved to  : {model_path}")


if __name__ == "__main__":
    main()
