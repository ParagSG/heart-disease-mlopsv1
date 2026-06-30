"""
Preprocessing pipeline for Heart Disease UCI Dataset.
The fitted pipeline is saved alongside the model so inference
uses identical transformations — no training-serving skew.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Feature groups
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_feature_target(df: pd.DataFrame):
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return X, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Returns an unfitted ColumnTransformer that:
    - Imputes + scales numeric features
    - Imputes + one-hot encodes categorical features
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor
