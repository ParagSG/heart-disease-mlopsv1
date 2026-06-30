"""
Unit tests — data preprocessing pipeline.
Run: pytest tests/
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from preprocess import (
    build_preprocessing_pipeline,
    get_feature_target,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET,
)


@pytest.fixture
def sample_df():
    """Minimal valid DataFrame matching the Heart Disease schema."""
    data = {
        "age":      [54, 62, 41],
        "sex":      [1,   0,  1],
        "cp":       [0,   2,  1],
        "trestbps": [130, 140, 120],
        "chol":     [245, 268, 210],
        "fbs":      [0,   0,   1],
        "restecg":  [0,   1,   0],
        "thalach":  [150, 160, 172],
        "exang":    [0,   0,   0],
        "oldpeak":  [1.4, 2.0, 0.5],
        "slope":    [1,   1,   2],
        "ca":       [0,   1,   0],
        "thal":     [2,   3,   2],
        "target":   [1,   0,   0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_nulls(sample_df):
    df = sample_df.copy()
    df.loc[0, "ca"]   = np.nan
    df.loc[1, "thal"] = np.nan
    return df


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_feature_columns_defined():
    assert len(NUMERIC_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0
    assert TARGET not in ALL_FEATURES


def test_get_feature_target_shapes(sample_df):
    X, y = get_feature_target(sample_df)
    assert X.shape == (3, len(ALL_FEATURES))
    assert y.shape == (3,)


def test_get_feature_target_no_target_in_X(sample_df):
    X, _ = get_feature_target(sample_df)
    assert TARGET not in X.columns


def test_pipeline_fit_transform(sample_df):
    X, y = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert X_transformed.shape[0] == 3
    assert not np.isnan(X_transformed).any(), "Transformed output must not contain NaN"


def test_pipeline_handles_missing_values(sample_df_with_nulls):
    X, y = get_feature_target(sample_df_with_nulls)
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert not np.isnan(X_transformed).any(), "Pipeline must impute missing values"


def test_pipeline_transform_consistency(sample_df):
    """Fit on full data, transform a single row — shape must be consistent."""
    X, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X)

    single = X.iloc[[0]]
    result = pipeline.transform(single)
    assert result.shape[0] == 1
    assert result.shape[1] == pipeline.transform(X).shape[1]
