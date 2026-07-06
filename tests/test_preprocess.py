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


# ── Corner Cases ───────────────────────────────────────────────────────────────

def test_pipeline_all_nulls_in_optional_cols(sample_df):
    """ca and thal can all be NaN — pipeline must still impute and not crash."""
    df = sample_df.copy()
    df["ca"]   = np.nan
    df["thal"] = np.nan
    X, _ = get_feature_target(df)
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert not np.isnan(X_transformed).any()


def test_pipeline_output_is_numeric(sample_df):
    """All output values must be finite floats after transformation."""
    X, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    assert np.isfinite(X_transformed).all(), "Pipeline output must be all finite"


def test_pipeline_scaled_numeric_features(sample_df):
    """StandardScaler should centre numeric features close to mean=0."""
    X, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    # Numeric features are the first len(NUMERIC_FEATURES) columns
    numeric_out = X_transformed[:, :len(NUMERIC_FEATURES)]
    assert abs(numeric_out.mean()) < 2.0, "Numeric features should be roughly centred"


def test_get_feature_target_binary_labels(sample_df):
    """Target must only contain 0 and 1."""
    _, y = get_feature_target(sample_df)
    assert set(y.unique()).issubset({0, 1}), "Target must be binary (0 or 1)"


def test_pipeline_new_categorical_value(sample_df):
    """Pipeline must not crash on unseen categorical value at inference time."""
    X_train, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)

    X_new = X_train.copy()
    X_new.loc[0, "cp"] = 99
    result = pipeline.transform(X_new)
    assert not np.isnan(result).any(), "Unseen category should be handled gracefully"


def test_pipeline_is_deterministic(sample_df):
    """Same input must always produce exactly the same output."""
    X, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    out1 = pipeline.fit_transform(X)
    pipeline2 = build_preprocessing_pipeline()
    out2 = pipeline2.fit_transform(X)
    np.testing.assert_array_almost_equal(out1, out2)


def test_pipeline_single_row(sample_df):
    """Fitting and transforming a single-row DataFrame must not crash."""
    X, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X)
    single = X.iloc[[0]]
    result = pipeline.transform(single)
    assert result.shape[0] == 1
    assert not np.isnan(result).any()


def test_all_features_no_overlap():
    """NUMERIC and CATEGORICAL feature lists must not overlap."""
    overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert len(overlap) == 0, f"Features appear in both lists: {overlap}"


def test_all_features_complete():
    """ALL_FEATURES must equal NUMERIC + CATEGORICAL — no missing, no extras."""
    assert sorted(ALL_FEATURES) == sorted(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_pipeline_transform_before_fit_raises(sample_df):
    """Calling transform before fit must raise an error."""
    X, _ = get_feature_target(sample_df)
    pipeline = build_preprocessing_pipeline()
    with pytest.raises(Exception):
        pipeline.transform(X)
