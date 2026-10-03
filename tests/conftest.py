"""Shared fixtures. Tests use synthetic data so they run offline and in seconds."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardiorisk import config
from cardiorisk.features import build_pipeline


def make_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame in the raw UCI layout, with a learnable signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 77, n).astype(float),
            "sex": rng.integers(0, 2, n).astype(float),
            "cp": rng.integers(1, 5, n).astype(float),
            "trestbps": rng.normal(132, 17, n).round(),
            "chol": rng.normal(247, 50, n).round(),
            "fbs": rng.integers(0, 2, n).astype(float),
            "restecg": rng.integers(0, 3, n).astype(float),
            "thalach": rng.normal(150, 22, n).round(),
            "exang": rng.integers(0, 2, n).astype(float),
            "oldpeak": rng.uniform(0, 4, n).round(1),
            "slope": rng.integers(1, 4, n).astype(float),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
        }
    )
    risk = (df["cp"] == 4).astype(int) + df["exang"] + (df["oldpeak"] > 1.5) + (df["ca"] > 0)
    df["num"] = np.where(risk >= 2, rng.integers(1, 5, n), 0)
    df.loc[:3, "ca"] = np.nan
    df.loc[4:5, "thal"] = np.nan
    return df[config.RAW_COLUMNS]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    from cardiorisk.data import clean

    return clean(raw_df)


@pytest.fixture
def sample_patient() -> dict:
    return {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2,
        "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
    }


@pytest.fixture
def model_dir(tmp_path, clean_df):
    """A trained pipeline + metadata persisted the same way train.py does it."""
    from cardiorisk.data import split_xy

    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    joblib.dump(pipe, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model_name": "test_lr", "model_version": "test", "decision_threshold": 0.5})
    )
    return tmp_path
