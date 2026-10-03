"""Feature engineering and the shared preprocessing pipeline.

The full chain (engineering -> imputation -> scaling/encoding -> model) is one
sklearn Pipeline, so the exact same transformations run in cross-validation,
in the saved artefact and inside the serving container.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cardiorisk import config


class ClinicalFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds two domain-motivated features.

    hr_reserve_pct
        Peak heart rate as a share of the age-predicted maximum (220 - age).
        A low value ("chronotropic incompetence") is a known marker of
        coronary disease and is more informative than raw thalach because it
        removes the age effect.
    bp_chol_load
        Product of standardised resting BP and cholesterol, capturing the
        combined vascular load of two risk factors that are weak on their own.
        Reference means/stds are learned in fit() so there is no leakage.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X, columns=config.INPUT_FEATURES)
        self.bp_mu_, self.bp_sd_ = float(X["trestbps"].mean()), float(X["trestbps"].std() or 1.0)
        self.ch_mu_, self.ch_sd_ = float(X["chol"].mean()), float(X["chol"].std() or 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=config.INPUT_FEATURES).copy()
        predicted_max = (220 - X["age"]).clip(lower=1)
        X["hr_reserve_pct"] = X["thalach"] / predicted_max
        bp_z = (X["trestbps"] - self.bp_mu_) / self.bp_sd_
        ch_z = (X["chol"] - self.ch_mu_) / self.ch_sd_
        X["bp_chol_load"] = bp_z * ch_z
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(config.INPUT_FEATURES + config.ENGINEERED_FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    # fixed, float-typed levels: stable output width even if a fold misses a level
                    categories=[
                        [float(v) for v in config.CATEGORY_LEVELS[c]] for c in config.CATEGORICAL_FEATURES
                    ],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES),
            ("bin", binary, config.BINARY_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("engineer", ClinicalFeatureBuilder()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
