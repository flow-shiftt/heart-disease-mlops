import numpy as np
import pandas as pd
import pytest

from cardiorisk import config
from cardiorisk.data import split_xy
from cardiorisk.features import ClinicalFeatureBuilder, build_preprocessor


def test_hr_reserve_pct_formula(clean_df):
    X, _ = split_xy(clean_df)
    out = ClinicalFeatureBuilder().fit(X).transform(X)
    expected = X["thalach"] / (220 - X["age"])
    np.testing.assert_allclose(out["hr_reserve_pct"], expected)


def test_bp_chol_load_uses_training_statistics_only(clean_df):
    X, _ = split_xy(clean_df)
    builder = ClinicalFeatureBuilder().fit(X)
    shifted = X.copy()
    shifted["chol"] += 100
    # Statistics must come from fit(), not from the batch being transformed.
    assert builder.transform(shifted)["bp_chol_load"].mean() != pytest.approx(
        ClinicalFeatureBuilder().fit(shifted).transform(shifted)["bp_chol_load"].mean()
    )


def test_preprocessor_output_has_no_nans_and_expected_width(clean_df):
    X, _ = split_xy(clean_df)
    engineered = ClinicalFeatureBuilder().fit_transform(X)
    pre = build_preprocessor().fit(engineered)
    Xt = pre.transform(engineered)
    n_onehot = sum(len(v) for v in config.CATEGORY_LEVELS.values())
    expected = len(config.NUMERIC_FEATURES) + len(config.ENGINEERED_FEATURES) + len(config.BINARY_FEATURES) + n_onehot
    assert Xt.shape == (len(X), expected)
    assert not np.isnan(Xt).any()


def test_unknown_category_does_not_crash(clean_df):
    X, _ = split_xy(clean_df)
    engineered = ClinicalFeatureBuilder().fit_transform(X)
    pre = build_preprocessor().fit(engineered)
    odd = engineered.head(1).copy()
    odd["cp"] = 9
    assert pre.transform(odd).shape[0] == 1


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_xy(clean_df)
    engineered = ClinicalFeatureBuilder().fit_transform(X)
    pre = build_preprocessor().fit(engineered)
    Xt = pd.DataFrame(pre.transform(engineered), columns=pre.get_feature_names_out())
    assert Xt["age"].mean() == pytest.approx(0, abs=1e-6)
    assert Xt["age"].std(ddof=0) == pytest.approx(1, abs=1e-6)
