import numpy as np
import pandas as pd

from cardiorisk import config
from cardiorisk.data import clean, load_raw, missing_report, split_xy


def test_target_is_binarised(raw_df):
    out = clean(raw_df)
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns
    assert out[config.TARGET].sum() == (raw_df["num"] > 0).sum()


def test_missing_values_preserved_for_pipeline_imputation(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() == 4
    assert out["thal"].isna().sum() == 2


def test_out_of_range_values_are_nulled(raw_df):
    raw = raw_df.copy()
    raw.loc[10, "chol"] = 5000
    raw.loc[11, "trestbps"] = 0
    out = clean(raw)
    assert pd.isna(out.loc[10, "chol"])
    assert pd.isna(out.loc[11, "trestbps"])


def test_invalid_category_codes_are_nulled(raw_df):
    raw = raw_df.copy()
    raw.loc[20, "thal"] = 5  # not a valid thal code
    out = clean(raw)
    assert pd.isna(out.loc[20, "thal"])


def test_duplicates_dropped(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[[7]]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_load_raw_parses_question_marks(tmp_path):
    line = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,6.0,0\n"
    path = tmp_path / "raw.data"
    path.write_text(line * 3)
    df = load_raw(path)
    assert df.shape == (3, 14)
    assert df["ca"].isna().all()
    assert missing_report(df).to_dict() == {"ca": 3}


def test_split_xy_shapes_and_dtypes(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == config.INPUT_FEATURES
    assert len(X) == len(y)
    assert all(np.issubdtype(t, np.floating) for t in X.dtypes)
