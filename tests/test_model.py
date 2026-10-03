import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from cardiorisk.data import split_xy
from cardiorisk.evaluate import classification_metrics, feature_importance
from cardiorisk.features import build_pipeline, transformed_feature_names
from cardiorisk.predict import RiskModel
from cardiorisk.train import candidate_models


def test_both_model_families_learn_signal(clean_df):
    X, y = split_xy(clean_df)
    for est in (LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=50, random_state=0)):
        auc = cross_val_score(build_pipeline(est), X, y, cv=3, scoring="roc_auc").mean()
        assert auc > 0.7, f"{type(est).__name__} AUC {auc:.2f} too low"


def test_pipeline_probabilities_are_valid(clean_df):
    X, y = split_xy(clean_df)
    proba = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_metrics_dict_complete():
    m = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert m["accuracy"] == 0.75 and m["roc_auc"] == 1.0


def test_feature_importance_aligned_with_names(clean_df):
    X, y = split_xy(clean_df)
    pipe = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    imp = feature_importance(pipe, transformed_feature_names(pipe))
    assert len(imp) == len(transformed_feature_names(pipe))
    assert "hr_reserve_pct" in imp.index


def test_search_spaces_cover_required_models():
    spaces = candidate_models()
    assert {"logistic_regression", "random_forest"} <= set(spaces)
    quick = candidate_models(quick=True)
    assert all(len(v) == 1 for _, grid in quick.values() for v in grid.values())


def test_saved_pipeline_round_trip(model_dir, clean_df, sample_patient):
    X, _ = split_xy(clean_df)
    reloaded = joblib.load(model_dir / "model.joblib")
    assert reloaded.predict_proba(X.head(5)).shape == (5, 2)

    model = RiskModel(model_dir)
    [pred] = model.predict([sample_patient])
    assert pred.prediction in (0, 1)
    assert 0.5 <= pred.confidence <= 1.0
    assert pred.risk_band in {"low", "moderate", "high"}


def test_predict_handles_missing_optional_fields(model_dir, sample_patient):
    patient = {**sample_patient, "ca": None, "thal": None}
    [pred] = RiskModel(model_dir).predict([patient])
    assert 0 <= pred.probability_disease <= 1
