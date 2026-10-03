"""Train, tune, compare and package heart-disease classifiers with MLflow tracking.

Run hierarchy in MLflow:
    model-selection-<timestamp>          (parent: data hash, comparison chart, winner)
      ├── logistic_regression            (best params, CV + hold-out metrics, plots, model)
      │     ├── logistic_regression-cand-00   (one run per grid candidate)
      │     └── ...
      ├── random_forest
      └── gradient_boosting
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from cardiorisk import __version__, config
from cardiorisk.data import load_clean, prepare, sha256, split_xy
from cardiorisk.evaluate import (
    SCORING,
    classification_metrics,
    feature_importance,
    plot_confusion,
    plot_importance,
    plot_model_comparison,
    plot_pr,
    plot_roc,
)
from cardiorisk.features import build_pipeline, transformed_feature_names

log = logging.getLogger(__name__)

# MLflow >=3 serialises sklearn models with skops, which refuses unknown classes.
# We explicitly allow-list our own transformer instead of falling back to pickle.
SKOPS_TRUSTED = [
    "cardiorisk.features.ClinicalFeatureBuilder",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",  # files are produced by this pipeline, not third parties
]


def candidate_models(quick: bool = False) -> dict[str, tuple[object, dict]]:
    """Model families and their search spaces.

    Grids are deliberately small: with ~300 rows, wide grids mostly fit noise
    in the CV estimate. `quick` shrinks them further for CI smoke runs.
    """
    rs = config.RANDOM_STATE
    spaces = {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=2000, random_state=rs),
            {
                "model__C": [0.01, 0.1, 0.3, 1.0, 3.0],
                # sklearn>=1.8: penalty is expressed via l1_ratio (0 = ridge/L2, 1 = lasso/L1)
                "model__l1_ratio": [0.0, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [4, 6, None],
                "model__min_samples_leaf": [2, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8],
            },
        ),
    }
    if quick:
        spaces = {name: (est, {k: v[:1] for k, v in grid.items()}) for name, (est, grid) in spaces.items()}
    return spaces


def _short(params: dict) -> dict:
    return {k.replace("model__", ""): ("None" if v is None else v) for k, v in params.items()}


def _log_candidates(name: str, search: GridSearchCV) -> None:
    res = pd.DataFrame(search.cv_results_)
    for i, row in res.iterrows():
        with mlflow.start_run(run_name=f"{name}-cand-{i:02d}", nested=True):
            mlflow.set_tags({"model_family": name, "run_type": "grid_candidate"})
            mlflow.log_params(_short(row["params"]))
            mlflow.log_metrics(
                {f"cv_{m}_mean": float(row[f"mean_test_{m}"]) for m in SCORING}
                | {f"cv_{m}_std": float(row[f"std_test_{m}"]) for m in SCORING}
                | {"rank_roc_auc": int(row["rank_test_roc_auc"])}
            )


def train(quick: bool = False, register: bool = True) -> dict:
    if not config.CLEAN_FILE.exists():
        prepare()
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    data_hash = sha256(config.CLEAN_FILE)

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    with mlflow.start_run(run_name=f"model-selection-{stamp}") as parent:
        mlflow.set_tags({"run_type": "selection", "quick_mode": str(quick)})
        mlflow.log_params(
            {
                "n_rows": len(df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "cv_folds": config.CV_FOLDS,
                "test_size": config.TEST_SIZE,
                "random_state": config.RANDOM_STATE,
                "data_sha256": data_hash[:16],
                "positive_rate": round(float(y.mean()), 4),
            }
        )
        mlflow.log_artifact(str(config.CLEAN_FILE), artifact_path="data")

        for name, (estimator, grid) in candidate_models(quick).items():
            log.info("Tuning %s over %d candidates", name, len(list(_grid_iter(grid))))
            search = GridSearchCV(
                build_pipeline(estimator),
                grid,
                scoring=SCORING,
                refit="roc_auc",
                cv=cv,
                n_jobs=-1,
                return_train_score=False,
            )
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"model_family": name, "run_type": "family_best"})
                search.fit(X_train, y_train)
                _log_candidates(name, search)

                best = search.best_estimator_
                i = search.best_index_
                cvr = search.cv_results_
                cv_metrics = {f"cv_{m}_mean": float(cvr[f"mean_test_{m}"][i]) for m in SCORING}
                cv_metrics |= {f"cv_{m}_std": float(cvr[f"std_test_{m}"][i]) for m in SCORING}

                proba = best.predict_proba(X_test)[:, 1]
                test_metrics = classification_metrics(y_test, (proba >= 0.5).astype(int), proba)

                mlflow.log_params(_short(search.best_params_))
                mlflow.log_metrics(cv_metrics | {f"test_{k}": v for k, v in test_metrics.items()})

                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    pd.DataFrame(cvr).drop(columns=["params"]).to_csv(tmp / "cv_results.csv", index=False)
                    plot_roc(best, X_test, y_test, tmp / "roc_curve.png", f"{name} – ROC (hold-out)")
                    plot_pr(best, X_test, y_test, tmp / "pr_curve.png", f"{name} – PR (hold-out)")
                    plot_confusion(best, X_test, y_test, tmp / "confusion_matrix.png", f"{name}")
                    imp = feature_importance(best, transformed_feature_names(best))
                    if imp is not None:
                        imp.to_csv(tmp / "feature_importance.csv", header=["importance"])
                        plot_importance(imp, tmp / "feature_importance.png", f"{name} – top features")
                    mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

                signature = infer_signature(X_train, best.predict_proba(X_train))
                mlflow.sklearn.log_model(
                    best,
                    name="model",
                    signature=signature,
                    input_example=X_train.head(3),
                    skops_trusted_types=SKOPS_TRUSTED,
                )

            results[name] = {
                "best_params": _short(search.best_params_),
                **cv_metrics,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            fitted[name] = best
            log.info(
                "%s: CV ROC-AUC %.3f ± %.3f | test ROC-AUC %.3f",
                name,
                cv_metrics["cv_roc_auc_mean"],
                cv_metrics["cv_roc_auc_std"],
                test_metrics["roc_auc"],
            )

        summary = pd.DataFrame(results).T
        # Selection rule: highest mean CV ROC-AUC. The hold-out set is reported but
        # never used to pick the winner, so its numbers stay an unbiased estimate.
        winner = summary["cv_roc_auc_mean"].astype(float).idxmax()
        final_model = fitted[winner]

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            summary.drop(columns=["best_params"]).astype(float).round(4).to_csv(tmp / "model_comparison.csv")
            plot_model_comparison(summary, tmp / "model_comparison.png")
            mlflow.log_artifacts(str(tmp), artifact_path="comparison")
            plot_model_comparison(summary, config.FIGURES_DIR / "07_model_comparison.png")
            summary.drop(columns=["best_params"]).astype(float).round(4).to_csv(
                config.FIGURES_DIR / "model_comparison.csv"
            )

        mlflow.set_tag("winner", winner)
        mlflow.log_metric("winner_cv_roc_auc", float(summary.loc[winner, "cv_roc_auc_mean"]))
        mlflow.log_metric("winner_test_roc_auc", float(summary.loc[winner, "test_roc_auc"]))

        metadata = _package(final_model, winner, results[winner], X_train, data_hash, parent.info.run_id)
        mlflow.log_artifact(str(config.MODELS_DIR / "metadata.json"), artifact_path="package")

        if register:
            signature = infer_signature(X_train, final_model.predict_proba(X_train))
            mlflow.sklearn.log_model(
                final_model,
                name="final_model",
                signature=signature,
                input_example=X_train.head(3),
                skops_trusted_types=SKOPS_TRUSTED,
                registered_model_name=config.REGISTERED_MODEL_NAME,
            )

    log.info("Winner: %s -> %s", winner, config.MODELS_DIR)
    return {"winner": winner, "results": results, "metadata": metadata}


def _grid_iter(grid: dict):
    from sklearn.model_selection import ParameterGrid

    return ParameterGrid(grid)


def _package(model, name: str, result: dict, X_train: pd.DataFrame, data_hash: str, run_id: str) -> dict:
    """Persist the winning pipeline as joblib + MLflow model + metadata.json."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "model.joblib")

    mlflow_dir = config.MODELS_DIR / "mlflow_model"
    if mlflow_dir.exists():
        import shutil

        shutil.rmtree(mlflow_dir)
    mlflow.sklearn.save_model(
        model,
        str(mlflow_dir),
        signature=infer_signature(X_train, model.predict_proba(X_train)),
        input_example=X_train.head(3),
        skops_trusted_types=SKOPS_TRUSTED,
    )

    metadata = {
        "model_name": name,
        "model_version": f"{__version__}+{datetime.now(UTC):%Y%m%d%H%M}",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "data_sha256": data_hash,
        "features": config.INPUT_FEATURES,
        "engineered_features": config.ENGINEERED_FEATURES,
        "decision_threshold": 0.5,
        "best_params": result["best_params"],
        "cv_metrics": {k: round(v, 4) for k, v in result.items() if k.startswith("cv_")},
        "test_metrics": {k: round(v, 4) for k, v in result.items() if k.startswith("test_")},
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and package the heart-disease model")
    parser.add_argument("--quick", action="store_true", help="tiny grids (CI smoke run)")
    parser.add_argument("--no-register", action="store_true", help="skip MLflow model registry")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = train(quick=args.quick, register=not args.no_register)
    print(
        json.dumps(
            {
                "winner": out["winner"],
                "test_metrics": out["metadata"]["test_metrics"],
                "cv_metrics": out["metadata"]["cv_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
