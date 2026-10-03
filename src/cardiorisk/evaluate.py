"""Metrics and diagnostic plots shared by training, notebooks and tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_roc(model, X, y, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, color="#c0392b")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title(title)
    return _save(fig, path)


def plot_pr(model, X, y, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax, color="#2c3e50")
    ax.set_title(title)
    return _save(fig, path)


def plot_confusion(model, X, y, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=["No disease", "Disease"], cmap="Reds", ax=ax, colorbar=False
    )
    ax.set_title(title)
    return _save(fig, path)


def feature_importance(pipeline, feature_names: list[str]) -> pd.Series | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return None
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def plot_importance(importance: pd.Series, path: Path, title: str, top: int = 15) -> Path:
    data = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(data) + 1))
    ax.barh(data.index, data.values, color="#c0392b")
    ax.set_title(title)
    ax.set_xlabel("importance (|coef| for linear models)")
    return _save(fig, path)


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> Path:
    """Grouped bar chart of CV mean metrics per model family."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(summary)
    x = np.arange(len(metrics))
    palette = ["#c0392b", "#2c3e50", "#16a085", "#8e44ad"]
    for i, (name, row) in enumerate(summary.iterrows()):
        means = [row[f"cv_{m}_mean"] for m in metrics]
        stds = [row[f"cv_{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=name,
               color=palette[i % len(palette)])
    ax.set_xticks(x + width * (len(summary) - 1) / 2, metrics)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold CV score")
    ax.set_title("Model comparison (mean ± std across folds)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
