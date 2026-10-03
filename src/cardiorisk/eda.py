"""Exploratory data analysis figures (used by the EDA notebook and CI)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from cardiorisk import config  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = {0: "#2c3e50", 1: "#c0392b"}
LABELS = {0: "No disease", 1: "Disease"}

CATEGORY_NAMES = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal", 4: "asymptomatic"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "restecg": {0: "normal", 1: "ST-T abnormal", 2: "LV hypertrophy"},
}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    counts = df[config.TARGET].map(LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]])
    for bar, n in zip(bars, counts.values, strict=False):
        ax.annotate(f"{n} ({n / counts.sum():.0%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom")
    ax.set_title("Class balance")
    ax.set_ylabel("patients")
    return _save(fig, out_dir, "01_class_balance.png")


def numeric_histograms(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        for cls in (0, 1):
            ax.hist(df.loc[df[config.TARGET] == cls, col].dropna(), bins=18, alpha=0.6,
                    color=PALETTE[cls], label=LABELS[cls])
        ax.set_title(col)
    axes[0, 0].legend()
    fig.suptitle("Numeric feature distributions by diagnosis")
    return _save(fig, out_dir, "02_numeric_histograms.png")


def correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df.astype("float64").corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Spearman correlation (lower triangle)")
    return _save(fig, out_dir, "03_correlation_heatmap.png")


def categorical_rates(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["cp", "thal", "slope", "exang", "sex", "restecg"]
    names = {**CATEGORY_NAMES, "exang": {0: "no", 1: "yes"}, "sex": {0: "female", 1: "male"}}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        labels = [f"{names[col].get(int(i), i)}\n(n={n})" for i, n in zip(rate.index, rate["size"], strict=False)]
        ax.bar(labels, rate["mean"], color="#c0392b")
        ax.axhline(df[config.TARGET].mean(), ls="--", c="grey", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Disease rate by {col}")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Disease prevalence per category (dashed = overall rate)")
    return _save(fig, out_dir, "04_categorical_rates.png")


def missing_values(raw: pd.DataFrame, out_dir: Path) -> Path:
    miss = raw.isna().sum()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(5, 3))
    if miss.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center")
    else:
        ax.bar(miss.index, miss.values, color="#8e44ad")
        for i, v in enumerate(miss.values):
            ax.annotate(str(v), (i, v), ha="center", va="bottom")
    ax.set_title("Missing values in raw data ('?' markers)")
    ax.set_ylabel("rows")
    return _save(fig, out_dir, "05_missing_values.png")


def boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    cols = ["thalach", "oldpeak", "age", "chol"]
    long = df[cols + [config.TARGET]].melt(id_vars=config.TARGET)
    long[config.TARGET] = long[config.TARGET].map(LABELS)
    g = sns.catplot(data=long, x=config.TARGET, y="value", col="variable", kind="box",
                    sharey=False, height=3.2, aspect=0.8,
                    palette=[PALETTE[0], PALETTE[1]], hue=config.TARGET, legend=False)
    g.set_titles("{col_name}")
    g.set_xlabels("")
    g.figure.suptitle("Outliers and separation of key numeric features", y=1.04)
    return _save(g.figure, out_dir, "06_boxplots.png")


def run_all(raw: pd.DataFrame, clean: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    paths = [
        class_balance(clean, out_dir),
        numeric_histograms(clean, out_dir),
        correlation_heatmap(clean, out_dir),
        categorical_rates(clean, out_dir),
        missing_values(raw, out_dir),
        boxplots(clean, out_dir),
    ]
    log.info("Wrote %d EDA figures to %s", len(paths), out_dir)
    return paths
