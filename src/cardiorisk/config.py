"""Central configuration: paths, column schema and feature groups.

Everything that more than one module needs to agree on lives here so that the
training pipeline, the API and the tests can never drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("CARDIORISK_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"

# Primary source plus the zipped bundle UCI serves from its new static host.
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

# The 14 attributes used by every published experiment on this dataset,
# in the order they appear in the raw file.
RAW_COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type 1-4
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl
    "restecg",   # resting ECG result 0-2
    "thalach",   # max heart rate achieved
    "exang",     # exercise induced angina
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment 1-3
    "ca",        # number of major vessels coloured by fluoroscopy 0-3
    "thal",      # 3 = normal, 6 = fixed defect, 7 = reversible defect
    "num",       # diagnosis 0 (no disease) .. 4
]
TARGET = "target"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
INPUT_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Derived by ClinicalFeatureBuilder (see features.py).
ENGINEERED_FEATURES = ["hr_reserve_pct", "bp_chol_load"]

# Physiologically plausible ranges. Used for cleaning and API validation.
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 3),
}
CATEGORY_LEVELS = {
    "cp": [1, 2, 3, 4],
    "restecg": [0, 1, 2],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "cardiorisk-heart-disease")
REGISTERED_MODEL_NAME = "cardiorisk-classifier"
