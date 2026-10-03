"""Model loading and inference helpers shared by the API and the CLI."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from cardiorisk import config


@dataclass
class Prediction:
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_band: str


class RiskModel:
    """Thin wrapper around the persisted sklearn pipeline."""

    def __init__(self, model_dir: Path | str | None = None):
        model_dir = Path(model_dir or os.getenv("MODEL_DIR", config.MODELS_DIR))
        self.pipeline = joblib.load(model_dir / "model.joblib")
        meta_path = model_dir / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.threshold = float(self.metadata.get("decision_threshold", 0.5))

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    @property
    def name(self) -> str:
        return self.metadata.get("model_name", type(self.pipeline.named_steps["model"]).__name__)

    def predict(self, records: list[dict]) -> list[Prediction]:
        frame = pd.DataFrame(records, columns=config.INPUT_FEATURES).astype("float64")
        proba = self.pipeline.predict_proba(frame)[:, 1]
        out = []
        for p in proba:
            pred = int(p >= self.threshold)
            out.append(
                Prediction(
                    prediction=pred,
                    label="disease" if pred else "no_disease",
                    probability_disease=round(float(p), 4),
                    confidence=round(float(p if pred else 1 - p), 4),
                    risk_band=_band(float(p)),
                )
            )
        return out


def _band(p: float) -> str:
    if p < 0.3:
        return "low"
    if p < 0.6:
        return "moderate"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score patients from a JSON file")
    parser.add_argument("input", type=Path, help="JSON object or list of objects")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    records = payload if isinstance(payload, list) else [payload]
    model = RiskModel()
    print(json.dumps([p.__dict__ for p in model.predict(records)], indent=2))


if __name__ == "__main__":
    main()
