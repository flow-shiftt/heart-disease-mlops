"""Request/response contracts for the serving API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2,
    "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
}


class Patient(BaseModel):
    """One patient's clinical measurements (UCI Cleveland encoding)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT})

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (4 = asymptomatic)")
    trestbps: float = Field(..., ge=70, le=220, description="Resting blood pressure, mm Hg")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured (0-3); null if unknown")
    thal: Literal[3, 6, 7] | None = Field(None, description="Thallium test; null if unknown")


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not")
    label: Literal["disease", "no_disease"]
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_band: Literal["low", "moderate", "high"]
    model_name: str
    model_version: str
    request_id: str


class BatchIn(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=500)


class BatchOut(BaseModel):
    count: int
    predictions: list[PredictionOut]
