"""Model loading and scoring, shared by the REST API, the dashboard and the monitoring/simulation tools."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .config import API_TO_MODEL, MODEL_FEATURES
from .modeling import risk_level, risk_levels

FEATURE_LABELS = {
    "TenureYears": "Tenure (years)",
    "PerformanceRating": "Performance rating",
    "MonthlySalary": "Monthly salary",
}


class ModelNotFoundError(FileNotFoundError):
    pass


class ModelService:
    """Wraps the persisted sklearn pipeline plus its model card (thresholds, version, reference data)."""

    def __init__(self, model_path: Path | None = None, card_path: Path | None = None,
                 reference_path: Path | None = None):
        model_path = Path(model_path or config.MODEL_PATH)
        card_path = Path(card_path or config.MODEL_CARD_PATH)
        reference_path = Path(reference_path or config.REFERENCE_PATH)
        if not model_path.exists() or not card_path.exists():
            raise ModelNotFoundError(
                f"Model artefacts not found ({model_path}, {card_path}). Run `python -m attrition.train` first."
            )
        self.model = joblib.load(model_path)
        self.card = json.loads(card_path.read_text(encoding="utf-8"))
        self.thresholds = self.card["risk_policy"]["thresholds"]
        self.version = self.card["model_id"]
        self._reference = pd.read_csv(reference_path) if reference_path.exists() else None
        # "typical employee" used for what-if explanations
        ref = self._reference[MODEL_FEATURES] if self._reference is not None else None
        self._baseline = (
            ref.median().to_dict() if ref is not None
            else {"TenureYears": 4.0, "PerformanceRating": 3.0, "MonthlySalary": 6000.0}
        )

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def to_frame(records: list[dict]) -> pd.DataFrame:
        """Accepts snake_case (API) or CamelCase (CSV) keys and returns the model's input frame."""
        df = pd.DataFrame(records).rename(columns=API_TO_MODEL)
        missing = [c for c in MODEL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[MODEL_FEATURES].astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[MODEL_FEATURES])[:, 1]

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        probs = self.predict_proba(X)
        return pd.DataFrame(
            {"attrition_probability": probs, "risk_level": risk_levels(probs, self.thresholds).to_numpy()},
            index=X.index,
        )

    def predict_one(self, features: dict, explain: bool = True) -> dict:
        X = self.to_frame([features])
        prob = float(self.predict_proba(X)[0])
        out = {
            "attrition_probability": round(prob, 4),
            "risk_level": risk_level(prob, self.thresholds),
            "model_version": self.version,
        }
        if explain:
            out["top_drivers"] = self.explain(X.iloc[0])
        return out

    def explain(self, row: pd.Series, top: int = 3) -> list[dict]:
        """Model-agnostic what-if attribution.

        For each feature: how much would the predicted probability change if this employee had the *typical*
        (median) value instead? Positive impact = this feature currently pushes risk up.
        """
        base = pd.DataFrame([row.to_dict()])
        p0 = float(self.predict_proba(base)[0])
        drivers = []
        for feat in MODEL_FEATURES:
            alt = base.copy()
            alt[feat] = self._baseline[feat]
            impact = p0 - float(self.predict_proba(alt)[0])
            drivers.append(
                {
                    "feature": feat,
                    "label": FEATURE_LABELS[feat],
                    "value": float(row[feat]),
                    "typical_value": float(self._baseline[feat]),
                    "impact": round(impact, 4),
                    "direction": "increases risk" if impact > 0 else "decreases risk",
                }
            )
        return sorted(drivers, key=lambda d: abs(d["impact"]), reverse=True)[:top]

    @property
    def reference(self) -> pd.DataFrame | None:
        return self._reference
