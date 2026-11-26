"""Feature engineering as a scikit-learn transformer, so training and serving share one code path.

The transformer is **stateless** (no bin edges learned from data), which removes the training/serving skew
risk of the Milestone 2 notebook, where quantile-based salary bands were computed on the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import MODEL_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Raw (TenureYears, PerformanceRating, MonthlySalary) -> model features.

    Engineered features follow the Milestone 2 findings:
      * ``Tenure_le2``          : attrition is 29 % in the first 2 years vs 14 % after 10 (non-linear tenure effect)
      * ``LogTenure``           : smooth, saturating tenure effect
      * ``SalaryToTenureRatio`` : same definition as the Milestone 2 notebook, ``salary / (tenure + 1)``
    Extra columns in the input (e.g. Age, Gender, JobRole in the training CSV) are ignored.
    """

    required_columns = MODEL_FEATURES

    def __init__(self, engineered: bool = True):
        self.engineered = engineered  # False -> pass the 3 raw features through unchanged

    def fit(self, X, y=None):  # noqa: D401 - sklearn API
        self._check(X)
        self.feature_names_out_ = list(self._transform_frame(X.head(2), self.engineered).columns)
        return self

    def transform(self, X) -> pd.DataFrame:
        self._check(X)
        return self._transform_frame(X, self.engineered)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    # ------------------------------------------------------------------------------------------
    def _check(self, X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame with named columns.")
        missing = [c for c in self.required_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def _transform_frame(X: pd.DataFrame, engineered: bool = True) -> pd.DataFrame:
        tenure = X["TenureYears"].astype(float)
        salary = X["MonthlySalary"].astype(float)
        out = pd.DataFrame(
            {
                "TenureYears": tenure,
                "PerformanceRating": X["PerformanceRating"].astype(float),
                "MonthlySalary": salary,
            },
            index=X.index,
        )
        if engineered:
            out["Tenure_le2"] = (tenure <= 2).astype(float)
            out["LogTenure"] = np.log1p(tenure)
            out["SalaryToTenureRatio"] = salary / (tenure + 1.0)
        return out
