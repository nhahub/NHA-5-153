"""Feature engineering + prediction service."""

import pickle

import numpy as np
import pandas as pd
import pytest

from attrition.features import FeatureEngineer
from attrition.modeling import risk_level, risk_levels, topk_metrics


def test_feature_engineer_outputs_expected_columns():
    X = pd.DataFrame({"TenureYears": [0.0, 2.0, 9.0], "PerformanceRating": [1, 3, 5],
                      "MonthlySalary": [4000.0, 6000.0, 9000.0], "Age": [30, 40, 50], "Gender": list("MFM")})
    out = FeatureEngineer().fit_transform(X)
    assert list(out.columns) == ["TenureYears", "PerformanceRating", "MonthlySalary", "Tenure_le2", "LogTenure",
                                 "SalaryToTenureRatio"]  # extra columns (Age, Gender) are ignored
    assert out["Tenure_le2"].tolist() == [1.0, 1.0, 0.0]
    assert out["SalaryToTenureRatio"].iloc[1] == pytest.approx(6000 / 3)


def test_feature_engineer_raw_mode_and_missing_columns():
    X = pd.DataFrame({"TenureYears": [1.0], "PerformanceRating": [3], "MonthlySalary": [5000.0]})
    assert FeatureEngineer(engineered=False).fit_transform(X).shape[1] == 3
    with pytest.raises(ValueError, match="Missing required columns"):
        FeatureEngineer().fit(X.drop(columns="MonthlySalary"))


def test_feature_engineer_is_picklable():
    pickle.loads(pickle.dumps(FeatureEngineer()))


def test_risk_bands_follow_thresholds():
    th = {"high": 0.30, "medium": 0.25}
    assert [risk_level(p, th) for p in (0.1, 0.25, 0.29, 0.30, 0.9)] == ["Low", "Medium", "Medium", "High", "High"]
    assert risk_levels([0.1, 0.5], th).tolist() == ["Low", "High"]


def test_topk_metrics_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.linspace(1, 0, 10)
    m = topk_metrics(y, scores, 0.2)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["lift"] == pytest.approx(5.0)


def test_service_predicts_and_explains(service):
    out = service.predict_one({"tenure_years": 0.5, "performance_rating": 5, "monthly_salary": 3000})
    assert 0 < out["attrition_probability"] < 1 and out["risk_level"] in {"Low", "Medium", "High"}
    assert len(out["top_drivers"]) == 3
    # a brand-new, top-rated, low-paid employee is riskier than a veteran
    veteran = service.predict_one({"tenure_years": 20, "performance_rating": 2, "monthly_salary": 9000})
    assert out["attrition_probability"] > veteran["attrition_probability"]
    assert out["risk_level"] == "High" and veteran["risk_level"] == "Low"


def test_service_accepts_csv_style_columns(service):
    frame = service.to_frame([{"TenureYears": 3, "PerformanceRating": 3, "MonthlySalary": 6000}])
    assert frame.shape == (1, 3)
    with pytest.raises(ValueError):
        service.to_frame([{"TenureYears": 3}])
