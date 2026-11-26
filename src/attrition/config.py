"""Central configuration: paths, schema constants and tunables from ``params.yaml``.

Every path can be overridden with an environment variable so the same code runs locally,
in Docker and in CI without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("ATTRITION_HOME", Path(__file__).resolve().parents[2])).resolve()

# --- data ---------------------------------------------------------------------------------------
DATA_PATH = Path(os.environ.get("ATTRITION_DATA", ROOT / "employee_attriton_data_clean.csv"))
TARGET = "Attrition"  # "Yes" / "No" in the cleaned CSV

# Features the deployed model needs. Age, Gender and JobRole are deliberately NOT used:
# Milestone 2 statistical tests (chi-square, ANOVA, logistic LR tests) and Milestone 3/4 benchmarks show
# they add no predictive signal, and excluding protected attributes avoids discriminatory scoring.
MODEL_FEATURES = ["TenureYears", "PerformanceRating", "MonthlySalary"]

# API/JSON (snake_case) <-> model/CSV (CamelCase) column names
API_TO_MODEL = {
    "tenure_years": "TenureYears",
    "performance_rating": "PerformanceRating",
    "monthly_salary": "MonthlySalary",
}
MODEL_TO_API = {v: k for k, v in API_TO_MODEL.items()}

# Hard validation limits shared by the API, the dashboard and the data-quality monitor
FEATURE_LIMITS = {
    "TenureYears": (0.0, 60.0),
    "PerformanceRating": (1, 5),
    "MonthlySalary": (0.0, 100_000.0),
}

# --- artefacts ----------------------------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "attrition_model.joblib"))
MODEL_CARD_PATH = Path(os.environ.get("MODEL_CARD_PATH", MODELS_DIR / "model_card.json"))
REFERENCE_PATH = Path(os.environ.get("REFERENCE_PATH", MODELS_DIR / "reference_sample.csv.gz"))
ARCHIVE_DIR = MODELS_DIR / "archive"

# --- monitoring ---------------------------------------------------------------------------------
MONITORING_DIR = Path(os.environ.get("MONITORING_DIR", ROOT / "monitoring"))
PREDICTION_LOG = MONITORING_DIR / "logs" / "predictions.jsonl"
OUTCOME_LOG = MONITORING_DIR / "logs" / "outcomes.jsonl"
ALERT_LOG = MONITORING_DIR / "alerts.jsonl"
REPORT_DIR = MONITORING_DIR / "reports"
LATEST_REPORT = REPORT_DIR / "latest.json"

# --- MLflow -------------------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
MLFLOW_ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", (ROOT / "mlartifacts").as_uri())
MLFLOW_EXPERIMENT = "employee-attrition"
REGISTERED_MODEL_NAME = "employee-attrition-classifier"

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")  # Slack-compatible incoming webhook


def load_params(path: Path | None = None) -> dict:
    """Load ``params.yaml`` (tunables shared by training, monitoring and retraining)."""
    path = path or Path(os.environ.get("PARAMS_PATH", ROOT / "params.yaml"))
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


PARAMS = load_params()
