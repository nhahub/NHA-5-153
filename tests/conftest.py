"""Shared fixtures: a quickly-trained model and fully isolated file locations (no test touches the repo)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

import joblib
import pytest

from attrition import config
from attrition.config import MODEL_FEATURES, PARAMS
from attrition.data import load_training_data, split_data
from attrition.modeling import evaluate, lr_pipeline, risk_thresholds
from attrition.predictor import ModelService
from attrition.train import build_reference


@pytest.fixture(scope="session")
def data_split():
    df = load_training_data()
    train_df, test_df = split_data(df)
    return df, train_df, test_df


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, data_split):
    """A real (logistic regression) model + card + reference sample saved the same way train.py saves them."""
    _, train_df, test_df = data_split
    pipe = lr_pipeline(1.0, engineered=True).fit(train_df[MODEL_FEATURES], train_df["y"])
    oof = pipe.predict_proba(train_df[MODEL_FEATURES])[:, 1]
    card = {
        "model_id": "vtest", "trained_at": datetime.now(timezone.utc).isoformat(), "algorithm": "lr_engineered",
        "risk_policy": {**PARAMS["risk"], "thresholds": risk_thresholds(oof)},
        "metrics": {"test": evaluate(test_df["y"], pipe.predict_proba(test_df[MODEL_FEATURES])[:, 1]),
                    "cv": {}, "test_roc_auc_ci95": [0.56, 0.59]},
        "limitations": ["test"], "training_data": {"rows": len(train_df)}, "mlflow": {"registered_version": None},
        "hyperparameters": {}, "features": {"inputs": MODEL_FEATURES},
    }
    d = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, d / "attrition_model.joblib")
    (d / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    build_reference(train_df, train_df["y"].to_numpy(), oof, 10000, 42).to_csv(
        d / "reference_sample.csv.gz", index=False, compression="gzip")
    return d


@pytest.fixture()
def isolated(tmp_path, monkeypatch, model_dir):
    """Point every path / tracking URI of the package at a throw-away directory."""
    models = tmp_path / "models"
    shutil.copytree(model_dir, models)
    mon = tmp_path / "monitoring"
    patch = {
        "MODELS_DIR": models, "MODEL_PATH": models / "attrition_model.joblib",
        "MODEL_CARD_PATH": models / "model_card.json", "REFERENCE_PATH": models / "reference_sample.csv.gz",
        "ARCHIVE_DIR": models / "archive", "MONITORING_DIR": mon,
        "PREDICTION_LOG": mon / "logs" / "predictions.jsonl", "OUTCOME_LOG": mon / "logs" / "outcomes.jsonl",
        "ALERT_LOG": mon / "alerts.jsonl", "REPORT_DIR": mon / "reports",
        "LATEST_REPORT": mon / "reports" / "latest.json", "ALERT_WEBHOOK_URL": "",
        "MLFLOW_TRACKING_URI": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "MLFLOW_ARTIFACT_ROOT": (tmp_path / "mlartifacts").as_uri(),
    }
    for k, v in patch.items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.delenv("API_KEY", raising=False)
    return tmp_path


@pytest.fixture()
def service(isolated):
    return ModelService()
