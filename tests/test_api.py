"""REST API contract, logging and metrics."""

import json

import pytest
from fastapi.testclient import TestClient

from attrition import config
from attrition.api import create_app
from attrition.data import read_jsonl

GOOD = {"employee_id": "E-1", "tenure_years": 1.0, "performance_rating": 4, "monthly_salary": 5000.0}


@pytest.fixture()
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_health_and_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True
    info = client.get("/model/info").json()
    assert info["model_id"] == "vtest" and "thresholds" in info["risk_policy"]


def test_health_degraded_without_model(isolated, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", isolated / "missing.joblib")
    with TestClient(create_app()) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=GOOD).status_code == 503


def test_predict_returns_probability_risk_and_drivers(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["attrition_probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert body["employee_id"] == "E-1" and body["model_version"] == "vtest" and body["request_id"]
    assert len(body["top_drivers"]) == 3


@pytest.mark.parametrize("patch", [
    {"performance_rating": 6}, {"performance_rating": 0}, {"tenure_years": -1}, {"tenure_years": 80},
    {"monthly_salary": 0}, {"monthly_salary": -5}, {"monthly_salary": 1e9}, {"tenure_years": "abc"},
    {"gender": "F"},  # unknown fields are rejected (the model does not use gender / age)
])
def test_predict_rejects_invalid_input(client, patch):
    assert client.post("/predict", json={**GOOD, **patch}).status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "monthly_salary"}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_prediction_and_limits(client):
    r = client.post("/predict/batch", json={"employees": [GOOD, {**GOOD, "tenure_years": 15}]})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert r.json()["predictions"][0]["top_drivers"] is None  # explanations are opt-in for batches
    assert client.post("/predict/batch", json={"employees": []}).status_code == 422


def test_predictions_are_logged_with_features(client):
    pid = client.post("/predict", json=GOOD).json()["request_id"]
    rows = read_jsonl(config.PREDICTION_LOG)
    assert len(rows) == 1 and rows[0]["request_id"] == pid
    assert rows[0]["TenureYears"] == 1.0 and rows[0]["risk_level"] in {"Low", "Medium", "High"}


def test_outcome_feedback_is_recorded_and_joinable(client):
    from attrition.data import load_labelled_predictions

    pid = client.post("/predict", json=GOOD).json()["request_id"]
    assert client.post("/outcomes", json={"request_id": pid, "left": True}).status_code == 201
    joined = load_labelled_predictions()
    assert len(joined) == 1 and int(joined["y"].iloc[0]) == 1


def test_metrics_endpoint_counts_traffic(client):
    client.post("/predict", json=GOOD)
    client.post("/predict", json={**GOOD, "performance_rating": 9})
    text = client.get("/metrics").text
    assert 'attrition_http_requests_total{path="/predict",status="200"} 1' in text
    assert 'attrition_http_requests_total{path="/predict",status="422"} 1' in text
    assert "attrition_request_latency_seconds_count" in text and 'attrition_model_info{version="vtest"} 1' in text
    assert sum(int(l.split()[-1]) for l in text.splitlines() if l.startswith("attrition_predictions_total{")) == 1


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/predict", json=GOOD).status_code == 401
    assert client.post("/predict", json=GOOD, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict", json=GOOD, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for orchestrators


def test_logging_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("DISABLE_PREDICTION_LOGGING", "true")
    client.post("/predict", json=GOOD)
    assert not config.PREDICTION_LOG.exists()


def test_monitoring_report_endpoint(client):
    assert client.get("/monitoring/report").status_code == 404
    config.LATEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
    config.LATEST_REPORT.write_text(json.dumps({"overall_status": "ok"}), encoding="utf-8")
    assert client.get("/monitoring/report").json() == {"overall_status": "ok"}
