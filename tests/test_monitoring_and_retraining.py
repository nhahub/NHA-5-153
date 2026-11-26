"""Drift statistics, monitoring scenarios (stable / covariate shift / concept drift) and retraining gate."""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from attrition import config, retrain
from attrition.config import MODEL_FEATURES
from attrition.data import read_jsonl
from attrition.monitoring import drift
from attrition.monitoring.monitor import run_monitoring, to_markdown
from attrition.monitoring.simulate import simulate

P = {"psi_warn": 0.10, "psi_critical": 0.25, "ks_alpha": 0.001}


# ------------------------------------------------------------------ drift statistics
def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert drift.psi_numeric(a, rng.normal(0, 1, 20000)) < 0.02
    assert drift.psi_numeric(a, rng.normal(1.0, 1, 20000)) > 0.25
    assert drift.psi_categorical([1] * 50 + [2] * 50, [1] * 50 + [2] * 50) == pytest.approx(0, abs=1e-6)
    assert drift.psi_categorical([1] * 90 + [2] * 10, [1] * 10 + [2] * 90) > 0.25


def test_numeric_drift_status_requires_psi_and_significance():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    assert drift.numeric_drift("x", ref, rng.normal(0, 1, 500), P)["status"] == "ok"
    assert drift.numeric_drift("x", ref, rng.normal(1.5, 1, 500), P)["status"] == "critical"
    assert drift.numeric_drift("x", ref, rng.normal(0.35, 1, 5000), P)["status"] in {"warning", "critical"}


def test_binomial_share_test():
    assert drift.binomial_share_test(0.20, 1000, 0.20) == pytest.approx(1.0)
    assert drift.binomial_share_test(0.35, 1000, 0.20) < 1e-6


# ------------------------------------------------------------------ monitoring scenarios
def _monitor(service, scenario, data_split, n=3000, notify=False):
    _, _, test_df = data_split
    simulate(scenario, n=n, service=service, out_dir=config.MONITORING_DIR, source=test_df)
    return run_monitoring(window_hours=168, notify=notify)


def _by_name(report):
    return {c["check"]: c for c in report["checks"]}


def test_stable_traffic_raises_no_warning(service, data_split):
    rep = _monitor(service, "stable", data_split)
    bad = [c for c in rep["checks"] if c["status"] in ("warning", "critical")]
    assert rep["overall_status"] == "ok", bad
    assert not rep["recommend_retraining"]
    assert not config.ALERT_LOG.exists()
    assert rep["n_predictions"] == 3000 and rep["n_labelled"] > 1500


def test_covariate_shift_triggers_feature_and_prediction_drift(service, data_split):
    rep = _monitor(service, "covariate_shift", data_split)
    c = _by_name(rep)
    assert c["feature_drift:TenureYears"]["status"] == "critical"
    assert c["feature_drift:MonthlySalary"]["status"] in {"warning", "critical"}
    assert c["prediction_drift:high_risk_share"]["status"] in {"warning", "critical"}
    assert rep["overall_status"] == "critical" and rep["recommend_retraining"]
    alerts = read_jsonl(config.ALERT_LOG)
    assert any(a["severity"] == "CRITICAL" and "TenureYears" in a["message"] for a in alerts)
    assert config.LATEST_REPORT.exists() and (config.REPORT_DIR / "latest.md").exists()


def test_concept_drift_is_caught_by_performance_monitoring(service, data_split):
    rep = _monitor(service, "concept_drift", data_split)
    c = _by_name(rep)
    assert c["performance:roc_auc"]["status"] == "critical"
    assert c["performance:calibration"]["status"] == "warning"
    assert c["feature_drift:TenureYears"]["status"] == "ok"  # inputs did NOT change: only performance reveals it
    assert rep["recommend_retraining"]


def test_monitoring_handles_empty_and_sparse_logs(service):
    rep = run_monitoring(window_hours=168, notify=False)
    assert rep["n_predictions"] == 0 and rep["overall_status"] in {"ok"}
    md = to_markdown(rep)
    assert "Overall status" in md


def test_traffic_volume_drop_and_model_age(service, data_split, monkeypatch):
    _, _, test_df = data_split
    now = datetime.now(timezone.utc)
    old = now - pd.Timedelta(hours=170)
    simulate("stable", n=1000, days=1, service=service, out_dir=config.MONITORING_DIR, source=test_df,
             now=old)  # previous window busy ...
    simulate("stable", n=250, days=1, service=service, out_dir=config.MONITORING_DIR, source=test_df,
             now=now)  # ... current window quiet
    rep = run_monitoring(window_hours=168, notify=False, now=now)
    assert _by_name(rep)["traffic_volume"]["status"] in {"ok", "warning"}
    card = json.loads(config.MODEL_CARD_PATH.read_text())
    card["trained_at"] = (now - pd.Timedelta(days=200)).isoformat()
    config.MODEL_CARD_PATH.write_text(json.dumps(card))
    rep = run_monitoring(window_hours=168, notify=False, now=now)
    assert _by_name(rep)["model_age"]["status"] == "warning" and rep["recommend_retraining"]


def test_webhook_receives_alert(service, data_split, monkeypatch):
    sent = []
    monkeypatch.setattr(config, "ALERT_WEBHOOK_URL", "http://hooks.example/test")
    monkeypatch.setattr("attrition.monitoring.monitor.requests.post",
                        lambda url, json=None, timeout=None: sent.append((url, json)))
    _monitor(service, "covariate_shift", data_split, n=1500, notify=True)
    assert sent and "CRITICAL" in sent[0][1]["text"] and "Retraining recommended" in sent[0][1]["text"]


# ------------------------------------------------------------------ retraining
@pytest.fixture()
def fast_retrain(monkeypatch):
    monkeypatch.setattr(retrain, "CHALLENGERS", ["lr_core3"])
    monkeypatch.setitem(retrain.PARAMS["retraining"], "bootstrap_iterations", 100)


def test_no_trigger_means_no_retraining(service, fast_retrain):
    res = retrain.retrain()
    assert res["outcome"].startswith("skipped") and not res["promoted"]


def test_equal_challenger_is_rejected(service, fast_retrain):
    res = retrain.retrain(force=True, folds=3)
    assert not res["promoted"] and res["gate"]["improves_enough"] is False
    assert "champion kept" in res["outcome"]


def test_dry_run_never_promotes(service, fast_retrain, isolated):
    # champion = coin flip, so the challenger would win ...
    pipe = joblib.load(config.MODEL_PATH)
    pipe.named_steps["model"].coef_ = pipe.named_steps["model"].coef_ * 0
    joblib.dump(pipe, config.MODEL_PATH)
    res = retrain.retrain(force=True, dry_run=True, folds=3)
    assert res["gate"]["improves_enough"] and not res["promoted"] and "dry-run" in res["outcome"]
    assert json.loads(config.MODEL_CARD_PATH.read_text())["model_id"] == "vtest"


def test_better_challenger_is_promoted_archived_and_rollback_works(service, fast_retrain, isolated):
    pipe = joblib.load(config.MODEL_PATH)
    pipe.named_steps["model"].coef_ = pipe.named_steps["model"].coef_ * 0  # useless champion (AUC 0.5)
    joblib.dump(pipe, config.MODEL_PATH)
    res = retrain.retrain(force=True, folds=3)
    assert res["promoted"], res
    card = json.loads(config.MODEL_CARD_PATH.read_text())
    assert card["model_id"] != "vtest" and card["parent_model_id"] == "vtest"
    assert (config.ARCHIVE_DIR / "vtest" / "attrition_model.joblib").exists()
    assert retrain.rollback() == "vtest"
    assert json.loads(config.MODEL_CARD_PATH.read_text())["model_id"] == "vtest"
