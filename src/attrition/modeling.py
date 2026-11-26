"""Candidate models, evaluation metrics and risk-band policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import PARAMS
from .features import FeatureEngineer

# Column groups of the M3 baseline (uses every column of the cleaned CSV)
_M3_NUM = ["Age", "TenureYears", "PerformanceRating", "MonthlySalary"]
_M3_CAT = ["Gender", "JobRole"]

# Lower = simpler / easier to explain. Used as the tie-breaker in model selection.
COMPLEXITY = {
    "lr_core3": 1,
    "lr_engineered": 2,
    "hgb_engineered": 3,
    "rf_engineered": 3,
    "xgb_engineered": 4,
    "m3_baseline_lr_smote_all_features": 5,
}
DEPLOYABLE = {"lr_core3", "lr_engineered", "hgb_engineered", "rf_engineered", "xgb_engineered"}


def lr_pipeline(C: float = 1.0, engineered: bool = True) -> Pipeline:
    """Logistic regression on (engineered) tenure / performance / salary features."""
    return Pipeline(
        [
            ("features", FeatureEngineer(engineered=engineered)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=2000)),
        ]
    )


def candidate_models(C: float = 1.0) -> dict:
    """Candidates compared in MLflow. ``m3_baseline...`` reproduces the Milestone 3 approach (reference only)."""
    seed = PARAMS["data"]["random_state"]
    tree_front = [("features", FeatureEngineer())]
    m3_pre = ColumnTransformer(
        [("num", StandardScaler(), _M3_NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), _M3_CAT)]
    )
    return {
        "m3_baseline_lr_smote_all_features": ImbPipeline(
            [("pre", m3_pre), ("smote", SMOTE(random_state=seed)), ("model", LogisticRegression(max_iter=1000, random_state=seed))]
        ),
        "lr_core3": lr_pipeline(C, engineered=False),
        "lr_engineered": lr_pipeline(C, engineered=True),
        "hgb_engineered": Pipeline(
            tree_front
            + [("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed))]
        ),
        "rf_engineered": Pipeline(
            tree_front
            + [("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=seed))]
        ),
        "xgb_engineered": Pipeline(
            tree_front
            + [
                (
                    "model",
                    XGBClassifier(
                        n_estimators=150, max_depth=2, learning_rate=0.05, min_child_weight=50,
                        subsample=0.8, eval_metric="logloss", random_state=seed, n_jobs=1,
                    ),
                )
            ]
        ),
    }


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------
def topk_metrics(y_true, scores, k: float) -> dict:
    """Precision / recall / lift when only the top ``k`` share of employees (by score) is flagged."""
    y = np.asarray(y_true)
    n = max(1, int(round(len(y) * k)))
    order = np.argsort(-np.asarray(scores), kind="stable")[:n]
    hits = y[order].sum()
    base = y.mean()
    precision = hits / n
    return {
        "precision": float(precision),
        "recall": float(hits / max(1, y.sum())),
        "lift": float(precision / base) if base > 0 else float("nan"),
    }


def evaluate(y_true, scores, k: float | None = None) -> dict:
    """Ranking + probability-quality metrics. ``k`` defaults to the 'High' risk share."""
    k = k if k is not None else PARAMS["risk"]["high_pct"]
    y = np.asarray(y_true)
    scores = np.asarray(scores)
    top = topk_metrics(y, scores, k)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, np.clip(scores, 1e-6, 1 - 1e-6))),
        f"precision_top{int(k * 100)}": top["precision"],
        f"recall_top{int(k * 100)}": top["recall"],
        f"lift_top{int(k * 100)}": top["lift"],
        "base_rate": float(y.mean()),
    }


def bootstrap_auc(y_true, scores, n_boot: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """AUC with a percentile bootstrap 95 % CI."""
    y, s = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(roc_auc_score(y, s)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_auc_diff(y_true, s_new, s_old, n_boot: int = 500, seed: int = 0) -> dict:
    """AUC(new) - AUC(old) on identical resamples -> point estimate and 95 % CI."""
    y, a, b = np.asarray(y_true), np.asarray(s_new), np.asarray(s_old)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return {
        "diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


# ---------------------------------------------------------------------------------------------
# Risk policy
# ---------------------------------------------------------------------------------------------
def risk_thresholds(reference_scores) -> dict:
    """Probability cut-offs for the High / Medium bands from a reference score distribution."""
    r = PARAMS["risk"]
    s = np.asarray(reference_scores)
    return {
        "high": float(np.quantile(s, 1 - r["high_pct"])),
        "medium": float(np.quantile(s, 1 - r["high_pct"] - r["medium_pct"])),
    }


def risk_level(prob: float, thresholds: dict) -> str:
    if prob >= thresholds["high"]:
        return "High"
    if prob >= thresholds["medium"]:
        return "Medium"
    return "Low"


def risk_levels(probs, thresholds: dict) -> pd.Series:
    p = pd.Series(np.asarray(probs, dtype=float))
    return pd.Series(
        np.where(p >= thresholds["high"], "High", np.where(p >= thresholds["medium"], "Medium", "Low")), index=p.index
    )
