"""Training pipeline with MLflow experiment tracking and model registry.

    python -m attrition.train                 # full model selection + packaging
    python -m attrition.train --quick         # 3-fold CV, fewer candidates (smoke test)

What it does
1. Loads the cleaned Milestone 1 dataset and reproduces the Milestone 3 train/test split.
2. Compares candidate models with stratified CV on the **training split only** - each one is an MLflow
   *nested run* (params, CV metrics, timing). A Milestone 3 re-implementation (SMOTE + all features) is
   included as a reference so improvements are measured, not asserted.
3. Picks the champion with an explicit rule: best CV ROC-AUC, ties (within ``selection_tolerance``) broken
   towards the simplest model.
4. Evaluates the champion once on the untouched test split (with bootstrap CI), derives the High/Medium/Low
   risk thresholds, and saves model + model card + drift-reference sample.
5. Logs everything to MLflow and registers the model (alias ``champion``).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from . import __version__, config
from .config import MODEL_FEATURES, PARAMS, TARGET
from .data import file_sha256, load_training_data, split_data
from .modeling import (
    COMPLEXITY,
    DEPLOYABLE,
    bootstrap_auc,
    candidate_models,
    evaluate,
    risk_thresholds,
)

FIG_DIR = config.ROOT / "reports" / "mlops" / "figures"

LIMITATIONS = [
    "Predictive signal in the available features is weak (test ROC-AUC ~0.58). The model ranks employees by "
    "relative risk; it must not be used for individual employment decisions.",
    "Tenure, salary and performance rating are the only informative features; satisfaction, workload, manager "
    "quality and market conditions are not in the data.",
    "Probabilities are calibrated to the 25.6% historical attrition rate; a changed base rate requires recalibration.",
    "Age, gender and job role are excluded on purpose (no signal, and protected attributes should not drive scores).",
]


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - git may be missing in containers
        return "unknown"


def new_model_id() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def scalar_params(pipe) -> dict:
    model = pipe.named_steps["model"]
    return {
        f"{type(model).__name__}.{k}": v
        for k, v in model.get_params().items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate input columns (drop only target columns); pipelines pick what they need."""
    return df.drop(columns=[c for c in ("y", TARGET) if c in df.columns])


def ensure_experiment() -> str:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        return mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    return exp.experiment_id


@dataclass
class CandidateResult:
    name: str
    estimator: object  # unfitted pipeline with the selected hyper-parameters
    cv: dict
    oof_scores: np.ndarray
    fold_aucs: list
    params: dict
    seconds: float
    tuning: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Cross-validated comparison
# ---------------------------------------------------------------------------------------------
def oof_predict(est, X: pd.DataFrame, y: np.ndarray, folds: int, seed: int):
    """Out-of-fold probabilities and per-fold AUCs (refit per fold, no leakage)."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr, va in cv.split(X, y):
        m = clone(est).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        fold_aucs.append(roc_auc_score(y[va], oof[va]))
    return oof, fold_aucs


def evaluate_candidate(name: str, est, X, y, folds: int, seed: int) -> CandidateResult:
    t0 = time.time()
    tuning = None
    if name.startswith("lr_"):  # tune regularisation strength on the same folds
        rows = []
        for C in PARAMS["train"]["lr_C_grid"]:
            cand = clone(est).set_params(model__C=C)
            oof, fa = oof_predict(cand, X, y, folds, seed)
            rows.append({"C": C, "cv_roc_auc": roc_auc_score(y, oof)})
        tuning = pd.DataFrame(rows)
        best_C = float(tuning.loc[tuning["cv_roc_auc"].idxmax(), "C"])
        est = clone(est).set_params(model__C=best_C)
    oof, fold_aucs = oof_predict(est, X, y, folds, seed)
    cv = evaluate(y, oof)
    cv["roc_auc_fold_std"] = float(np.std(fold_aucs, ddof=1))
    return CandidateResult(name, est, cv, oof, fold_aucs, scalar_params(est), time.time() - t0, tuning)


def choose_champion(results: list[CandidateResult], tolerance: float | None = None) -> CandidateResult:
    """Best CV ROC-AUC among deployable models; within ``tolerance`` prefer the simplest (parsimony)."""
    tol = PARAMS["train"]["selection_tolerance"] if tolerance is None else tolerance
    pool = [r for r in results if r.name in DEPLOYABLE]
    best_auc = max(r.cv["roc_auc"] for r in pool)
    tied = [r for r in pool if best_auc - r.cv["roc_auc"] <= tol]
    return sorted(tied, key=lambda r: (COMPLEXITY[r.name], -r.cv["roc_auc"]))[0]


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def make_figures(champion_name, champion_model, baseline_model, y_test, X_test, scores_test, out_dir=FIG_DIR) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    base_scores = baseline_model.predict_proba(X_test)[:, 1] if baseline_model is not None else None

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, s in (("Champion: " + champion_name, scores_test), ("M3 baseline (SMOTE, all features)", base_scores)):
        if s is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, label=f"{label}  AUC={roc_auc_score(y_test, s):.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test split)")
    ax.legend(fontsize=7, loc="lower right")
    p = out_dir / "roc_curve.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for label, s in (("Champion", scores_test), ("M3 baseline (SMOTE)", base_scores)):
        if s is None:
            continue
        frac, mean_pred = calibration_curve(y_test, s, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed attrition rate", title="Calibration (test split)")
    ax.legend(fontsize=8)
    ax = axes[1]
    order = np.argsort(-scores_test)
    cum = np.cumsum(np.asarray(y_test)[order]) / np.sum(y_test)
    x = np.arange(1, len(order) + 1) / len(order)
    ax.plot(x, cum, label="Champion")
    ax.plot([0, 1], [0, 1], "k--", label="random targeting")
    ax.axvline(PARAMS["risk"]["high_pct"], color="grey", ls=":")
    ax.set(xlabel="Share of employees flagged (highest risk first)", ylabel="Share of leavers captured",
           title="Cumulative gains (test split)")
    ax.legend(fontsize=8)
    p = out_dir / "calibration_and_gains.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    final = champion_model.named_steps.get("model")
    if hasattr(final, "coef_"):
        names = champion_model.named_steps["features"].get_feature_names_out()
        coefs = pd.Series(final.coef_[0], index=names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(coefs.index, coefs.values, color=np.where(coefs.values > 0, "#DD8452", "#4C72B0"))
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel="Standardised coefficient (log-odds per 1 SD)", title="What drives predicted attrition risk")
        p = out_dir / "model_coefficients.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------------------------
def build_reference(X_train: pd.DataFrame, y_train: np.ndarray, oof_scores: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    """Sample of training data + out-of-fold scores: the baseline monitoring compares live traffic against."""
    ref = X_train[MODEL_FEATURES].copy()
    ref["score"] = oof_scores
    ref["y"] = y_train
    return ref.sample(n=min(n, len(ref)), random_state=seed).reset_index(drop=True)


def save_artifacts(model, card: dict, reference: pd.DataFrame, models_dir=None) -> None:
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / config.MODEL_PATH.name, compress=3)
    (models_dir / config.MODEL_CARD_PATH.name).write_text(json.dumps(card, indent=2), encoding="utf-8")
    reference.to_csv(models_dir / config.REFERENCE_PATH.name, index=False, compression="gzip")


def make_model_card(*, model_id, champion: CandidateResult, test_metrics, auc_ci, thresholds, data_info, results,
                    baseline_test, mlflow_info, extra=None) -> dict:
    card = {
        "model_id": model_id,
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": champion.name,
        "hyperparameters": {k: (v if v is None or isinstance(v, (int, float, str, bool)) else str(v))
                            for k, v in champion.params.items()},
        "features": {
            "inputs": MODEL_FEATURES,
            "limits": {k: list(v) for k, v in config.FEATURE_LIMITS.items()},
            "engineered": ["Tenure_le2", "LogTenure", "SalaryToTenureRatio"] if champion.name != "lr_core3" else [],
            "excluded": {"Age": "no signal", "Gender": "no signal / protected attribute", "JobRole": "no signal"},
        },
        "metrics": {
            "cv": champion.cv,
            "test": test_metrics,
            "test_roc_auc_ci95": list(auc_ci),
            "m3_baseline_test": baseline_test,
        },
        "risk_policy": {**PARAMS["risk"], "thresholds": thresholds},
        "candidates_cv": {
            r.name: {k: round(v, 5) for k, v in r.cv.items()} for r in results
        },
        "training_data": data_info,
        "mlflow": mlflow_info,
        "git_commit": git_commit(),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "intended_use": "Rank current employees by relative attrition risk to prioritise retention conversations.",
        "limitations": LIMITATIONS,
    }
    if extra:
        card.update(extra)
    return card


# ---------------------------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------------------------
def train(quick: bool = False, register: bool = True, save: bool = True, data_path=None, models_dir=None) -> dict:
    p_train, seed = PARAMS["train"], PARAMS["data"]["random_state"]
    folds = 3 if quick else p_train["cv_folds"]
    df = load_training_data(data_path)
    train_df, test_df = split_data(df)
    X_tr, y_tr = feature_frame(train_df), train_df["y"].to_numpy()
    X_te, y_te = feature_frame(test_df), test_df["y"].to_numpy()
    model_id = new_model_id()
    data_info = {
        "path": str(data_path or config.DATA_PATH).replace("\\", "/").split("/")[-1],
        "sha256": file_sha256(data_path or config.DATA_PATH),
        "rows": int(len(df)), "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "attrition_rate": float(df["y"].mean()),
        "split": {"test_size": PARAMS["data"]["test_size"], "random_state": seed, "stratified": True},
    }

    exp_id = ensure_experiment()
    names = ["m3_baseline_lr_smote_all_features", "lr_core3", "lr_engineered", "hgb_engineered"]
    if not quick:
        names += ["rf_engineered", "xgb_engineered"]
    all_models = candidate_models()

    with mlflow.start_run(experiment_id=exp_id, run_name=f"training-{model_id}") as parent:
        mlflow.set_tags({"stage": "training", "git_commit": git_commit(), "data_sha256": data_info["sha256"][:16],
                         "model_id": model_id})
        mlflow.log_params({f"data.{k}": v for k, v in PARAMS["data"].items()})
        mlflow.log_params({"cv_folds": folds, "selection_tolerance": p_train["selection_tolerance"],
                           "n_train": len(train_df), "n_test": len(test_df), "base_rate": round(data_info["attrition_rate"], 4)})

        # --- 1. cross-validated comparison (nested MLflow runs) ---
        results: list[CandidateResult] = []
        for name in names:
            print(f"  CV  {name} ...", flush=True)
            res = evaluate_candidate(name, all_models[name], X_tr, y_tr, folds, seed)
            results.append(res)
            with mlflow.start_run(experiment_id=exp_id, run_name=name, nested=True):
                mlflow.set_tags({"stage": "candidate", "complexity": COMPLEXITY[name], "deployable": name in DEPLOYABLE})
                mlflow.log_params(res.params)
                mlflow.log_metrics({f"cv_{k}": v for k, v in res.cv.items()})
                mlflow.log_metric("fit_seconds", res.seconds)
                if res.tuning is not None:
                    mlflow.log_table(res.tuning, "lr_regularisation_search.json")
        summary = pd.DataFrame([{"model": r.name, **r.cv, "complexity": COMPLEXITY[r.name]} for r in results]) \
            .sort_values("roc_auc", ascending=False)
        print(summary[["model", "roc_auc", "pr_auc", "brier", "lift_top20", "roc_auc_fold_std"]].round(4).to_string(index=False))

        # --- 2. champion selection (explicit, auditable rule) ---
        champion = choose_champion(results)
        print(f"\nChampion: {champion.name} (CV AUC {champion.cv['roc_auc']:.4f})")
        mlflow.set_tag("champion", champion.name)

        # --- 3. final fit + one-shot test evaluation ---
        final_model = clone(champion.estimator).fit(X_tr, y_tr)
        test_scores = final_model.predict_proba(X_te)[:, 1]
        test_metrics = evaluate(y_te, test_scores)
        auc, lo, hi = bootstrap_auc(y_te, test_scores, n_boot=PARAMS["retraining"]["bootstrap_iterations"], seed=seed)
        base_fit = clone(all_models["m3_baseline_lr_smote_all_features"]).fit(X_tr, y_tr)
        baseline_scores = base_fit.predict_proba(X_te)[:, 1]
        baseline_test = evaluate(y_te, baseline_scores)
        thresholds = risk_thresholds(champion.oof_scores)
        print(f"Test AUC {auc:.4f} (95% CI {lo:.4f}-{hi:.4f}); M3 baseline AUC {baseline_test['roc_auc']:.4f}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({"test_roc_auc_ci_low": lo, "test_roc_auc_ci_high": hi,
                            "m3_baseline_test_roc_auc": baseline_test["roc_auc"],
                            "m3_baseline_test_brier": baseline_test["brier"],
                            "risk_threshold_high": thresholds["high"], "risk_threshold_medium": thresholds["medium"]})

        # --- 4. artefacts: figures, tables, model, registry ---
        figs = make_figures(champion.name, final_model, base_fit, y_te, X_te, test_scores)
        for f in figs:
            mlflow.log_artifact(str(f), "figures")
        mlflow.log_table(summary, "candidate_comparison.json")
        reference = build_reference(X_tr, y_tr, champion.oof_scores, p_train["reference_sample_size"], seed)

        mlflow_info = {"tracking_uri": config.MLFLOW_TRACKING_URI.split("///")[0] + "///<local>"
                       if config.MLFLOW_TRACKING_URI.startswith("sqlite") else config.MLFLOW_TRACKING_URI,
                       "experiment": config.MLFLOW_EXPERIMENT, "run_id": parent.info.run_id,
                       "registered_model": config.REGISTERED_MODEL_NAME, "registered_version": None}
        if register:
            from mlflow.models import infer_signature

            sig = infer_signature(X_te[MODEL_FEATURES].head(50), final_model.predict_proba(X_te.head(50))[:, 1])
            info = mlflow.sklearn.log_model(
                final_model, name="model", signature=sig, input_example=X_te[MODEL_FEATURES].head(3),
                registered_model_name=config.REGISTERED_MODEL_NAME,
                serialization_format="cloudpickle",  # custom FeatureEngineer is not on the skops allow-list
            )
            version = str(info.registered_model_version)
            mlflow.MlflowClient().set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", version)
            mlflow_info["registered_version"] = version
            print(f"Registered {config.REGISTERED_MODEL_NAME} v{version} (alias 'champion')")

        card = make_model_card(model_id=model_id, champion=champion, test_metrics=test_metrics, auc_ci=(auc, lo, hi),
                               thresholds=thresholds, data_info=data_info, results=results,
                               baseline_test=baseline_test, mlflow_info=mlflow_info)
        if save:
            save_artifacts(final_model, card, reference, models_dir)
            mlflow.log_artifact(str(config.MODEL_CARD_PATH if models_dir is None else models_dir / config.MODEL_CARD_PATH.name))
            print(f"Saved model + card + reference sample to {models_dir or config.MODELS_DIR}")
        summary.to_csv(config.ROOT / "reports" / "mlops" / "candidate_comparison.csv", index=False) if save else None
        if save:  # tracked by DVC as the stage's metrics file
            (config.ROOT / "reports" / "mlops" / "metrics.json").write_text(json.dumps(
                {"model_id": model_id, "algorithm": champion.name, **{f"test_{k}": round(v, 5) for k, v in test_metrics.items()},
                 "test_roc_auc_ci_low": round(lo, 5), "test_roc_auc_ci_high": round(hi, 5),
                 "cv_roc_auc": round(champion.cv["roc_auc"], 5),
                 "m3_baseline_test_roc_auc": round(baseline_test["roc_auc"], 5),
                 "m3_baseline_test_brier": round(baseline_test["brier"], 5)}, indent=2), encoding="utf-8")
    return {"model_id": model_id, "card": card, "model": final_model, "reference": reference, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3-fold CV and fewer candidates")
    ap.add_argument("--no-register", action="store_true", help="skip the MLflow model registry")
    args = ap.parse_args()
    (config.ROOT / "reports" / "mlops").mkdir(parents=True, exist_ok=True)
    out = train(quick=args.quick, register=not args.no_register)
    print(json.dumps({"model_id": out["model_id"], "algorithm": out["card"]["algorithm"],
                      "test": {k: round(v, 4) for k, v in out["card"]["metrics"]["test"].items()}}, indent=2))


if __name__ == "__main__":
    main()
