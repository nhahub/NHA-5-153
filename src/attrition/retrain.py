"""Automated retraining with a champion/challenger gate and one-command rollback.

    python -m attrition.retrain               # retrain only if monitoring/data/age say so
    python -m attrition.retrain --force       # retrain now
    python -m attrition.retrain --dry-run     # evaluate the challenger but never promote
    python -m attrition.retrain --rollback    # restore the previously archived champion

Triggers (any one)               Gate for promotion (all must hold)
  * critical drift / performance   * challenger beats champion on the hold-out by >= ``min_auc_gain`` (paired
    alert in the latest monitoring   bootstrap, evaluated on recent feedback rows when enough exist)
    report                         * challenger is not worse than ``max_regression`` overall (non-inferiority)
  * model older than max age       * challenger passes the same data validation as the champion
  * >= ``min_new_labels`` new labelled outcomes since the last training
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from . import config
from .config import MODEL_FEATURES, PARAMS
from .data import load_labelled_predictions, load_training_data, validate_training_frame
from .modeling import (DEPLOYABLE, candidate_models, evaluate, paired_bootstrap_auc_diff, risk_thresholds)
from .train import (build_reference, choose_champion, ensure_experiment, evaluate_candidate, git_commit,
                    new_model_id, save_artifacts)

CHALLENGERS = ["lr_core3", "lr_engineered", "hgb_engineered"]


def retraining_decision(now: datetime | None = None) -> dict:
    """Combine the latest monitoring report, model age and label volume into a go / no-go."""
    p = PARAMS["retraining"]
    reasons = []
    if config.LATEST_REPORT.exists():
        report = json.loads(config.LATEST_REPORT.read_text(encoding="utf-8"))
        if report.get("recommend_retraining"):
            reasons += report.get("retraining_reasons", [])
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    labelled = load_labelled_predictions()
    trained_at = datetime.fromisoformat(card["trained_at"])
    new = labelled[labelled["outcome_timestamp"] > trained_at] if len(labelled) else labelled
    if len(new) >= p["min_new_labels"]:
        reasons.append(f"{len(new)} new labelled outcomes since training (>= {p['min_new_labels']}).")
    age = ((now or datetime.now(timezone.utc)) - trained_at).days
    if age > PARAMS["monitoring"]["max_model_age_days"]:
        reasons.append(f"Model is {age} days old (> {PARAMS['monitoring']['max_model_age_days']}).")
    return {"retrain": bool(reasons), "reasons": sorted(set(reasons)), "new_labels": int(len(new))}


def build_training_frame() -> pd.DataFrame:
    """Original training data + labelled production feedback (features as scored, outcome as label)."""
    base = load_training_data()[MODEL_FEATURES + ["y"]].assign(source="original", timestamp=pd.NaT)
    fb = load_labelled_predictions()
    if len(fb):
        fb = fb[MODEL_FEATURES + ["y", "timestamp"]].assign(source="feedback")
        base = pd.concat([base, fb], ignore_index=True)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    validate_training_frame(base)
    return base


def archive_champion() -> str:
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    dest = config.ARCHIVE_DIR / card["model_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for src in (config.MODEL_PATH, config.MODEL_CARD_PATH, config.REFERENCE_PATH):
        if src.exists():
            shutil.copy2(src, dest / src.name)
    return card["model_id"]


def rollback() -> str:
    archives = sorted(p for p in config.ARCHIVE_DIR.glob("v*") if p.is_dir()) if config.ARCHIVE_DIR.exists() else []
    if not archives:
        raise SystemExit("No archived model to roll back to.")
    latest = archives[-1]
    for name in (config.MODEL_PATH.name, config.MODEL_CARD_PATH.name, config.REFERENCE_PATH.name):
        if (latest / name).exists():
            shutil.copy2(latest / name, config.MODELS_DIR / name)
    return latest.name


def retrain(force: bool = False, dry_run: bool = False, folds: int | None = None) -> dict:
    p, seed = PARAMS["retraining"], PARAMS["data"]["random_state"]
    decision = retraining_decision()
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "decision": decision, "forced": force,
              "dry_run": dry_run, "promoted": False}
    if not decision["retrain"] and not force:
        result["outcome"] = "skipped: no retraining trigger"
        return _persist(result)

    champion_model = joblib.load(config.MODEL_PATH)
    champion_card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    df = build_training_frame()
    train_df, hold_df = train_test_split(df, test_size=PARAMS["data"]["test_size"], random_state=seed,
                                         stratify=df["y"])
    X_tr, y_tr = train_df[MODEL_FEATURES], train_df["y"].to_numpy()
    X_ho, y_ho = hold_df[MODEL_FEATURES], hold_df["y"].to_numpy()

    folds = folds or PARAMS["train"]["cv_folds"]
    models = candidate_models()
    results = [evaluate_candidate(n, models[n], X_tr, y_tr, folds, seed) for n in CHALLENGERS if n in DEPLOYABLE]
    best = choose_champion(results)
    challenger = clone(best.estimator).fit(X_tr, y_tr)

    s_new, s_old = challenger.predict_proba(X_ho)[:, 1], champion_model.predict_proba(X_ho)[:, 1]
    overall = paired_bootstrap_auc_diff(y_ho, s_new, s_old, p["bootstrap_iterations"], seed)
    recent_mask = (hold_df["source"] == "feedback").to_numpy()
    recent = (paired_bootstrap_auc_diff(y_ho[recent_mask], s_new[recent_mask], s_old[recent_mask],
                                        p["bootstrap_iterations"], seed)
              if recent_mask.sum() >= PARAMS["monitoring"]["min_labels"] and len(np.unique(y_ho[recent_mask])) > 1
              else None)
    gate_diff = recent if recent is not None else overall
    checks = {
        "improves_enough": gate_diff["diff"] >= p["min_auc_gain"],
        "not_worse_overall": overall["ci_low"] >= -p["max_regression"],
    }
    promote = all(checks.values())
    result.update({
        "challenger": best.name, "hold_out_rows": int(len(hold_df)), "recent_rows_in_hold_out": int(recent_mask.sum()),
        "champion_hold_out": evaluate(y_ho, s_old), "challenger_hold_out": evaluate(y_ho, s_new),
        "auc_diff_overall": overall, "auc_diff_recent": recent, "gate": checks,
        "gate_basis": "recent feedback rows" if recent is not None else "full hold-out",
    })

    exp_id = ensure_experiment()
    with mlflow.start_run(experiment_id=exp_id, run_name=f"retraining-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        mlflow.set_tags({"stage": "retraining", "promoted": str(promote and not dry_run), "git_commit": git_commit(),
                         "trigger": "; ".join(decision["reasons"])[:250] or "forced"})
        mlflow.log_metrics({"champion_holdout_auc": result["champion_hold_out"]["roc_auc"],
                            "challenger_holdout_auc": result["challenger_hold_out"]["roc_auc"],
                            "auc_diff": overall["diff"], "auc_diff_ci_low": overall["ci_low"]})
        if promote and not dry_run:
            parent_id = archive_champion()
            model_id = new_model_id()
            thresholds = risk_thresholds(best.oof_scores)
            reference = build_reference(X_tr, y_tr, best.oof_scores, PARAMS["train"]["reference_sample_size"], seed)
            from mlflow.models import infer_signature

            info = mlflow.sklearn.log_model(
                challenger, name="model", signature=infer_signature(X_ho.head(50), s_new[:50]),
                registered_model_name=config.REGISTERED_MODEL_NAME, serialization_format="cloudpickle")
            version = str(info.registered_model_version)
            mlflow.MlflowClient().set_registered_model_alias(config.REGISTERED_MODEL_NAME, "champion", version)
            card = {**champion_card, "model_id": model_id, "trained_at": datetime.now(timezone.utc).isoformat(),
                    "algorithm": best.name, "parent_model_id": parent_id, "git_commit": git_commit(),
                    "hyperparameters": {k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
                                        for k, v in best.params.items()},
                    "risk_policy": {**champion_card["risk_policy"], "thresholds": thresholds},
                    "metrics": {**champion_card["metrics"], "test": result["challenger_hold_out"],
                                "test_roc_auc_ci95": None, "note": "hold-out of the retraining dataset"},
                    "training_data": {**champion_card["training_data"], "rows": int(len(df)),
                                      "feedback_rows": int((df["source"] == "feedback").sum()),
                                      "attrition_rate": float(df["y"].mean())},
                    "mlflow": {**champion_card["mlflow"], "registered_version": version},
                    "retraining": {"reasons": decision["reasons"], "gate": checks}}
            save_artifacts(challenger, card, reference)
            result.update({"promoted": True, "new_model_id": model_id, "previous_model_id": parent_id,
                           "outcome": f"promoted {best.name} as {model_id} (previous champion archived as {parent_id})"})
        else:
            result["outcome"] = ("challenger passed the gate but --dry-run: not promoted" if promote else
                                 "challenger rejected: champion kept (" +
                                 ", ".join(k for k, v in checks.items() if not v) + ")")
    return _persist(result)


def _persist(result: dict) -> dict:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORT_DIR / f"retraining_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()
    if args.rollback:
        print("Rolled back to", rollback())
        return
    res = retrain(force=args.force, dry_run=args.dry_run)
    print(json.dumps({k: res[k] for k in res if k in ("decision", "outcome", "gate", "auc_diff_overall",
                                                       "auc_diff_recent", "gate_basis")}, indent=2, default=str))


if __name__ == "__main__":
    main()
