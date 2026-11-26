"""Data access helpers shared by training, monitoring and retraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .config import MODEL_FEATURES, PARAMS, TARGET


def file_sha256(path: Path) -> str:
    """Content hash used as a lightweight data version (DVC stores the same idea in ``dvc.lock``)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_data(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned Milestone 1 dataset with a binary ``y`` target column (1 = left)."""
    df = pd.read_csv(path or config.DATA_PATH)
    df["y"] = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)
    validate_training_frame(df)
    return df


def validate_training_frame(df: pd.DataFrame) -> None:
    """Fail fast on schema or quality problems instead of silently training on bad data."""
    missing = [c for c in MODEL_FEATURES + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    if df[MODEL_FEATURES].isna().any().any():
        raise ValueError("Training data contains missing values in model features.")
    for col, (lo, hi) in config.FEATURE_LIMITS.items():
        if not df[col].between(lo, hi).all():
            raise ValueError(f"Column {col} has values outside [{lo}, {hi}].")
    if df["y"].nunique() < 2:
        raise ValueError("Training data must contain both classes.")


def split_data(df: pd.DataFrame, random_state: int | None = None, test_size: float | None = None):
    """Stratified train/test split. Defaults reproduce the Milestone 3 split exactly."""
    p = PARAMS["data"]
    return train_test_split(
        df,
        test_size=test_size if test_size is not None else p["test_size"],
        random_state=random_state if random_state is not None else p["random_state"],
        stratify=df["y"],
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # tolerate a half-written last line
                    continue
    return rows


def load_prediction_log(path: Path | None = None) -> pd.DataFrame:
    """Prediction log written by the API -> DataFrame (one row per scored employee)."""
    rows = read_jsonl(path or config.PREDICTION_LOG)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_labelled_predictions(pred_path: Path | None = None, outcome_path: Path | None = None) -> pd.DataFrame:
    """Join predictions with later-observed outcomes (ground truth feedback) on ``request_id``."""
    preds = load_prediction_log(pred_path)
    outcomes = read_jsonl(outcome_path or config.OUTCOME_LOG)
    if preds.empty or not outcomes:
        return pd.DataFrame()
    out = pd.DataFrame(outcomes).drop_duplicates("request_id", keep="last")
    out = out.rename(columns={"timestamp": "outcome_timestamp"})
    merged = preds.merge(out[["request_id", "left", "outcome_timestamp"]], on="request_id", how="inner")
    merged["y"] = merged["left"].astype(int)
    return merged
