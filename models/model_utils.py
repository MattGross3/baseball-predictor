"""
Shared helpers for the model-training scripts: date-based splitting (never
random - Section 11's leakage warning applies to model training too, not
just backtesting), classification metrics, and model-registry persistence
(both the pickled artifact under models/registry/ and the DB row in
`model_registry` from Section 5).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sqlalchemy.orm import Session

from config import settings
from database.models import ModelRegistryEntry

log = logging.getLogger(__name__)

# Columns build_feature_matrix.py adds that are identifiers/labels/raw
# outcome data, never features. home_score/away_score/first_inning_* are
# literally what every label is derived from - including them as model
# inputs would be direct leakage of the target.
NON_FEATURE_COLUMNS = {
    "game_id", "date", "label",
    "home_score", "away_score", "first_inning_home_runs", "first_inning_away_runs",
}


def date_split(df: pd.DataFrame, test_start: dt.date, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date, never by random shuffle. Rows before `test_start` are
    train, rows on/after are test."""
    dates = pd.to_datetime(df[date_col]).dt.date
    train = df[dates < test_start].reset_index(drop=True)
    test = df[dates >= test_start].reset_index(drop=True)
    return train, test


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Numeric feature matrix (missing values median-imputed per column,
    non-numeric columns dropped) + label vector."""
    cols = feature_columns(df)
    # See train_totals._prep for why the explicit float cast matters: bool
    # feature columns otherwise survive as dtype `bool`, and a DataFrame
    # mixing bool/int/float dtypes trips stricter estimators than sklearn.
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).astype(float)
    y = df["label"]
    return X, y


def classification_metrics(y_true, y_prob) -> dict:
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_prob, labels=[0, 1]), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "n": int(len(y_true)),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        "n": int(len(y_true)),
    }


def save_model(db: Session, model, model_name: str, target_type: str, version: str, metrics: dict, feature_cols: list[str]) -> Path:
    """Pickle the model under models/registry/ and record it in the
    `model_registry` table. `feature_cols` is stored alongside the model so
    prediction-time code can rebuild the exact same column order."""
    path = settings.model_registry_path / f"{model_name}_{version}.pkl"
    with path.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols}, f)

    entry = ModelRegistryEntry(
        model_name=model_name,
        target_type=target_type,
        version=version,
        trained_at=dt.datetime.now(dt.timezone.utc),
        metrics_json=metrics,
        file_path=str(path),
    )
    db.add(entry)
    db.flush()
    log.info("Saved model %s %s -> %s (%s)", model_name, version, path, metrics)
    return path


def load_model(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def next_version(db: Session, model_name: str) -> str:
    """Simple incrementing version string per model name, e.g. v1, v2, ..."""
    count = (
        db.query(ModelRegistryEntry)
        .filter(ModelRegistryEntry.model_name == model_name)
        .count()
    )
    return f"v{count + 1}"
