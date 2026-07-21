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
from functools import lru_cache
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


def walk_forward_splits(
    df: pd.DataFrame, n_splits: int, test_window_days: int, date_col: str = "date"
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window walk-forward folds: each fold trains on every row
    strictly before some cutoff date and tests on the following
    `test_window_days` days, then the cutoff advances and repeats - never a
    random shuffle (Section 11's leakage rule: a random split leaks future
    team form into past predictions, and that applies here just as much as
    to a single train/test split).

    A single chronological train/test split (what date_split above does)
    reports one number that could just be a lucky or unlucky test window.
    Walk-forward reports `n_splits` of them - see the mean/std reported
    alongside each single-split metric in train_moneyline.py/
    train_totals.py/train_nrfi.py - so it's possible to tell whether the
    single-split number is representative or noise.

    Test windows are laid out backward from the most recent date in `df`
    (so the last fold's test window ends at the most recent data available)
    and returned in chronological order, earliest fold first. Folds that
    would have an empty train or test set (not enough history for that
    many splits) are silently dropped rather than raising, since the exact
    number of viable folds depends on how much data is available - the
    caller should check `len(folds)` if that matters.
    """
    dates = pd.to_datetime(df[date_col]).dt.date
    min_date, max_date = dates.min(), dates.max()

    boundaries = []
    test_end = max_date + dt.timedelta(days=1)  # exclusive upper bound
    for _ in range(n_splits):
        test_start = test_end - dt.timedelta(days=test_window_days)
        boundaries.append((test_start, test_end))
        test_end = test_start
    boundaries.reverse()

    folds = []
    for test_start, test_end in boundaries:
        if test_start <= min_date:
            continue  # not enough history yet for a non-empty train set
        train_df = df[dates < test_start].reset_index(drop=True)
        test_df = df[(dates >= test_start) & (dates < test_end)].reset_index(drop=True)
        if not train_df.empty and not test_df.empty:
            folds.append((train_df, test_df))
    return folds


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Numeric feature matrix (missing values median-imputed per column,
    non-numeric columns dropped) + label vector + the per-column medians
    used for that imputation. Callers training a production model must
    persist those medians alongside it (see save_model's feature_medians
    param) - serving code needs the exact same fill values the model was
    trained against, not 0 (see models/predict.py's _score_with_model)."""
    cols = feature_columns(df)
    # See train_totals._prep for why the explicit float cast matters: bool
    # feature columns otherwise survive as dtype `bool`, and a DataFrame
    # mixing bool/int/float dtypes trips stricter estimators than sklearn.
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians).astype(float)
    y = df["label"]
    return X, y, medians.to_dict()


def classification_metrics(y_true, y_prob) -> dict:
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_prob, labels=[0, 1]), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "n": int(len(y_true)),
    }


def summarize_walk_forward(fold_metrics: list[dict]) -> dict:
    """Reduces a list of per-fold metric dicts (each one classification_
    metrics()'s or regression_metrics()'s output) into mean/std per metric
    across folds, e.g. {"accuracy_mean": ..., "accuracy_std": ..., ...}.
    `n` (games per fold) is summed into `n_total` instead of averaged - a
    single-split metric only tells you about one train/test boundary; this
    is what lets a caller tell whether that number is representative or
    just a lucky/unlucky window (see walk_forward_splits above). Returns
    {} if there were no viable folds (not enough history for the
    requested n_splits/test_window_days)."""
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0] if k != "n"]
    summary: dict = {"n_folds": len(fold_metrics), "n_total": sum(int(m["n"]) for m in fold_metrics)}
    for key in keys:
        values = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{key}_std"] = round(float(np.std(values)), 4)
    return summary


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        "n": int(len(y_true)),
    }


def save_model(
    db: Session,
    model,
    model_name: str,
    target_type: str,
    version: str,
    metrics: dict,
    feature_cols: list[str],
    feature_medians: dict[str, float] | None = None,
) -> Path:
    """Pickle the model under models/registry/ and record it in the
    `model_registry` table. `feature_cols` is stored alongside the model so
    prediction-time code can rebuild the exact same column order.
    `feature_medians` (column -> training-set median) is stored the same
    way so live-serving code fills missing values the same way training
    did, instead of a nonsense 0 (see models/predict.py's
    _score_with_model - 0 reads as "elite" for stats like ERA/win_pct)."""
    path = settings.model_registry_path / f"{model_name}_{version}.pkl"
    with path.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols, "feature_medians": feature_medians or {}}, f)

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


@lru_cache(maxsize=32)
def _load_model_cached(path: str) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def load_model(path: str | Path) -> dict:
    return _load_model_cached(str(path))


def next_version(db: Session, model_name: str) -> str:
    """Simple incrementing version string per model name, e.g. v1, v2, ..."""
    count = (
        db.query(ModelRegistryEntry)
        .filter(ModelRegistryEntry.model_name == model_name)
        .count()
    )
    return f"v{count + 1}"
