"""
Moneyline win-probability model (Section 7.1).

Two estimators, both calibrated (isotonic) since the whole point is
comparing our predicted probability to the market's implied probability -
an uncalibrated classifier's raw scores aren't directly comparable to that:

- Baseline: LogisticRegression(class_weight='balanced'), wrapped in
  CalibratedClassifierCV.
- Production: XGBClassifier with early stopping on log-loss, then
  calibrated on a held-out slice (`cv="prefit"` - fitting a fresh
  CalibratedClassifierCV with cv=N would re-fit XGBoost N times and lose
  the early-stopping eval set).

Run as a script to train + compare both against a date-split test set and
print backtest metrics; import `train_baseline_logistic` /
`train_xgboost_calibrated` / `predict_win_probability` to use from
elsewhere (e.g. the daily prediction job).
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from database.db import session_scope
from features.build_feature_matrix import build_training_matrix
from models.model_utils import (
    classification_metrics,
    date_split,
    feature_columns,
    next_version,
    prepare_xy,
    save_model,
    seasonal_walk_forward_splits,
    summarize_walk_forward,
    walk_forward_splits_by_games,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME_BASELINE = "moneyline_logistic"
MODEL_NAME_XGB = "moneyline_xgboost"

# Walk-forward validation reporting (see model_utils.walk_forward_splits_by_games):
# 5 folds of 150 games each - counted in games, not calendar days, so a
# fold is never silently dropped for landing in the Nov-Mar off-season the
# way a calendar-day window could once the backfill spans multiple
# seasons. 150 games is roughly the same order of magnitude as the old
# 14-day calendar window's real game count (~15 games/day on a typical
# in-season MLB slate), just measured in a way that's immune to off-days
# and the off-season gap.
WALK_FORWARD_N_SPLITS = 5
WALK_FORWARD_TEST_SIZE_GAMES = 150


def train_baseline_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    # Feature scales vary wildly here (e.g. era_season ~0-10 vs win_pct
    # ~0-1), which is exactly what makes lbfgs slow/non-convergent - scale
    # first rather than just cranking max_iter.
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=4000, solver="lbfgs", random_state=42),
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    return model


def train_xgboost_calibrated(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    """Splits X_train further into fit/calibration slices (still
    chronologically - the last 20% of the training window becomes both the
    early-stopping eval set and the isotonic calibration set)."""
    fit_X, calib_X, fit_y, calib_y = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    positive_ratio = float(y_train.mean())
    negative_ratio = 1.0 - positive_ratio
    scale_pos_weight = negative_ratio / positive_ratio if positive_ratio > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=700,
        max_depth=5,
        min_child_weight=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=25,
        random_state=42,
    )
    xgb.fit(fit_X, fit_y, eval_set=[(calib_X, calib_y)], verbose=False)

    # sklearn >=1.6 replaced CalibratedClassifierCV(cv="prefit") with
    # wrapping the already-fitted estimator in FrozenEstimator - the old
    # "prefit" string is a hard InvalidParameterError now, not a warning.
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    calibrated.fit(calib_X, calib_y)
    return calibrated


def predict_win_probability(model, feature_row: pd.DataFrame) -> float:
    """`feature_row` must already be aligned to the model's training
    feature columns (see model_utils.save_model / load_model, which store
    `feature_columns` alongside the pickle for exactly this)."""
    return float(model.predict_proba(feature_row)[:, 1][0])


def _walk_forward_metrics(df: pd.DataFrame, train_fn) -> list[dict]:
    """Trains `train_fn(X_train, y_train) -> fitted model` on each
    walk-forward fold (model_utils.walk_forward_splits_by_games) and scores
    it on that fold's own held-out test window - a fresh model per fold,
    not the single production model saved by run() below."""
    fold_metrics = []
    for train_fold, test_fold in walk_forward_splits_by_games(df, WALK_FORWARD_N_SPLITS, WALK_FORWARD_TEST_SIZE_GAMES):
        X_tr, y_tr, _ = prepare_xy(train_fold)
        X_te, y_te, _ = prepare_xy(test_fold)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=X_tr.median(numeric_only=True))
        model = train_fn(X_tr, y_tr)
        fold_metrics.append(classification_metrics(y_te, model.predict_proba(X_te)[:, 1]))
    return fold_metrics


def _seasonal_metrics(df: pd.DataFrame, train_fn) -> list[tuple[int, dict]]:
    """Same per-fold pattern as _walk_forward_metrics above, but one fold
    per season (model_utils.seasonal_walk_forward_splits) - reports
    metrics labeled by the season tested rather than averaged together, so
    it's possible to eyeball whether performance holds steady across
    seasons rather than just across the last few thousand games."""
    results = []
    for train_fold, test_fold in seasonal_walk_forward_splits(df):
        season = int(pd.to_datetime(test_fold["date"]).dt.year.iloc[0])
        X_tr, y_tr, _ = prepare_xy(train_fold)
        X_te, y_te, _ = prepare_xy(test_fold)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=X_tr.median(numeric_only=True))
        model = train_fn(X_tr, y_tr)
        results.append((season, classification_metrics(y_te, model.predict_proba(X_te)[:, 1])))
    return results


def run(train_start: dt.date, test_start: dt.date, test_end: dt.date) -> None:
    with session_scope() as db:
        log.info("Building training matrix %s -> %s (moneyline)", train_start, test_end)
        df = build_training_matrix(db, train_start, test_end, target="moneyline")
        log.info("Matrix shape: %s", df.shape)
        if df.empty:
            log.error("No games found in range - did you run scripts/backfill_data.py first?")
            return

        train_df, test_df = date_split(df, test_start)
        log.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))
        if train_df.empty or test_df.empty:
            log.error("Train or test split is empty - widen the date range")
            return

        X_train, y_train, train_medians = prepare_xy(train_df)
        X_test, y_test, _ = prepare_xy(test_df)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=X_train.median(numeric_only=True))
        cols = list(X_train.columns)

        log.info("Training baseline logistic regression...")
        baseline = train_baseline_logistic(X_train, y_train)
        baseline_metrics = classification_metrics(y_test, baseline.predict_proba(X_test)[:, 1])
        log.info("Baseline (logistic) test metrics: %s", baseline_metrics)
        save_model(db, baseline, MODEL_NAME_BASELINE, "moneyline", next_version(db, MODEL_NAME_BASELINE), baseline_metrics, cols, feature_medians=train_medians)

        log.info("Training XGBoost (calibrated)...")
        xgb_model = train_xgboost_calibrated(X_train, y_train)
        xgb_metrics = classification_metrics(y_test, xgb_model.predict_proba(X_test)[:, 1])
        log.info("XGBoost test metrics: %s", xgb_metrics)
        save_model(db, xgb_model, MODEL_NAME_XGB, "moneyline", next_version(db, MODEL_NAME_XGB), xgb_metrics, cols, feature_medians=train_medians)

        winner = MODEL_NAME_XGB if xgb_metrics["log_loss"] < baseline_metrics["log_loss"] else MODEL_NAME_BASELINE
        log.info("Lower log-loss on held-out test set: %s", winner)

        # Walk-forward validation: the single split above is one train/test
        # boundary - could be a lucky or unlucky window. Retrain fresh
        # per-fold models across several expanding-window folds (counted in
        # games, not days - see WALK_FORWARD_TEST_SIZE_GAMES) and report
        # mean±std so it's clear whether the single-split number above is
        # representative or noise.
        log.info("Running walk-forward validation (%d folds x %d games)...", WALK_FORWARD_N_SPLITS, WALK_FORWARD_TEST_SIZE_GAMES)
        baseline_wf = summarize_walk_forward(_walk_forward_metrics(df, train_baseline_logistic))
        xgb_wf = summarize_walk_forward(_walk_forward_metrics(df, train_xgboost_calibrated))
        log.info("Baseline (logistic) - single split: %s | walk-forward: %s", baseline_metrics, baseline_wf or "not enough history for a walk-forward fold")
        log.info("XGBoost - single split: %s | walk-forward: %s", xgb_metrics, xgb_wf or "not enough history for a walk-forward fold")

        # Seasonal validation: walk-forward above only ever looks back
        # n_splits x test_size games, which never reaches past the last few
        # months even with several backfilled seasons in `df`. One fold per
        # season answers the actual question of interest once the backfill
        # is widened: is 2024/2025 accuracy in the same ballpark as 2026's,
        # or did the model only ever "work" on the most recent season?
        log.info("Running seasonal walk-forward validation (one fold per season after the first)...")
        baseline_seasonal = _seasonal_metrics(df, train_baseline_logistic)
        xgb_seasonal = _seasonal_metrics(df, train_xgboost_calibrated)
        if not baseline_seasonal:
            log.info("Only one season in range - no seasonal comparison possible yet")
        for season, metrics in baseline_seasonal:
            log.info("Baseline (logistic) - season %d tested: %s", season, metrics)
        for season, metrics in xgb_seasonal:
            log.info("XGBoost - season %d tested: %s", season, metrics)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]))
    else:
        print("Usage: python -m models.train_moneyline TRAIN_START TEST_START TEST_END")
