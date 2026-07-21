"""
NRFI/YRFI model (Section 7.3).

Per the spec: NRFI is treated as a low-noise, well-behaved binary target,
so LogisticRegression is the default and XGBoost is only worth keeping if
the backtest shows real lift over it - this script trains both and prints
the comparison so that decision is visible, but `predict_nrfi_probability`
below defaults to the logistic model unless you explicitly pass the
XGBoost one.

Feature caveat: the spec calls for "starter first-inning-specific ERA/WHIP"
and "leadoff hitter OBP". Leadoff OBP is now real (features/batter_features
.compute_leadoff_obp, folded into home_lineup/away_lineup) - the confirmed
or projected leadoff hitter's season-to-date OBP. Starter first-inning-
specific ERA/WHIP is still a gap: our schema (Section 5, as given) only
stores whole-game pitching lines, not inning-level splits, so this reuses
the same season-level era_season the other targets use rather than
fabricating a first-inning-specific number we don't have. Closing that
gap needs play-by-play parsing (gamePk -> playByPlay endpoint) - a
reasonable follow-up, still out of scope for this pass.
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
    next_version,
    prepare_xy,
    save_model,
    seasonal_walk_forward_splits,
    summarize_walk_forward,
    walk_forward_splits_by_games,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME_LOGISTIC = "nrfi_logistic"
MODEL_NAME_XGB = "nrfi_xgboost"
MEANINGFUL_LIFT_LOGLOSS = 0.01  # XGBoost must beat logistic by at least this much log-loss to be worth the complexity

# Walk-forward validation reporting (see model_utils.walk_forward_splits_by_games
# and train_moneyline.py's identical convention) - 5 folds of 150 games each,
# counted in games rather than calendar days so a fold can't land in the
# Nov-Mar off-season and come back empty.
WALK_FORWARD_N_SPLITS = 5
WALK_FORWARD_TEST_SIZE_GAMES = 150


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    # Same early-stopping + calibration pattern as
    # train_moneyline.train_xgboost_calibrated - this file's own docstring
    # says probability comparison needs calibration, which previously only
    # applied to the logistic model; an uncalibrated XGBoost here was an
    # inconsistency, not a deliberate choice. The held-out 20% slice
    # (chronological, not shuffled) serves double duty as the early-stopping
    # eval set and the isotonic calibration set.
    fit_X, calib_X, fit_y, calib_y = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    positive_ratio = float(y_train.mean())
    negative_ratio = 1.0 - positive_ratio
    scale_pos_weight = negative_ratio / positive_ratio if positive_ratio > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        min_child_weight=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=25,
        random_state=42,
    )
    xgb.fit(fit_X, fit_y, eval_set=[(calib_X, calib_y)], verbose=False)

    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    calibrated.fit(calib_X, calib_y)
    return calibrated


def predict_nrfi_probability(model, feature_row: pd.DataFrame) -> float:
    return float(model.predict_proba(feature_row)[:, 1][0])


def _walk_forward_metrics(df: pd.DataFrame, train_fn) -> list[dict]:
    """See train_moneyline._walk_forward_metrics - identical pattern, a
    fresh model per walk-forward fold, scored on that fold's own held-out
    window."""
    fold_metrics = []
    for train_fold, test_fold in walk_forward_splits_by_games(df, WALK_FORWARD_N_SPLITS, WALK_FORWARD_TEST_SIZE_GAMES):
        X_tr, y_tr, _ = prepare_xy(train_fold)
        X_te, y_te, _ = prepare_xy(test_fold)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=X_tr.median(numeric_only=True))
        model = train_fn(X_tr, y_tr)
        fold_metrics.append(classification_metrics(y_te, model.predict_proba(X_te)[:, 1]))
    return fold_metrics


def _seasonal_metrics(df: pd.DataFrame, train_fn) -> list[tuple[int, dict]]:
    """See train_moneyline._seasonal_metrics - identical pattern, one fold
    per season (model_utils.seasonal_walk_forward_splits), labeled by the
    season tested rather than averaged together."""
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
        log.info("Building training matrix %s -> %s (nrfi)", train_start, test_end)
        df = build_training_matrix(db, train_start, test_end, target="nrfi")
        if df.empty:
            log.error("No games with linescore data in range - run scripts/backfill_data.py first")
            return

        train_df, test_df = date_split(df, test_start)
        if train_df.empty or test_df.empty:
            log.error("Train or test split is empty - widen the date range")
            return
        log.info("Train rows: %d (NRFI rate %.1f%%), Test rows: %d", len(train_df), 100 * train_df["label"].mean(), len(test_df))

        X_train, y_train, train_medians = prepare_xy(train_df)
        X_test, y_test, _ = prepare_xy(test_df)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=X_train.median(numeric_only=True))
        cols = list(X_train.columns)

        log.info("Training logistic regression...")
        logistic = train_logistic(X_train, y_train)
        logistic_metrics = classification_metrics(y_test, logistic.predict_proba(X_test)[:, 1])
        log.info("Logistic test metrics: %s", logistic_metrics)
        save_model(db, logistic, MODEL_NAME_LOGISTIC, "nrfi", next_version(db, MODEL_NAME_LOGISTIC), logistic_metrics, cols, feature_medians=train_medians)

        log.info("Training XGBoost for comparison...")
        xgb = train_xgboost(X_train, y_train)
        xgb_metrics = classification_metrics(y_test, xgb.predict_proba(X_test)[:, 1])
        log.info("XGBoost test metrics: %s", xgb_metrics)
        save_model(db, xgb, MODEL_NAME_XGB, "nrfi", next_version(db, MODEL_NAME_XGB), xgb_metrics, cols, feature_medians=train_medians)

        lift = logistic_metrics["log_loss"] - xgb_metrics["log_loss"]
        if lift > MEANINGFUL_LIFT_LOGLOSS:
            log.info("XGBoost beats logistic by %.4f log-loss (> %.2f threshold) - real lift, worth the extra complexity", lift, MEANINGFUL_LIFT_LOGLOSS)
        else:
            log.info("XGBoost does not meaningfully beat logistic (delta %.4f) - stick with the logistic baseline per the spec's own rule", lift)

        # Walk-forward validation - see train_moneyline.py's identical block
        # for why this matters: one single-split number could be a lucky or
        # unlucky test window.
        log.info("Running walk-forward validation (%d folds x %d games)...", WALK_FORWARD_N_SPLITS, WALK_FORWARD_TEST_SIZE_GAMES)
        logistic_wf = summarize_walk_forward(_walk_forward_metrics(df, train_logistic))
        xgb_wf = summarize_walk_forward(_walk_forward_metrics(df, train_xgboost))
        log.info("Logistic - single split: %s | walk-forward: %s", logistic_metrics, logistic_wf or "not enough history for a walk-forward fold")
        log.info("XGBoost - single split: %s | walk-forward: %s", xgb_metrics, xgb_wf or "not enough history for a walk-forward fold")

        # Seasonal validation - see train_moneyline.py's identical block:
        # answers whether accuracy holds steady across seasons, which the
        # walk-forward block above can't (its lookback never reaches past
        # the last few months of games).
        log.info("Running seasonal walk-forward validation (one fold per season after the first)...")
        logistic_seasonal = _seasonal_metrics(df, train_logistic)
        xgb_seasonal = _seasonal_metrics(df, train_xgboost)
        if not logistic_seasonal:
            log.info("Only one season in range - no seasonal comparison possible yet")
        for season, metrics in logistic_seasonal:
            log.info("Logistic - season %d tested: %s", season, metrics)
        for season, metrics in xgb_seasonal:
            log.info("XGBoost - season %d tested: %s", season, metrics)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]))
    else:
        print("Usage: python -m models.train_nrfi TRAIN_START TEST_START TEST_END")
