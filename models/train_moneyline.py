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
from models.model_utils import classification_metrics, date_split, feature_columns, next_version, prepare_xy, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME_BASELINE = "moneyline_logistic"
MODEL_NAME_XGB = "moneyline_xgboost"


def train_baseline_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    # Feature scales vary wildly here (e.g. era_season ~0-10 vs win_pct
    # ~0-1), which is exactly what makes lbfgs slow/non-convergent - scale
    # first rather than just cranking max_iter.
    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    return model


def train_xgboost_calibrated(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    """Splits X_train further into fit/calibration slices (still
    chronologically - the last 20% of the training window becomes both the
    early-stopping eval set and the isotonic calibration set)."""
    fit_X, calib_X, fit_y, calib_y = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        early_stopping_rounds=20,
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

        X_train, y_train = prepare_xy(train_df)
        X_test, y_test = prepare_xy(test_df)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=X_train.median(numeric_only=True))
        cols = list(X_train.columns)

        log.info("Training baseline logistic regression...")
        baseline = train_baseline_logistic(X_train, y_train)
        baseline_metrics = classification_metrics(y_test, baseline.predict_proba(X_test)[:, 1])
        log.info("Baseline (logistic) test metrics: %s", baseline_metrics)
        save_model(db, baseline, MODEL_NAME_BASELINE, "moneyline", next_version(db, MODEL_NAME_BASELINE), baseline_metrics, cols)

        log.info("Training XGBoost (calibrated)...")
        xgb_model = train_xgboost_calibrated(X_train, y_train)
        xgb_metrics = classification_metrics(y_test, xgb_model.predict_proba(X_test)[:, 1])
        log.info("XGBoost test metrics: %s", xgb_metrics)
        save_model(db, xgb_model, MODEL_NAME_XGB, "moneyline", next_version(db, MODEL_NAME_XGB), xgb_metrics, cols)

        winner = MODEL_NAME_XGB if xgb_metrics["log_loss"] < baseline_metrics["log_loss"] else MODEL_NAME_BASELINE
        log.info("Lower log-loss on held-out test set: %s", winner)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]))
    else:
        print("Usage: python -m models.train_moneyline TRAIN_START TEST_START TEST_END")
