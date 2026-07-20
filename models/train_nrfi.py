"""
NRFI/YRFI model (Section 7.3).

Per the spec: NRFI is treated as a low-noise, well-behaved binary target,
so LogisticRegression is the default and XGBoost is only worth keeping if
the backtest shows real lift over it - this script trains both and prints
the comparison so that decision is visible, but `predict_nrfi_probability`
below defaults to the logistic model unless you explicitly pass the
XGBoost one.

Feature caveat: the spec calls for "starter first-inning-specific ERA/WHIP"
and "leadoff hitter OBP" - our schema (Section 5, as given) only stores
game-level pitching/batting lines, not inning-level splits or lineup-slot-
specific rate stats, so this reuses the same season-level starter/lineup/
park features as the other targets (era_season, lineup_wOBA, park factors)
rather than fabricating first-inning-specific numbers we don't have. Adding
true first-inning splits would mean parsing play-by-play data
(gamePk -> playByPlay endpoint), which is a reasonable follow-up but out of
scope for this pass.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from database.db import session_scope
from features.build_feature_matrix import build_training_matrix
from models.model_utils import classification_metrics, date_split, next_version, prepare_xy, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME_LOGISTIC = "nrfi_logistic"
MODEL_NAME_XGB = "nrfi_xgboost"
MEANINGFUL_LIFT_LOGLOSS = 0.01  # XGBoost must beat logistic by at least this much log-loss to be worth the complexity


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


def predict_nrfi_probability(model, feature_row: pd.DataFrame) -> float:
    return float(model.predict_proba(feature_row)[:, 1][0])


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

        X_train, y_train = prepare_xy(train_df)
        X_test, y_test = prepare_xy(test_df)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=X_train.median(numeric_only=True))
        cols = list(X_train.columns)

        log.info("Training logistic regression...")
        logistic = train_logistic(X_train, y_train)
        logistic_metrics = classification_metrics(y_test, logistic.predict_proba(X_test)[:, 1])
        log.info("Logistic test metrics: %s", logistic_metrics)
        save_model(db, logistic, MODEL_NAME_LOGISTIC, "nrfi", next_version(db, MODEL_NAME_LOGISTIC), logistic_metrics, cols)

        log.info("Training XGBoost for comparison...")
        xgb = train_xgboost(X_train, y_train)
        xgb_metrics = classification_metrics(y_test, xgb.predict_proba(X_test)[:, 1])
        log.info("XGBoost test metrics: %s", xgb_metrics)
        save_model(db, xgb, MODEL_NAME_XGB, "nrfi", next_version(db, MODEL_NAME_XGB), xgb_metrics, cols)

        lift = logistic_metrics["log_loss"] - xgb_metrics["log_loss"]
        if lift > MEANINGFUL_LIFT_LOGLOSS:
            log.info("XGBoost beats logistic by %.4f log-loss (> %.2f threshold) - real lift, worth the extra complexity", lift, MEANINGFUL_LIFT_LOGLOSS)
        else:
            log.info("XGBoost does not meaningfully beat logistic (delta %.4f) - stick with the logistic baseline per the spec's own rule", lift)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]))
    else:
        print("Usage: python -m models.train_nrfi TRAIN_START TEST_START TEST_END")
