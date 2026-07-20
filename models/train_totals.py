"""
Run totals model (Section 7.2).

- Baseline: separate Poisson GLM per side (home runs ~ features, away runs
  ~ features), then convolve the two independent Poisson distributions to
  get a full probability distribution over the game total - not just a
  point estimate. Two independent Poissons is a simplification (real
  scoring has some home/away correlation via pace-of-play, bullpen usage
  etc.) but it's the standard first-pass baseline for this exact problem.
- Production: XGBRegressor on the combined total directly, paired with a
  Negative Binomial variance estimate (fit once on the training residuals)
  to turn the point prediction into a distribution shape too - XGBoost
  itself only gives a point estimate, so the NB dispersion parameter is
  what lets `predict_run_distribution` return over/under probabilities for
  arbitrary total lines.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.signal import fftconvolve
from xgboost import XGBRegressor

from database.db import session_scope
from features.build_feature_matrix import build_training_matrix
from models.model_utils import date_split, feature_columns, next_version, regression_metrics, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME_POISSON = "totals_poisson"
MODEL_NAME_XGB = "totals_xgboost"
MAX_TOTAL_RUNS = 30  # distribution support: 0..MAX_TOTAL_RUNS combined runs


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns(df)
    # Boolean feature columns (closer_available, lineup_confirmed,
    # roof_closed) survive pd.to_numeric as dtype `bool`, not `bool`->float -
    # a DataFrame that mixes bool/int64/float64 columns trips statsmodels'
    # stricter-than-sklearn dtype check in GLM. Force a uniform float matrix.
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    return X.fillna(X.median(numeric_only=True)).astype(float)


def train_poisson_baseline(train_df: pd.DataFrame) -> dict:
    """Returns {'home': fitted GLM, 'away': fitted GLM, 'columns': [...]}."""
    X = _prep(train_df)
    X_const = sm.add_constant(X, has_constant="add")

    home_model = sm.GLM(train_df["home_score"], X_const, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(train_df["away_score"], X_const, family=sm.families.Poisson()).fit()
    return {"home": home_model, "away": away_model, "columns": list(X.columns)}


def poisson_run_distribution(poisson_models: dict, feature_row: pd.DataFrame) -> dict:
    """Convolve the two independent Poisson(lambda_home), Poisson(lambda_away)
    distributions into a distribution over combined totals 0..MAX_TOTAL_RUNS."""
    # `feature_row` may be a raw DataFrame slice (e.g. train_totals.run()
    # passes test_df.iloc[[i]] straight through) that still has bool-dtype
    # columns - reindex first, then force numeric/float uniformly so
    # statsmodels doesn't choke on a mixed-dtype frame (see _prep).
    X = feature_row.reindex(columns=poisson_models["columns"], fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    X_const = sm.add_constant(X, has_constant="add")
    # statsmodels' add_constant needs the same column ordering the model was
    # fit with; reindex against the fitted params' index to be safe.
    X_const = X_const.reindex(columns=poisson_models["home"].params.index, fill_value=0)

    lambda_home = float(poisson_models["home"].predict(X_const).iloc[0])
    lambda_away = float(poisson_models["away"].predict(X_const).iloc[0])

    support = np.arange(0, MAX_TOTAL_RUNS + 1)
    pmf_home = stats.poisson.pmf(support, lambda_home)
    pmf_away = stats.poisson.pmf(support, lambda_away)
    total_pmf = fftconvolve(pmf_home, pmf_away)[: MAX_TOTAL_RUNS + 1]
    total_pmf = total_pmf / total_pmf.sum()

    return {
        "mean": round(lambda_home + lambda_away, 2),
        "lambda_home": round(lambda_home, 2),
        "lambda_away": round(lambda_away, 2),
        "distribution_over_totals": {int(t): round(float(p), 4) for t, p in zip(support, total_pmf)},
    }


def train_xgb_totals(train_df: pd.DataFrame) -> dict:
    X = _prep(train_df)
    y = train_df["label"]
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)
    model.fit(X, y)

    residuals = y - model.predict(X)
    # Negative binomial dispersion from the variance-to-mean ratio of the
    # residual-adjusted counts: var = mean + mean^2/r  =>  r = mean^2/(var-mean).
    resid_mean = max(float(y.mean()), 1e-6)
    resid_var = max(float(residuals.var()) + resid_mean, resid_mean + 1e-6)
    r = resid_mean**2 / max(resid_var - resid_mean, 1e-6)

    return {"model": model, "columns": list(X.columns), "nb_r": r}


def xgb_run_distribution(xgb_bundle: dict, feature_row: pd.DataFrame) -> dict:
    X = feature_row.reindex(columns=xgb_bundle["columns"], fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    mean = max(float(xgb_bundle["model"].predict(X)[0]), 0.1)

    r = xgb_bundle["nb_r"]
    p = r / (r + mean)  # NB parameterized by (r, p) with mean = r*(1-p)/p
    support = np.arange(0, MAX_TOTAL_RUNS + 1)
    pmf = stats.nbinom.pmf(support, r, p)
    pmf = pmf / pmf.sum()

    return {"mean": round(mean, 2), "distribution_over_totals": {int(t): round(float(prob), 4) for t, prob in zip(support, pmf)}}


def run(train_start: dt.date, test_start: dt.date, test_end: dt.date) -> None:
    with session_scope() as db:
        log.info("Building training matrix %s -> %s (totals)", train_start, test_end)
        df = build_training_matrix(db, train_start, test_end, target="total")
        if df.empty:
            log.error("No games found in range - run scripts/backfill_data.py first")
            return

        train_df, test_df = date_split(df, test_start)
        if train_df.empty or test_df.empty:
            log.error("Train or test split is empty - widen the date range")
            return
        log.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

        log.info("Training Poisson baseline (per-side GLM)...")
        poisson_models = train_poisson_baseline(train_df)
        poisson_preds = [poisson_run_distribution(poisson_models, test_df.iloc[[i]])["mean"] for i in range(len(test_df))]
        poisson_metrics = regression_metrics(test_df["label"], poisson_preds)
        log.info("Poisson baseline test metrics: %s", poisson_metrics)
        save_model(db, poisson_models, MODEL_NAME_POISSON, "total", next_version(db, MODEL_NAME_POISSON), poisson_metrics, poisson_models["columns"])

        log.info("Training XGBoost totals model...")
        xgb_bundle = train_xgb_totals(train_df)
        xgb_preds = xgb_bundle["model"].predict(_prep(test_df).reindex(columns=xgb_bundle["columns"], fill_value=0))
        xgb_metrics = regression_metrics(test_df["label"], xgb_preds)
        log.info("XGBoost totals test metrics: %s", xgb_metrics)
        save_model(db, xgb_bundle, MODEL_NAME_XGB, "total", next_version(db, MODEL_NAME_XGB), xgb_metrics, xgb_bundle["columns"])

        winner = MODEL_NAME_XGB if xgb_metrics["mae"] < poisson_metrics["mae"] else MODEL_NAME_POISSON
        log.info("Lower MAE on held-out test set: %s", winner)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]))
    else:
        print("Usage: python -m models.train_totals TRAIN_START TEST_START TEST_END")
