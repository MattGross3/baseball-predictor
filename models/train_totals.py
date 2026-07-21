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
from models.model_utils import (
    date_split,
    feature_columns,
    next_version,
    regression_metrics,
    save_model,
    seasonal_walk_forward_splits,
    summarize_walk_forward,
    walk_forward_splits_by_games,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME_POISSON = "totals_poisson"
MODEL_NAME_XGB = "totals_xgboost"
MAX_TOTAL_RUNS = 30  # distribution support: 0..MAX_TOTAL_RUNS combined runs

# Walk-forward validation reporting (see model_utils.walk_forward_splits_by_games
# and train_moneyline.py's identical convention) - 5 folds of 150 games each,
# counted in games rather than calendar days so a fold can't land in the
# Nov-Mar off-season and come back empty.
WALK_FORWARD_N_SPLITS = 5
WALK_FORWARD_TEST_SIZE_GAMES = 150


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cols = feature_columns(df)
    # Boolean feature columns (closer_available, lineup_confirmed,
    # roof_closed) survive pd.to_numeric as dtype `bool`, not `bool`->float -
    # a DataFrame that mixes bool/int64/float64 columns trips statsmodels'
    # stricter-than-sklearn dtype check in GLM. Force a uniform float matrix.
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    medians = X.median(numeric_only=True)
    return X.fillna(medians).astype(float), medians.to_dict()


def train_poisson_baseline(train_df: pd.DataFrame) -> dict:
    """Returns {'home': fitted GLM, 'away': fitted GLM, 'columns': [...],
    'medians': {...}}. `medians` is the training-set per-column median,
    persisted so live-serving/backtest calls to poisson_run_distribution
    below impute missing values the same way training did, not with 0."""
    X, medians = _prep(train_df)
    X_const = sm.add_constant(X, has_constant="add")

    home_model = sm.GLM(train_df["home_score"], X_const, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(train_df["away_score"], X_const, family=sm.families.Poisson()).fit()
    return {"home": home_model, "away": away_model, "columns": list(X.columns), "medians": medians}


def poisson_run_distribution(poisson_models: dict, feature_row: pd.DataFrame) -> dict:
    """Convolve the two independent Poisson(lambda_home), Poisson(lambda_away)
    distributions into a distribution over combined totals 0..MAX_TOTAL_RUNS."""
    # `feature_row` may be a raw DataFrame slice (e.g. train_totals.run()
    # passes test_df.iloc[[i]] straight through) that still has bool-dtype
    # columns - reindex first, then force numeric/float uniformly so
    # statsmodels doesn't choke on a mixed-dtype frame (see _prep).
    #
    # Missing/absent features are filled with this bundle's training-set
    # median, not 0 - 0 reads to the model as "elite" for stats like
    # ERA/win_pct, not "unknown", which is exactly wrong for the thin-data
    # cases (rookies, call-ups, an ingestion hiccup) this matters most for.
    # Falls back to 0 only for legacy bundles saved before medians were
    # persisted (no "medians" key).
    medians = poisson_models.get("medians", {})
    X = feature_row.reindex(columns=poisson_models["columns"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(medians)).fillna(0).astype(float)
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
    X, medians = _prep(train_df)
    y = train_df["label"]
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        min_child_weight=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X, y)

    residuals = y - model.predict(X)
    # Negative binomial dispersion from the variance-to-mean ratio of the
    # residual-adjusted counts: var = mean + mean^2/r  =>  r = mean^2/(var-mean).
    resid_mean = max(float(y.mean()), 1e-6)
    resid_var = max(float(residuals.var()) + resid_mean, resid_mean + 1e-6)
    r = resid_mean**2 / max(resid_var - resid_mean, 1e-6)

    return {"model": model, "columns": list(X.columns), "nb_r": r, "medians": medians}


def xgb_run_distribution(xgb_bundle: dict, feature_row: pd.DataFrame) -> dict:
    # See poisson_run_distribution above for why this fills with the
    # bundle's training-set median, not 0.
    medians = xgb_bundle.get("medians", {})
    X = feature_row.reindex(columns=xgb_bundle["columns"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(medians)).fillna(0).astype(float)
    mean = max(float(xgb_bundle["model"].predict(X)[0]), 0.1)

    r = xgb_bundle["nb_r"]
    p = r / (r + mean)  # NB parameterized by (r, p) with mean = r*(1-p)/p
    support = np.arange(0, MAX_TOTAL_RUNS + 1)
    pmf = stats.nbinom.pmf(support, r, p)
    pmf = pmf / pmf.sum()

    return {"mean": round(mean, 2), "distribution_over_totals": {int(t): round(float(prob), 4) for t, prob in zip(support, pmf)}}


def _poisson_fold_preds(poisson_models: dict, test_fold: pd.DataFrame) -> list[float]:
    return [poisson_run_distribution(poisson_models, test_fold.iloc[[i]])["mean"] for i in range(len(test_fold))]


def _xgb_fold_preds(xgb_bundle: dict, test_fold: pd.DataFrame):
    X_test, _ = _prep(test_fold)
    return xgb_bundle["model"].predict(X_test.reindex(columns=xgb_bundle["columns"], fill_value=0))


def _walk_forward_metrics(df: pd.DataFrame, train_fn, predict_fn) -> list[dict]:
    """Trains `train_fn(train_fold) -> bundle` on each walk-forward fold
    (model_utils.walk_forward_splits_by_games) and scores it with
    `predict_fn(bundle, test_fold) -> point predictions` on that fold's own
    held-out window - a fresh model per fold, not the single production
    model saved by run() below. Mirrors train_moneyline.py's
    identical-in-spirit helper, just for regression (mean total runs)
    instead of classification."""
    fold_metrics = []
    for train_fold, test_fold in walk_forward_splits_by_games(df, WALK_FORWARD_N_SPLITS, WALK_FORWARD_TEST_SIZE_GAMES):
        bundle = train_fn(train_fold)
        preds = predict_fn(bundle, test_fold)
        fold_metrics.append(regression_metrics(test_fold["label"], preds))
    return fold_metrics


def _seasonal_metrics(df: pd.DataFrame, train_fn, predict_fn) -> list[tuple[int, dict]]:
    """See train_moneyline._seasonal_metrics - identical pattern, one fold
    per season (model_utils.seasonal_walk_forward_splits), labeled by the
    season tested rather than averaged together."""
    results = []
    for train_fold, test_fold in seasonal_walk_forward_splits(df):
        season = int(pd.to_datetime(test_fold["date"]).dt.year.iloc[0])
        bundle = train_fn(train_fold)
        preds = predict_fn(bundle, test_fold)
        results.append((season, regression_metrics(test_fold["label"], preds)))
    return results


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
        save_model(db, poisson_models, MODEL_NAME_POISSON, "total", next_version(db, MODEL_NAME_POISSON), poisson_metrics, poisson_models["columns"], feature_medians=poisson_models["medians"])

        log.info("Training XGBoost totals model...")
        xgb_bundle = train_xgb_totals(train_df)
        X_test_xgb, _ = _prep(test_df)
        xgb_preds = xgb_bundle["model"].predict(X_test_xgb.reindex(columns=xgb_bundle["columns"], fill_value=0))
        xgb_metrics = regression_metrics(test_df["label"], xgb_preds)
        log.info("XGBoost totals test metrics: %s", xgb_metrics)
        save_model(db, xgb_bundle, MODEL_NAME_XGB, "total", next_version(db, MODEL_NAME_XGB), xgb_metrics, xgb_bundle["columns"], feature_medians=xgb_bundle["medians"])

        winner = MODEL_NAME_XGB if xgb_metrics["mae"] < poisson_metrics["mae"] else MODEL_NAME_POISSON
        log.info("Lower MAE on held-out test set: %s", winner)

        # Walk-forward validation - see train_moneyline.py's identical block
        # for why this matters: one single-split number could be a lucky or
        # unlucky test window.
        log.info("Running walk-forward validation (%d folds x %d games)...", WALK_FORWARD_N_SPLITS, WALK_FORWARD_TEST_SIZE_GAMES)
        poisson_wf = summarize_walk_forward(_walk_forward_metrics(df, train_poisson_baseline, _poisson_fold_preds))
        xgb_wf = summarize_walk_forward(_walk_forward_metrics(df, train_xgb_totals, _xgb_fold_preds))
        log.info("Poisson baseline - single split: %s | walk-forward: %s", poisson_metrics, poisson_wf or "not enough history for a walk-forward fold")
        log.info("XGBoost totals - single split: %s | walk-forward: %s", xgb_metrics, xgb_wf or "not enough history for a walk-forward fold")

        # Seasonal validation - see train_moneyline.py's identical block:
        # answers whether accuracy holds steady across seasons, which the
        # walk-forward block above can't (its lookback never reaches past
        # the last few months of games).
        log.info("Running seasonal walk-forward validation (one fold per season after the first)...")
        poisson_seasonal = _seasonal_metrics(df, train_poisson_baseline, _poisson_fold_preds)
        xgb_seasonal = _seasonal_metrics(df, train_xgb_totals, _xgb_fold_preds)
        if not poisson_seasonal:
            log.info("Only one season in range - no seasonal comparison possible yet")
        for season, metrics in poisson_seasonal:
            log.info("Poisson baseline - season %d tested: %s", season, metrics)
        for season, metrics in xgb_seasonal:
            log.info("XGBoost totals - season %d tested: %s", season, metrics)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]), dt.date.fromisoformat(sys.argv[3]))
    else:
        print("Usage: python -m models.train_totals TRAIN_START TEST_START TEST_END")
