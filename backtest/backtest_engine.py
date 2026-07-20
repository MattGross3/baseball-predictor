"""
Backtesting framework (Section 11).

**Critical rule the spec calls out and this module enforces**: train/test
splits are always by date, never a random shuffle - `run_backtest` only
ever evaluates a model against games *after* its training window (the
model registry entry's `trained_at` combined with the caller-supplied
`start_date` of the backtest window is the caller's responsibility to get
right; this function itself just scores predictions against actuals for
whatever date range you give it, so always pass a range that's disjoint
from - and later than - what the model was trained on).

ROI/CLV numbers require odds_snapshots to exist for the games in range,
which requires ODDS_API_KEY to have been configured while those games were
upcoming (odds aren't retroactively available on the free tier - see
ingestion/odds_api.py). Without that, roi_* and clv_avg come back None
rather than a fabricated number.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backtest.clv_tracker import american_to_implied_prob, compute_clv_for_range
from database.models import Game, ModelRegistryEntry, OddsSnapshot
from features.build_feature_matrix import build_training_matrix
from models.model_utils import classification_metrics, load_model, regression_metrics

log = logging.getLogger(__name__)

FLAT_BET_SIZE = 100.0
KELLY_FRACTION_CAP = 0.25  # cap Kelly stake at 25% of bankroll per bet - full Kelly is too aggressive for a single game's variance
MIN_EDGE_TO_BET = 0.02  # require at least a 2-point edge over market-implied probability before "placing" a bet


def _latest_registry_entry(db: Session, model_name: str) -> ModelRegistryEntry:
    entry = db.execute(
        select(ModelRegistryEntry)
        .where(ModelRegistryEntry.model_name == model_name)
        .order_by(ModelRegistryEntry.trained_at.desc())
    ).scalars().first()
    if entry is None:
        raise ValueError(f"No trained model found in registry with name '{model_name}'")
    return entry


def _american_profit_per_dollar(odds: int) -> float:
    return odds / 100 if odds > 0 else 100 / abs(odds)


def _earliest_odds(db: Session, game_id: int) -> OddsSnapshot | None:
    return db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id == game_id).order_by(OddsSnapshot.timestamp)
    ).scalars().first()


def _simulate_moneyline_bets(db: Session, df: pd.DataFrame, y_prob: np.ndarray) -> dict:
    """Flat-bet and Kelly-bet ROI, betting the model's favored side only
    when it beats the market-implied probability by MIN_EDGE_TO_BET."""
    flat_profit = flat_staked = 0.0
    kelly_profit = kelly_staked = 0.0
    n_bets = 0
    bet_records = []

    for i, row in df.reset_index(drop=True).iterrows():
        odds_row = _earliest_odds(db, int(row["game_id"]))
        if odds_row is None or odds_row.moneyline_home is None or odds_row.moneyline_away is None:
            continue

        p_home = float(y_prob[i])
        side, odds, p_model = ("home", odds_row.moneyline_home, p_home) if p_home >= 0.5 else ("away", odds_row.moneyline_away, 1 - p_home)
        p_implied = american_to_implied_prob(odds)
        edge = p_model - p_implied
        if edge < MIN_EDGE_TO_BET:
            continue

        won = (row["label"] == 1 and side == "home") or (row["label"] == 0 and side == "away")
        profit_per_dollar = _american_profit_per_dollar(odds)

        flat_staked += FLAT_BET_SIZE
        flat_profit += FLAT_BET_SIZE * profit_per_dollar if won else -FLAT_BET_SIZE

        b = profit_per_dollar
        kelly_pct = max(0.0, min((p_model * (b + 1) - 1) / b, KELLY_FRACTION_CAP)) if b > 0 else 0.0
        kelly_stake = FLAT_BET_SIZE * kelly_pct  # sized against a notional $100 "unit" bankroll per bet, not compounding
        kelly_staked += kelly_stake
        kelly_profit += kelly_stake * profit_per_dollar if won else -kelly_stake

        n_bets += 1
        bet_records.append({"game_id": int(row["game_id"]), "side": side, "odds": odds})

    return {
        "roi_flat_bet": round(flat_profit / flat_staked, 4) if flat_staked else None,
        "roi_kelly": round(kelly_profit / kelly_staked, 4) if kelly_staked else None,
        "n_bets": n_bets,
        "bet_records": bet_records,
    }


def run_backtest(db: Session, model_name: str, start_date: dt.date, end_date: dt.date) -> dict:
    entry = _latest_registry_entry(db, model_name)
    bundle = load_model(entry.file_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]

    target = entry.target_type
    df = build_training_matrix(db, start_date, end_date, target=target)
    if df.empty:
        return {
            "model": model_name, "target_type": target, "date_range": f"{start_date}..{end_date}",
            "accuracy": None, "log_loss": None, "brier_score": None,
            "roi_flat_bet": None, "roi_kelly": None, "clv_avg": None, "n_bets": 0,
        }

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce") if set(feature_cols).issubset(df.columns) else df.reindex(columns=feature_cols)
    # Explicit float cast: bool feature columns (closer_available,
    # lineup_confirmed, roof_closed) otherwise survive as dtype `bool`,
    # and statsmodels' GLM (the Poisson totals baseline) rejects a frame
    # mixing bool/int/float dtypes - see train_totals._prep.
    X = X.reindex(columns=feature_cols, fill_value=0).fillna(0).astype(float)

    result = {"model": model_name, "target_type": target, "date_range": f"{start_date}..{end_date}"}

    if target in ("moneyline", "nrfi"):
        y_prob = model.predict_proba(X)[:, 1]
        result.update(classification_metrics(df["label"], y_prob))

        if target == "moneyline":
            bet_sim = _simulate_moneyline_bets(db, df, y_prob)
            result["roi_flat_bet"] = bet_sim["roi_flat_bet"]
            result["roi_kelly"] = bet_sim["roi_kelly"]
            result["n_bets"] = bet_sim["n_bets"]

            def _bet_side_fn(game, records={r["game_id"]: r for r in bet_sim["bet_records"]}):
                rec = records.get(game.id)
                return (rec["side"], rec["odds"]) if rec else None

            clv_results = compute_clv_for_range(db, start_date, end_date, _bet_side_fn)
            result["clv_avg"] = round(float(np.mean([c["clv_pct"] for c in clv_results])), 3) if clv_results else None
        else:
            result["roi_flat_bet"] = result["roi_kelly"] = result["clv_avg"] = None
            result["n_bets"] = len(df)
    else:  # total: regression. `model` here is one of the compound dicts
        # train_totals.py builds (Poisson: {'home','away','columns'};
        # XGBoost: {'model','columns','nb_r'}), not a bare estimator, so
        # dispatch on which shape it is rather than calling .predict directly.
        from models.train_totals import poisson_run_distribution, xgb_run_distribution

        if "home" in model and "away" in model:
            y_pred = [poisson_run_distribution(model, X.iloc[[i]])["mean"] for i in range(len(X))]
        else:
            y_pred = [xgb_run_distribution(model, X.iloc[[i]])["mean"] for i in range(len(X))]
        reg_metrics = regression_metrics(df["label"], y_pred)
        # regression_metrics returns {"mae", "rmse", "n"} - "n" here means
        # "games scored", not "bets placed" (totals aren't backtested as
        # bets), but n_bets is the one field name the schema/frontend use
        # across both branches, so reuse it rather than adding a second
        # count field just for this target.
        result["n_bets"] = reg_metrics.pop("n")
        result.update(reg_metrics)
        result["roi_flat_bet"] = result["roi_kelly"] = result["clv_avg"] = None

    result.setdefault("n_bets", 0)
    return result
