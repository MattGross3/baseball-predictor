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
RUN_LINE_MIN_EDGE_RUNS = 0.5  # require the model's predicted margin to clear the run line by at least half a run before "placing" a spread bet
CONFIDENCE_THRESHOLD_DEFAULT = 0.6  # see high_confidence_accuracy - >=60% (or <=40%) either direction counts as "confident"


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


def simulate_run_line_bets(db: Session, start_date: dt.date, end_date: dt.date) -> dict:
    """Run-line ("spread") backtest over a date range.

    Deliberately not parameterized by model_name the way run_backtest is -
    a run-line pick needs a predicted *home/away split*, and only the
    Poisson baseline (`totals_poisson`) actually produces one (XGBoost
    predicts one combined number - see features/build_feature_matrix.py's
    split_source fallback for the live-prediction version of this same
    constraint), so this always uses whatever the latest totals_poisson
    entry is.

    Only ever stakes the *home* side: ingestion/odds_api._extract_best_lines
    only stores the home side's run-line price, so betting "away" would
    need a fabricated number. Games where the model favors away covering
    still count toward win/loss (the model's *call* is still checkable
    against the actual margin), just never get a real dollar bet.
    """
    empty = {"roi_flat_bet": None, "n_bets": 0, "wins": 0, "losses": 0, "n_games": 0}
    try:
        entry = _latest_registry_entry(db, "totals_poisson")
    except ValueError:
        return empty
    bundle = load_model(entry.file_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]
    if not ("home" in model and "away" in model):
        return empty  # registry entry isn't actually the Poisson bundle shape

    df = build_training_matrix(db, start_date, end_date, target="total")
    if df.empty:
        return empty

    from models.train_totals import poisson_run_distribution

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce") if set(feature_cols).issubset(df.columns) else df.reindex(columns=feature_cols)
    X = X.reindex(columns=feature_cols, fill_value=0).fillna(0).astype(float)

    flat_profit = flat_staked = 0.0
    n_bets = wins = losses = n_games = 0

    for i, row in df.reset_index(drop=True).iterrows():
        odds_row = _earliest_odds(db, int(row["game_id"]))
        if odds_row is None or odds_row.run_line is None:
            continue

        dist = poisson_run_distribution(model, X.iloc[[i]])
        predicted_margin = dist["lambda_home"] - dist["lambda_away"]
        actual_margin = row["home_score"] - row["away_score"]

        # run_line is the home team's own line (see
        # ingestion/odds_api._extract_best_lines) - home covers whenever
        # margin + run_line > 0, for either the predicted or actual margin.
        edge = predicted_margin + odds_row.run_line
        pick_home_covers = edge > 0
        actual_home_covers = (actual_margin + odds_row.run_line) > 0

        n_games += 1
        if pick_home_covers == actual_home_covers:
            wins += 1
        else:
            losses += 1

        if not pick_home_covers or abs(edge) < RUN_LINE_MIN_EDGE_RUNS or odds_row.run_line_odds is None:
            continue

        profit_per_dollar = _american_profit_per_dollar(odds_row.run_line_odds)
        flat_staked += FLAT_BET_SIZE
        flat_profit += FLAT_BET_SIZE * profit_per_dollar if actual_home_covers else -FLAT_BET_SIZE
        n_bets += 1

    return {
        "roi_flat_bet": round(flat_profit / flat_staked, 4) if flat_staked else None,
        "n_bets": n_bets,
        "wins": wins,
        "losses": losses,
        "n_games": n_games,
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

        # Raw win/loss counts, not just accuracy - lets a caller show a
        # record (e.g. "61-45") or compute a rate of return without odds,
        # which the ROI tab needs as a fallback for stretches with no
        # odds_snapshots coverage (see api/routers/backtest.py's
        # /backtest/roi).
        pred_label = (y_prob >= 0.5).astype(int)
        actual_label = df["label"].to_numpy()
        result["n_games"] = len(df)
        result["wins"] = int((pred_label == actual_label).sum())
        result["losses"] = result["n_games"] - result["wins"]

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


def _total_over_probability(distribution_over_totals: dict[int, float], line: float) -> float:
    """P(actual combined total > line) from a poisson_run_distribution/
    xgb_run_distribution's distribution_over_totals dict - the only way to
    get a probability at all out of a regression target, so this is what
    high_confidence_accuracy uses to define "60% confident" for totals."""
    return sum(p for t, p in distribution_over_totals.items() if t > line)


def high_confidence_accuracy(
    db: Session,
    model_name: str,
    start_date: dt.date,
    end_date: dt.date,
    threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
) -> dict:
    """Accuracy restricted to games where the model's own probability
    clears `threshold` in either direction - a genuinely different, harder
    question than run_backtest's plain accuracy above: is the model
    actually *right more often* on the games it claims to be confident
    about, not just right on average across everything, including the
    genuine coin-flip games where "confidence" isn't meaningful.

    - Moneyline/NRFI: confident whenever predicted_probability >=
      threshold or <= 1-threshold.
    - Totals: has no native probability (it's a regression target) - a
      game only counts if it has a market total line (from
      odds_snapshots), used to compute the model's own implied P(over) /
      P(under) via its run distribution (see _total_over_probability).
      Games with no odds snapshot are excluded from `n_considered`
      entirely, not counted as "not confident" - there's no probability
      to even check without a line to check it against.

    Returns the same {accuracy, log_loss, brier_score, n} shape
    classification_metrics does, computed against "was this confident
    pick actually correct" - which for totals means treating "the model's
    own confidence in its over/under pick" as the probability and "did
    that pick turn out right" as the binary outcome, since there's no
    other way to define calibration for a point-estimate target. Also
    includes `n_considered`, the number of games that had a probability to
    check at all (every game for moneyline/NRFI, only odds-covered games
    for totals) - distinct from `n`, the subset that actually cleared the
    threshold.
    """
    entry = _latest_registry_entry(db, model_name)
    bundle = load_model(entry.file_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]
    target = entry.target_type

    base = {
        "model": model_name, "target_type": target, "date_range": f"{start_date}..{end_date}",
        "threshold": threshold, "accuracy": None, "log_loss": None, "brier_score": None,
        "n": 0, "n_considered": 0,
    }

    df = build_training_matrix(db, start_date, end_date, target=target)
    if df.empty:
        return base

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce") if set(feature_cols).issubset(df.columns) else df.reindex(columns=feature_cols)
    X = X.reindex(columns=feature_cols, fill_value=0).fillna(0).astype(float)

    if target in ("moneyline", "nrfi"):
        y_prob = model.predict_proba(X)[:, 1]
        confident = (y_prob >= threshold) | (y_prob <= 1 - threshold)
        base["n_considered"] = len(df)
        if not confident.any():
            return base
        metrics = classification_metrics(df["label"][confident].to_numpy(), y_prob[confident])
        return {**base, **metrics}

    # total: needs a market line per game to define a probability at all.
    from models.train_totals import poisson_run_distribution, xgb_run_distribution

    is_poisson = "home" in model and "away" in model
    picked_confidence: list[float] = []
    pick_was_correct: list[int] = []
    n_considered = 0

    for i, row in df.reset_index(drop=True).iterrows():
        odds_row = _earliest_odds(db, int(row["game_id"]))
        if odds_row is None or odds_row.total is None:
            continue
        n_considered += 1

        line = float(odds_row.total)
        dist = poisson_run_distribution(model, X.iloc[[i]]) if is_poisson else xgb_run_distribution(model, X.iloc[[i]])
        p_over = _total_over_probability(dist["distribution_over_totals"], line)
        p_side = p_over if p_over >= 0.5 else 1 - p_over
        if p_side < threshold:
            continue

        pick_over = p_over >= 0.5
        actual_over = row["label"] > line
        picked_confidence.append(p_side)
        pick_was_correct.append(int(pick_over == actual_over))

    base["n_considered"] = n_considered
    if not pick_was_correct:
        return base

    # p_side (picked_confidence) is always >= threshold > 0.5 by
    # construction, so classification_metrics' accuracy (which thresholds
    # y_prob at 0.5) reduces to exactly mean(pick_was_correct) - and its
    # log_loss/brier_score become a real calibration check: are 65%
    # confident picks actually right about 65% of the time?
    metrics = classification_metrics(pick_was_correct, picked_confidence)
    return {**base, **metrics}
