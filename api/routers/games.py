"""GET /games/* routes (Section 8)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import GameFeaturesOut, GameOut, GamePredictionsOut, GameSlateSummaryOut, GameSyncOut, OddsOut, PredictionOut
from backtest.clv_tracker import american_to_implied_prob, devig_two_way
from database.db import get_db
from database.models import Game, GameFeatureCache, OddsSnapshot, Player, Prediction
from features.build_feature_matrix import build_game_feature_row
from features.pitcher_features import compute_starter_features

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/today", response_model=list[GameOut])
def games_today(date: dt.date | None = None, db: Session = Depends(get_db)):
    """`date` defaults to today; pass it explicitly to see any date's slate
    (handy in dev/backtesting when "today" in the data isn't today)."""
    target_date = date or dt.date.today()
    games = db.execute(select(Game).where(Game.date == target_date).order_by(Game.start_time)).scalars().all()
    return games


@router.post("/sync", response_model=GameSyncOut)
def sync_games(days_ahead: int = 3):
    """On-demand schedule pull (Section 10's job_morning_schedule), for
    whenever a user doesn't want to wait for the scheduler process - the
    common case in a dev environment where scheduler/daily_jobs.py isn't
    running continuously, which is exactly why a newly-scheduled day's
    games (e.g. tomorrow's slate) can be invisible until this runs.

    Reuses scripts.backfill_data.backfill_date per day (today through
    today + days_ahead) - it always re-pulls the schedule first (an
    idempotent upsert, safe to call repeatedly) and, for any games that
    have since gone final, also backfills box scores/lineups/linescores/
    umpires in the same pass - so one click both brings in newly-scheduled
    games and catches up on recently-finished ones. Each day manages its
    own DB session internally (see backfill_date), so this doesn't need
    the shared request-scoped session.
    """
    from scripts.backfill_data import backfill_date

    today = dt.date.today()
    total_games = 0
    for offset in range(days_ahead + 1):
        stats = backfill_date(today + dt.timedelta(days=offset))
        total_games += stats["games"]

    end_date = today + dt.timedelta(days=days_ahead)
    return GameSyncOut(
        days_synced=days_ahead + 1,
        games_seen=total_games,
        message=f"Synced {today} through {end_date} - {total_games} game{'s' if total_games != 1 else ''} seen.",
    )


@router.get("/{game_id}", response_model=GameOut)
def get_game(game_id: int, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, f"No game with id={game_id}")
    return game


@router.get("/{game_id}/features", response_model=GameFeaturesOut)
def get_game_features(game_id: int, refresh: bool = False, db: Session = Depends(get_db)):
    """Full feature breakdown - what the model sees for this game. Powers
    the dashboard's "why does the model like this side" detail view
    (Section 9).

    Building this row does several DB queries plus, for the live-Statcast
    features, real network calls - slow enough on every single page view
    that it was the dominant source of "buffering" complaints on Game
    Detail. Cached in `game_feature_cache` and served from there unless
    `refresh=true` is passed; a completed game's features never change,
    and a scheduled game's are cheap to intentionally refresh but not
    worth recomputing on every view.
    """
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, f"No game with id={game_id}")

    cached = db.get(GameFeatureCache, game_id)
    if cached is not None and not refresh:
        return GameFeaturesOut(game_id=game_id, features=cached.features_json, computed_at=cached.computed_at)

    try:
        nested = build_game_feature_row(db, game_id)
    except Exception as exc:
        raise HTTPException(500, f"Failed building features: {exc}") from exc

    if cached is None:
        cached = GameFeatureCache(game_id=game_id)
        db.add(cached)
    # jsonable_encoder, not the raw dict - `nested` has real `date`/`Decimal`
    # values that stdlib json.dumps (what the JSON column type uses) can't
    # serialize directly, unlike FastAPI's own response encoding which
    # handles this automatically.
    cached.features_json = jsonable_encoder(nested)
    cached.computed_at = dt.datetime.now(dt.timezone.utc)
    db.commit()

    return GameFeaturesOut(game_id=game_id, features=nested, computed_at=cached.computed_at)


@router.get("/today/summary", response_model=list[GameSlateSummaryOut])
def get_games_today_summary(date: dt.date | None = None, db: Session = Depends(get_db)):
    target_date = date or dt.date.today()
    games = db.execute(select(Game).where(Game.date == target_date).order_by(Game.start_time)).scalars().all()

    summaries: list[GameSlateSummaryOut] = []
    for game in games:
        predictions = db.execute(
            select(Prediction).where(Prediction.game_id == game.id).order_by(Prediction.created_at.desc())
        ).scalars().all()

        moneyline = _preferred_prediction(predictions, "moneyline")
        total = _preferred_prediction(predictions, "total")
        nrfi = _preferred_prediction(predictions, "nrfi")

        latest_odds = db.execute(
            select(OddsSnapshot).where(OddsSnapshot.game_id == game.id).order_by(OddsSnapshot.timestamp.desc())
        ).scalars().first()

        # The preferred total model (XGBoost) predicts one combined number,
        # not a per-side split - only the Poisson baseline models each
        # side as its own distribution and can say "3.4 home, 5.1 away".
        # Fall back to whichever total prediction actually has a split
        # rather than showing "-" for the score breakdown just because the
        # more-accurate overall model can't produce one.
        split_source = total if (total and total.predicted_home_value is not None) else next(
            (p for p in predictions if p.target_type == "total" and p.predicted_home_value is not None), None
        )

        # split_source is a genuinely different model from `total` whenever
        # the fallback above kicked in - Poisson's own combined number
        # (predicted_home_value + predicted_away_value) doesn't necessarily
        # match XGBoost's `total_prediction`, the number the Over/Under
        # pick below is actually computed against. Showing Poisson's raw,
        # unscaled split next to a pick decided by a different number the
        # user never sees reads as a contradiction (e.g. a 7.7-run split
        # displayed next to a highlighted "Under 9" pick that was actually
        # decided by an invisible 8.5). Rescale the split proportionally so
        # it always sums to exactly the headline number the pick used -
        # this keeps Poisson's home/away *ratio* (the only real signal for
        # how a total divides between two teams) while never displaying a
        # split total that disagrees with the total actually driving the pick.
        home_split, away_split = None, None
        if split_source is not None:
            fallback_combined = split_source.predicted_home_value + split_source.predicted_away_value
            headline = total.predicted_value if (total and total.predicted_value is not None) else fallback_combined
            if fallback_combined:
                scale = headline / fallback_combined
                home_split = round(split_source.predicted_home_value * scale, 2)
                away_split = round(split_source.predicted_away_value * scale, 2)

        # Pitching matchup - include_statcast_trend=False keeps this to a
        # cheap DB aggregation (season ERA/WHIP), not a live Statcast pull;
        # fine for a per-game feature build, too slow to repeat for every
        # game on the slate on every page load.
        home_starter = db.get(Player, game.home_starter_id) if game.home_starter_id else None
        away_starter = db.get(Player, game.away_starter_id) if game.away_starter_id else None
        home_pitcher_stats = (
            compute_starter_features(db, game.home_starter_id, game.date, include_statcast_trend=False)
            if game.home_starter_id else None
        )
        away_pitcher_stats = (
            compute_starter_features(db, game.away_starter_id, game.date, include_statcast_trend=False)
            if game.away_starter_id else None
        )

        summary = GameSlateSummaryOut(game_id=game.id)
        summary.moneyline_probability = moneyline.predicted_probability if moneyline and moneyline.predicted_probability is not None else None
        summary.total_prediction = total.predicted_value if total and total.predicted_value is not None else None
        summary.total_home_prediction = home_split
        summary.total_away_prediction = away_split
        summary.nrfi_probability = nrfi.predicted_probability if nrfi and nrfi.predicted_probability is not None else None
        summary.latest_odds = latest_odds
        summary.home_starter_name = home_starter.name if home_starter else None
        summary.home_starter_era = home_pitcher_stats["era_season"] if home_pitcher_stats else None
        summary.home_starter_whip = home_pitcher_stats["whip_season"] if home_pitcher_stats else None
        summary.away_starter_name = away_starter.name if away_starter else None
        summary.away_starter_era = away_pitcher_stats["era_season"] if away_pitcher_stats else None
        summary.away_starter_whip = away_pitcher_stats["whip_season"] if away_pitcher_stats else None

        # Run line ("spread") pick: which side the model favors against the
        # market's line, from the same predicted home/away run values the
        # totals model already produces - no separate model needed, since
        # "predicted home runs minus predicted away runs" is exactly a
        # predicted margin, directly comparable to a run line.
        if (
            home_split is not None and away_split is not None
            and latest_odds is not None and latest_odds.run_line is not None
        ):
            # Use the rescaled split (home_split/away_split), not
            # split_source's raw values - same reasoning as the total
            # split above, so the run-line pick is derived from the same
            # number the Over/Under pick and displayed score use, not a
            # third, invisible one.
            predicted_margin = home_split - away_split
            # run_line is always the home team's own line (see
            # ingestion/odds_api._extract_best_lines) - e.g. -1.5 means home
            # is favored by 1.5, +1.5 means home is the underdog getting
            # 1.5. Home covers whenever actual_margin + run_line > 0;
            # this is that same comparison using the model's predicted
            # margin instead of the actual one.
            run_line_edge = round(predicted_margin + latest_odds.run_line, 4)
            summary.run_line_pick_side = "home" if run_line_edge > 0 else "away"
            summary.run_line_edge = run_line_edge

        if moneyline and moneyline.predicted_probability is not None and latest_odds is not None and latest_odds.moneyline_home is not None and latest_odds.moneyline_away is not None:
            # Devigged, not raw, implied probability - a book's home/away
            # prices always sum to >100% (the vig), so comparing the
            # model's probability to one side's raw number systematically
            # understates the edge actually needed to beat the market.
            home_fair, away_fair = devig_two_way(
                american_to_implied_prob(latest_odds.moneyline_home),
                american_to_implied_prob(latest_odds.moneyline_away),
            )
            if moneyline.predicted_probability >= 0.5 and (moneyline.predicted_probability - home_fair) >= 0.02:
                summary.pick_type = "moneyline"
                summary.pick_side = "home"
                summary.projected_value = moneyline.predicted_probability
                summary.market_value = home_fair
                summary.edge = round(moneyline.predicted_probability - home_fair, 4)
                summary.confidence = round(abs(moneyline.predicted_probability - 0.5) * 2, 4)
            elif (1 - moneyline.predicted_probability) >= 0.5 and ((1 - moneyline.predicted_probability) - away_fair) >= 0.02:
                summary.pick_type = "moneyline"
                summary.pick_side = "away"
                summary.projected_value = 1 - moneyline.predicted_probability
                summary.market_value = away_fair
                summary.edge = round((1 - moneyline.predicted_probability) - away_fair, 4)
                summary.confidence = round(abs((1 - moneyline.predicted_probability) - 0.5) * 2, 4)

        if summary.pick_type is None and total and total.predicted_value is not None and latest_odds is not None and latest_odds.total is not None:
            edge = total.predicted_value - float(latest_odds.total)
            if abs(edge) >= 0.5:
                summary.pick_type = "over" if edge > 0 else "under"
                summary.pick_side = "over" if edge > 0 else "under"
                summary.projected_value = total.predicted_value
                summary.market_value = float(latest_odds.total)
                summary.edge = round(edge, 4)
                summary.confidence = round(min(abs(edge) / 3.0, 1.0), 4)

        summaries.append(summary)

    return summaries


@router.get("/{game_id}/predictions", response_model=GamePredictionsOut)
def get_game_predictions(game_id: int, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, f"No game with id={game_id}")

    predictions = db.execute(
        select(Prediction).where(Prediction.game_id == game_id).order_by(Prediction.created_at.desc())
    ).scalars().all()

    edge = _compute_edge_vs_market(db, game_id, predictions)
    return GamePredictionsOut(game_id=game_id, predictions=predictions, edge_vs_market=edge)


@router.get("/{game_id}/odds", response_model=list[OddsOut])
def get_game_odds(game_id: int, db: Session = Depends(get_db)):
    """All odds snapshots for the game, oldest first - the full line-movement
    history (Section 9's LineMovementChart), not just the latest price."""
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, f"No game with id={game_id}")
    snapshots = db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id == game_id).order_by(OddsSnapshot.timestamp)
    ).scalars().all()
    return snapshots


def _preferred_prediction(predictions: list[Prediction], target: str) -> Prediction | None:
    candidates = [p for p in predictions if p.target_type == target]
    order = {
        "moneyline": ["moneyline_xgboost", "moneyline_logistic"],
        "nrfi": ["nrfi_logistic", "nrfi_xgboost"],
        "total": ["totals_xgboost", "totals_poisson"],
    }
    for name in order.get(target, []):
        match = next((p for p in candidates if p.model_name == name), None)
        if match is not None:
            return match
    return candidates[0] if candidates else None


def _compute_edge_vs_market(db: Session, game_id: int, predictions: list[Prediction]) -> dict | None:
    """Model win probability vs. the market's de-vigged fair probability
    from the latest odds snapshot. Returns None if there's no moneyline
    prediction or no odds for both sides - the dashboard shows "N/A" in
    that case rather than a fabricated edge.
    """
    moneyline_pred = next((p for p in predictions if p.target_type == "moneyline"), None)
    if moneyline_pred is None or moneyline_pred.predicted_probability is None:
        return None

    latest_odds = db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id == game_id).order_by(OddsSnapshot.timestamp.desc())
    ).scalars().first()
    if latest_odds is None or latest_odds.moneyline_home is None or latest_odds.moneyline_away is None:
        return None

    home_fair, _away_fair = devig_two_way(
        american_to_implied_prob(latest_odds.moneyline_home),
        american_to_implied_prob(latest_odds.moneyline_away),
    )
    model_prob_home = moneyline_pred.predicted_probability

    # Expected ROI: a real expected-value calculation (not just the
    # fair-probability edge above) - bet $1 on whichever side the model
    # favors, at that side's *actual* market price (not the devigged fair
    # price; you can only ever bet the real number a book offers).
    if model_prob_home >= 0.5:
        price, win_prob = latest_odds.moneyline_home, model_prob_home
    else:
        price, win_prob = latest_odds.moneyline_away, 1 - model_prob_home
    profit_per_dollar = price / 100 if price > 0 else 100 / abs(price)
    expected_roi = round(win_prob * profit_per_dollar - (1 - win_prob), 4)

    return {
        "model_probability_home": model_prob_home,
        "market_implied_probability_home": round(home_fair, 4),
        "edge": round(model_prob_home - home_fair, 4),
        "expected_roi": expected_roi,
    }
