"""GET /games/* routes (Section 8)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import GameFeaturesOut, GameOut, GamePredictionsOut, OddsOut, PredictionOut
from backtest.clv_tracker import american_to_implied_prob
from database.db import get_db
from database.models import Game, OddsSnapshot, Prediction
from features.build_feature_matrix import build_game_feature_row

router = APIRouter(prefix="/games", tags=["games"])


@router.get("/today", response_model=list[GameOut])
def games_today(date: dt.date | None = None, db: Session = Depends(get_db)):
    """`date` defaults to today; pass it explicitly to see any date's slate
    (handy in dev/backtesting when "today" in the data isn't today)."""
    target_date = date or dt.date.today()
    games = db.execute(select(Game).where(Game.date == target_date).order_by(Game.start_time)).scalars().all()
    return games


@router.get("/{game_id}", response_model=GameOut)
def get_game(game_id: int, db: Session = Depends(get_db)):
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, f"No game with id={game_id}")
    return game


@router.get("/{game_id}/features", response_model=GameFeaturesOut)
def get_game_features(game_id: int, db: Session = Depends(get_db)):
    """Full feature breakdown - what the model sees for this game. Powers
    the dashboard's "why does the model like this side" detail view
    (Section 9)."""
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, f"No game with id={game_id}")
    try:
        nested = build_game_feature_row(db, game_id)
    except Exception as exc:
        raise HTTPException(500, f"Failed building features: {exc}") from exc
    return GameFeaturesOut(game_id=game_id, features=nested)


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


def _compute_edge_vs_market(db: Session, game_id: int, predictions: list[Prediction]) -> dict | None:
    """Model win probability vs. the market's de-vigged... actually just
    raw implied probability (no de-vig applied) from the latest odds
    snapshot. Returns None if there's no moneyline prediction or no odds -
    the dashboard shows "N/A" in that case rather than a fabricated edge.
    """
    moneyline_pred = next((p for p in predictions if p.target_type == "moneyline"), None)
    if moneyline_pred is None or moneyline_pred.predicted_probability is None:
        return None

    latest_odds = db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id == game_id).order_by(OddsSnapshot.timestamp.desc())
    ).scalars().first()
    if latest_odds is None or latest_odds.moneyline_home is None:
        return None

    implied = american_to_implied_prob(latest_odds.moneyline_home)
    return {
        "model_probability_home": moneyline_pred.predicted_probability,
        "market_implied_probability_home": round(implied, 4),
        "edge": round(moneyline_pred.predicted_probability - implied, 4),
    }
