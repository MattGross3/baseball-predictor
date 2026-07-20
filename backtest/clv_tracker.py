"""
Closing Line Value tracking (Section 11).

CLV compares the price you'd have bet at to the closing (final pre-game)
line - consistently beating the closing line is the standard proxy for
"the model has real edge" independent of whether any individual bet won,
since results converge slowly but CLV converges fast.

This needs at least 2 odds_snapshots per game (an early price and a
closing price) to mean anything. Without an ODDS_API_KEY configured (see
config.settings.has_odds_key), the scheduler never polls odds, so this
degrades to "insufficient data" rather than raising - same pattern as the
rest of the ingestion layer.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, OddsSnapshot


def american_to_implied_prob(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def compute_clv(db: Session, game_id: int, bet_odds: int, bet_side: str) -> dict | None:
    """CLV for a single bet: bet_side is 'home' or 'away'; bet_odds is the
    American price you got when you placed the bet. Returns None if we
    don't have a later ("closing") snapshot to compare against.
    """
    snapshots = db.execute(
        select(OddsSnapshot).where(OddsSnapshot.game_id == game_id).order_by(OddsSnapshot.timestamp)
    ).scalars().all()
    if len(snapshots) < 2:
        return None

    closing = snapshots[-1]
    closing_odds = closing.moneyline_home if bet_side == "home" else closing.moneyline_away
    if closing_odds is None:
        return None

    bet_implied = _american_to_implied_prob(bet_odds)
    closing_implied = _american_to_implied_prob(closing_odds)
    # Positive CLV = you got a better (higher) implied-probability price
    # than the closing line, i.e. the market moved toward your side after
    # you bet - the textbook definition of beating the close.
    clv_pct = round(100 * (closing_implied - bet_implied), 2)

    return {"game_id": game_id, "bet_odds": bet_odds, "closing_odds": closing_odds, "clv_pct": clv_pct}


def compute_clv_for_range(db: Session, start_date: dt.date, end_date: dt.date, bet_side_fn) -> list[dict]:
    """`bet_side_fn(game) -> ('home'|'away', american_odds) | None` decides
    which side (if any) was bet for each game - typically the model's
    favored side at whatever the earliest available snapshot was, standing
    in for "the price you'd have gotten if you bet the moment the model's
    prediction was made".
    """
    games = db.execute(
        select(Game).where(Game.date >= start_date, Game.date < end_date, Game.status == "final")
    ).scalars().all()

    results = []
    for game in games:
        decision = bet_side_fn(game)
        if decision is None:
            continue
        side, odds = decision
        clv = compute_clv(db, game.id, odds, side)
        if clv:
            results.append(clv)
    return results
