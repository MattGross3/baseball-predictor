"""
Lineup / batter features (Section 6).

Schema caveat: the spec's `batter_game_logs` table (Section 5) only stores
at_bats/hits/hr/bb/k - no doubles/triples/HBP breakdown. True wOBA needs
those, so `_woba_proxy` below is a documented approximation using
standard-ish linear weights applied to what we actually have, not
FanGraphs' real wOBA. Good enough to rank hitters relative to each other
within this app; don't expect it to match FanGraphs' number exactly.
"""
from __future__ import annotations

import datetime as dt
import statistics

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import BatterGameLog, Game, Lineup, Player

# Rough linear weights (unweighted-by-year approximation). BB and H are
# blended since we can't separate 1B/2B/3B from the stored schema.
_W_BB = 0.69
_W_H = 0.89
_W_HR_BONUS = 0.63  # extra credit on top of the flat H weight, since HR >> single


def _woba_proxy(bb: int, h: int, hr: int, ab: int) -> float | None:
    denom = ab + bb
    if not denom:
        return None
    return round((_W_BB * bb + _W_H * h + _W_HR_BONUS * hr) / denom, 3)


def _batter_season_stats(db: Session, player_id: int, as_of_date: dt.date, lookback_days: int | None = None):
    since = as_of_date - dt.timedelta(days=lookback_days) if lookback_days else dt.date(as_of_date.year, 3, 1)
    rows = db.execute(
        select(BatterGameLog)
        .join(Game, Game.id == BatterGameLog.game_id)
        .where(BatterGameLog.player_id == player_id)
        .where(Game.date >= since, Game.date < as_of_date)
        .where(Game.status == "final")
    ).scalars().all()
    ab = sum(r.at_bats or 0 for r in rows)
    h = sum(r.hits or 0 for r in rows)
    bb = sum(r.bb or 0 for r in rows)
    hr = sum(r.hr or 0 for r in rows)
    return {"ab": ab, "h": h, "bb": bb, "hr": hr, "woba_proxy": _woba_proxy(bb, h, hr, ab), "n_games": len(rows)}


def _lineup_for_game(db: Session, game_id: int, team_id: int) -> list[Lineup]:
    return db.execute(
        select(Lineup)
        .where(Lineup.game_id == game_id, Lineup.team_id == team_id)
        .order_by(Lineup.batting_order_position)
    ).scalars().all()


def _projected_lineup(db: Session, team_id: int, as_of_date: dt.date, n: int = 9) -> list[int]:
    """Fallback when the real lineup isn't confirmed yet: the n players
    with the most starts (batting_order_position set) for this team over
    the trailing 30 days, ordered by their most common lineup slot."""
    since = as_of_date - dt.timedelta(days=30)
    rows = db.execute(
        select(Lineup.player_id, Lineup.batting_order_position)
        .join(Game, Game.id == Lineup.game_id)
        .where(Lineup.team_id == team_id)
        .where(Game.date >= since, Game.date < as_of_date)
    ).all()
    if not rows:
        return []
    counts: dict[int, int] = {}
    avg_slot: dict[int, list[int]] = {}
    for player_id, slot in rows:
        counts[player_id] = counts.get(player_id, 0) + 1
        avg_slot.setdefault(player_id, []).append(slot)
    ranked = sorted(counts, key=lambda pid: counts[pid], reverse=True)[:n]
    return sorted(ranked, key=lambda pid: statistics.mean(avg_slot[pid]))


def compute_lineup_features(db: Session, team_id: int, game_id: int, as_of_date: dt.date, opposing_pitcher_hand: str | None = None) -> dict:
    lineup_rows = _lineup_for_game(db, game_id, team_id)
    confirmed = bool(lineup_rows) and all(r.confirmed_at is not None for r in lineup_rows)

    player_ids = [r.player_id for r in lineup_rows] if lineup_rows else _projected_lineup(db, team_id, as_of_date)
    if not player_ids:
        return {
            "lineup_wOBA_weighted_by_order": None,
            "platoon_advantage_count": None,
            "hot_streak_players": [],
            "lineup_confirmed": False,
        }

    # Weight top-of-order more heavily - standard lineup-strength convention:
    # leadoff/2/3/4 hitters see more plate appearances per game than 7-9.
    order_weights = [1.15, 1.1, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8]

    weighted_sum = 0.0
    weight_total = 0.0
    platoon_count = 0
    hot_streak = []

    for idx, player_id in enumerate(player_ids):
        season = _batter_season_stats(db, player_id, as_of_date)
        recent = _batter_season_stats(db, player_id, as_of_date, lookback_days=7)
        weight = order_weights[idx] if idx < len(order_weights) else 0.8

        if season["woba_proxy"] is not None:
            weighted_sum += season["woba_proxy"] * weight
            weight_total += weight

        player = db.get(Player, player_id)
        if opposing_pitcher_hand and player and player.bats:
            favorable = (
                player.bats == "S"
                or (player.bats == "L" and opposing_pitcher_hand == "R")
                or (player.bats == "R" and opposing_pitcher_hand == "L")
            )
            if favorable:
                platoon_count += 1

        if season["woba_proxy"] and recent["woba_proxy"] and recent["n_games"] >= 3:
            z = _z_score(recent["woba_proxy"], season["woba_proxy"])
            if z is not None and z > 1.0:
                hot_streak.append({"player_id": player_id, "z_score": z})

    return {
        "lineup_wOBA_weighted_by_order": round(weighted_sum / weight_total, 3) if weight_total else None,
        "platoon_advantage_count": platoon_count if opposing_pitcher_hand else None,
        "hot_streak_players": hot_streak,
        "lineup_confirmed": confirmed,
    }


def _z_score(recent_woba: float, season_woba: float, assumed_league_std: float = 0.08) -> float | None:
    """Simplified z-score: how far the player's last-7-day rate sits from
    their own season rate, in units of a league-typical wOBA spread. A
    true z-score would need per-player game-to-game variance; this is a
    workable proxy given our sample sizes."""
    if assumed_league_std == 0:
        return None
    return round((recent_woba - season_woba) / assumed_league_std, 2)
