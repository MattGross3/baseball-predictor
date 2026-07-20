"""
Bullpen features (Section 6).

Same "as of" discipline as pitcher_features.py: only games strictly before
`as_of_date` are read.

Note on team attribution: `pitcher_game_logs` doesn't carry a team_id
(that's the given schema from the spec), so "which team did this relief
appearance belong to" is inferred from the game's home/away team versus
the pitcher's *current* `players.team_id`. That's wrong for a pitcher
traded mid-season until their `players` row is updated by the next roster
sync - an accepted limitation of the given schema, not a bug here.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, PitcherGameLog, Player


def _relief_logs(db: Session, team_id: int, as_of_date: dt.date, lookback_days: int):
    since = as_of_date - dt.timedelta(days=lookback_days)
    return db.execute(
        select(PitcherGameLog, Game.date)
        .join(Game, Game.id == PitcherGameLog.game_id)
        .join(Player, Player.id == PitcherGameLog.player_id)
        .where(PitcherGameLog.is_starter.is_(False))
        .where(Player.team_id == team_id)
        .where(Game.date >= since, Game.date < as_of_date)
        .where(Game.status == "final")
        .where((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
        .order_by(Game.date.desc())
    ).all()


def _era(rows) -> float | None:
    ip = sum((r.PitcherGameLog.ip or 0) for r in rows)
    er = sum((r.PitcherGameLog.er or 0) for r in rows)
    if not ip:
        return None
    return round(9 * er / ip, 2)


def compute_bullpen_features(db: Session, team_id: int, as_of_date: dt.date) -> dict:
    rows_7d = _relief_logs(db, team_id, as_of_date, 7)
    rows_14d = _relief_logs(db, team_id, as_of_date, 14)
    rows_3g_window = _relief_logs(db, team_id, as_of_date, 3)  # calendar days, proxy for "last 3 games"

    innings_last_3 = round(sum((r.PitcherGameLog.ip or 0) for r in rows_3g_window), 1)

    closer_available, closer_id = _closer_availability(db, team_id, as_of_date)
    hand_distribution = _hand_distribution(db, team_id, as_of_date)

    return {
        "bullpen_era_rolling_7d": _era(rows_7d),
        "bullpen_era_rolling_14d": _era(rows_14d),
        "innings_thrown_last_3_games": innings_last_3,
        "closer_available": closer_available,
        "bullpen_hand_distribution": hand_distribution,
    }


def _closer_availability(db: Session, team_id: int, as_of_date: dt.date) -> tuple[bool | None, int | None]:
    """Identify the likely closer (most relief appearances in the last 30
    days) and flag him unavailable if he's pitched on each of the last 2
    consecutive calendar days (back-to-back-to-back is a hard no; a single
    prior-day appearance is a soft flag most managers still use, so we
    require 2 straight to call it unavailable)."""
    rows = _relief_logs(db, team_id, as_of_date, 30)
    if not rows:
        return None, None

    appearances: dict[int, list[dt.date]] = {}
    for log_row, game_date in rows:
        appearances.setdefault(log_row.player_id, []).append(game_date)

    closer_id = max(appearances, key=lambda pid: len(appearances[pid]))
    dates = sorted(appearances[closer_id], reverse=True)
    if len(dates) >= 2 and (as_of_date - dates[0]).days <= 1 and (dates[0] - dates[1]).days <= 1:
        return False, closer_id
    return True, closer_id


def _hand_distribution(db: Session, team_id: int, as_of_date: dt.date) -> dict:
    rows = _relief_logs(db, team_id, as_of_date, 30)
    pitcher_ids = {r.PitcherGameLog.player_id for r in rows}
    if not pitcher_ids:
        return {"L": None, "R": None}

    hands = db.execute(select(Player.throws).where(Player.id.in_(pitcher_ids))).scalars().all()
    hands = [h for h in hands if h]
    if not hands:
        return {"L": None, "R": None}
    left = sum(1 for h in hands if h == "L")
    return {"L": round(100 * left / len(hands), 1), "R": round(100 * (len(hands) - left) / len(hands), 1)}
