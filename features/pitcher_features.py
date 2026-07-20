"""
Starting pitcher features (Section 6).

Every function here is strictly "as of" a date: it only reads games with
`Game.date < as_of_date` (or, for the target game's own opponent, games
strictly before it). Never pass today's own game into the historical
window - that's exactly the leakage the spec calls out.

Deviation from the spec's literal signature: `compute_starter_features`
needs a live DB session (there's no other way to compute rolling stats),
so it takes `db` as an explicit first argument. It also accepts an
optional `opponent_team_id` - the spec lists `vs_opponent_career_era` as
an output field but the given signature `(pitcher_id, as_of_date)` has no
way to know who the opponent is; `build_game_feature_row` always supplies
it, and the field is `None` if you call this standalone without one.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, PitcherGameLog, Player
from ingestion.fangraphs import estimate_fip
from ingestion.statcast import compute_pitcher_statcast_summary

SEASON_START_MONTH_DAY = (3, 1)  # spring training/opening day is always in March


def _season_start(as_of_date: dt.date) -> dt.date:
    return dt.date(as_of_date.year, *SEASON_START_MONTH_DAY)


def _pitcher_logs(db: Session, pitcher_id: int, as_of_date: dt.date, since: dt.date | None = None, starts_only: bool = False):
    since = since or _season_start(as_of_date)
    stmt = (
        select(PitcherGameLog, Game.date, Game.home_team_id, Game.away_team_id)
        .join(Game, Game.id == PitcherGameLog.game_id)
        .where(PitcherGameLog.player_id == pitcher_id)
        .where(Game.date >= since, Game.date < as_of_date)
        .where(Game.status == "final")
        .order_by(Game.date.desc())
    )
    if starts_only:
        stmt = stmt.where(PitcherGameLog.is_starter.is_(True))
    return db.execute(stmt).all()


def _era(rows) -> float | None:
    ip = sum((r.PitcherGameLog.ip or 0) for r in rows)
    er = sum((r.PitcherGameLog.er or 0) for r in rows)
    if not ip:
        return None
    return round(9 * er / ip, 2)


def compute_starter_features(
    db: Session,
    pitcher_id: int,
    as_of_date: dt.date,
    opponent_team_id: int | None = None,
    include_statcast_trend: bool = True,
) -> dict:
    """`include_statcast_trend` gates `velo_trend_last_3`, which costs 2
    live Statcast/pybaseball network pulls per call. That's negligible for
    a single live prediction (a handful of starters a day) but becomes the
    dominant cost when building a training matrix over hundreds of
    historical games - build_training_matrix passes False for that reason
    and leaves velo_trend_last_3 as None for those rows.
    """
    player = db.get(Player, pitcher_id)
    season_rows = _pitcher_logs(db, pitcher_id, as_of_date)
    starts = [r for r in season_rows if r.PitcherGameLog.is_starter]
    last_3_starts = starts[:3]

    ip_total = sum((r.PitcherGameLog.ip or 0) for r in season_rows)
    hr_total = sum((r.PitcherGameLog.hr or 0) for r in season_rows)
    bb_total = sum((r.PitcherGameLog.bb or 0) for r in season_rows)
    k_total = sum((r.PitcherGameLog.k or 0) for r in season_rows)
    h_total = sum((r.PitcherGameLog.h or 0) for r in season_rows)
    # Approximate plate appearances faced - our schema doesn't store batters
    # faced directly, so PA is estimated as outs recorded + baserunners
    # allowed (hits + walks). Good enough for a rate stat, not exact.
    approx_pa = ip_total * 3 + h_total + bb_total

    home_rows = [r for r in season_rows if r.home_team_id == (player.team_id if player else None)]
    away_rows = [r for r in season_rows if r.away_team_id == (player.team_id if player else None)]

    opponent_rows = []
    if opponent_team_id is not None:
        all_history = _pitcher_logs(db, pitcher_id, as_of_date, since=dt.date(2000, 1, 1))
        opponent_rows = [r for r in all_history if opponent_team_id in (r.home_team_id, r.away_team_id)]

    days_rest = None
    pitch_count_last_start = None
    if starts:
        last_start_date = starts[0].date
        days_rest = (as_of_date - last_start_date).days
        pitch_count_last_start = starts[0].PitcherGameLog.pitch_count

    velo_trend = _velo_trend_last_3(db, pitcher_id, as_of_date, starts) if include_statcast_trend else None

    return {
        "era_season": _era(season_rows),
        "fip_season": estimate_fip(hr_total, bb_total, k_total, ip_total) if ip_total else None,
        "siera_season": None,  # requires FanGraphs' regression model - see ingestion/fangraphs.py
        "era_last_3_starts": _era(last_3_starts),
        "k_pct_rolling": round(100 * k_total / approx_pa, 1) if approx_pa else None,
        "bb_pct_rolling": round(100 * bb_total / approx_pa, 1) if approx_pa else None,
        "velo_trend_last_3": velo_trend,
        "days_rest": days_rest,
        "pitch_count_last_start": pitch_count_last_start,
        "home_away_split_era": {"home": _era(home_rows), "away": _era(away_rows)},
        "vs_opponent_career_era": _era(opponent_rows) if opponent_team_id is not None else None,
        "handedness": player.throws if player else None,
    }


def _velo_trend_last_3(db: Session, pitcher_id: int, as_of_date: dt.date, starts) -> float | None:
    """Delta between avg fastball velo over the last 3 starts and the
    season-to-date average, using Statcast pitch-level data. A positive
    value means the pitcher is throwing harder recently than his season
    norm; negative can flag fatigue or an injury risk signal."""
    if not starts:
        return None

    season_summary = compute_pitcher_statcast_summary(
        pitcher_id, as_of_date, lookback_days=(as_of_date - _season_start(as_of_date)).days
    )
    window_start = starts[min(2, len(starts) - 1)].date
    lookback = (as_of_date - window_start).days + 1
    recent_summary = compute_pitcher_statcast_summary(pitcher_id, as_of_date, lookback_days=max(lookback, 1))

    if season_summary["avg_velo"] is None or recent_summary["avg_velo"] is None:
        return None
    return round(recent_summary["avg_velo"] - season_summary["avg_velo"], 1)
