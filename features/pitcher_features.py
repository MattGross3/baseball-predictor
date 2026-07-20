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
import threading
from functools import lru_cache

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, PitcherGameLog, Player
from ingestion.fangraphs import estimate_fip
from ingestion.statcast import fetch_pitcher_statcast, summarize_pitcher_statcast

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
    """`include_statcast_trend` used to gate `velo_trend_last_3` purely for
    speed (2 fresh Statcast network pulls per call, prohibitive across
    hundreds of training games) - now that those pulls are cached per
    (pitcher, season) instead of re-fetched per start (see
    `_season_pitcher_pitches`), this stays as an opt-out escape hatch for
    exceptional cases, not something build_training_matrix needs to lean
    on by default anymore.
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

    velo_trend = _velo_trend_last_3(player, as_of_date, starts) if (include_statcast_trend and player) else None

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


# Per-(pitcher, season) locks, not one global lock - unlike the umpire
# league-wide pull (effectively one active season at a time),
# game-detail/backtest requests for many different games can legitimately
# want many different pitchers concurrently, and serializing all of them
# behind a single lock would just recreate the slowness this cache exists
# to avoid. A per-key lock still prevents the specific failure mode this
# guards against: two threads racing on the *same* uncached pitcher both
# firing their own fetch instead of one waiting for the other's result
# (see ingestion.umpire_scorecards._season_league_pitches_lock, where
# that race turned a ~90s request into a 12-minute one).
_pitcher_pitches_locks: dict[tuple[int, int], threading.Lock] = {}
_pitcher_pitches_locks_guard = threading.Lock()


def _pitcher_pitches_lock(key: tuple[int, int]) -> threading.Lock:
    with _pitcher_pitches_locks_guard:
        return _pitcher_pitches_locks.setdefault(key, threading.Lock())


@lru_cache(maxsize=512)
def _season_pitcher_pitches_uncached(pitcher_mlb_id: int, season: int):
    """Whole-season Statcast pitch log for one pitcher, fetched once and
    cached in-process - reused for every start of theirs in a build
    instead of 2 fresh, narrow, shifting-window fetches per start (the
    old approach was so slow across hundreds of historical starts that
    build_training_matrix disabled this feature for training entirely,
    which meant it was 100% None in every training row and got dropped
    from the model's learned feature set - see model_utils.prepare_xy's
    dropna - so real values at live prediction time were silently
    discarded by a model that had never learned any split on this column).

    Also where a real, separate bug got caught and fixed: this used to be
    called with our internal `players.id` (a small sequential integer)
    instead of the pitcher's actual MLB Advanced Media id - Statcast
    always returned zero pitches for that bogus id, so `velo_trend_last_3`
    was silently None even on the "live, full-cost" path, before caching
    was ever a factor. Call `_season_pitcher_pitches` (below), not this
    directly, to get the concurrency-safe wrapper.
    """
    start = dt.date(season, *SEASON_START_MONTH_DAY)
    end = dt.date(season, 12, 31)
    try:
        return fetch_pitcher_statcast(pitcher_mlb_id, start, end)
    except Exception:
        return pd.DataFrame()


def _season_pitcher_pitches(pitcher_mlb_id: int, season: int):
    with _pitcher_pitches_lock((pitcher_mlb_id, season)):
        return _season_pitcher_pitches_uncached(pitcher_mlb_id, season)


def _velo_trend_last_3(player: Player, as_of_date: dt.date, starts) -> float | None:
    """Delta between avg fastball velo over the last 3 starts and the
    season-to-date average, using Statcast pitch-level data. A positive
    value means the pitcher is throwing harder recently than his season
    norm; negative can flag fatigue or an injury risk signal."""
    if not starts or not player.mlb_player_id:
        return None

    season_pitches = _season_pitcher_pitches(player.mlb_player_id, as_of_date.year)
    if season_pitches.empty:
        return None

    # Slice the cached season data locally instead of a new network call -
    # "game_date" is a real date column on every Statcast pitch row.
    pitch_dates = pd.to_datetime(season_pitches["game_date"]).dt.date
    season_window = season_pitches[pitch_dates < as_of_date]
    season_summary = summarize_pitcher_statcast(season_window)

    window_start = starts[min(2, len(starts) - 1)].date
    recent_window = season_pitches[(pitch_dates >= window_start) & (pitch_dates < as_of_date)]
    recent_summary = summarize_pitcher_statcast(recent_window)

    if season_summary["avg_velo"] is None or recent_summary["avg_velo"] is None:
        return None
    return round(recent_summary["avg_velo"] - season_summary["avg_velo"], 1)
