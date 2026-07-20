"""
Umpire assignments and zone-history aggregation (Section 4.6).

Umpire assignment turns out to be free: the MLB Stats API boxscore includes
an `officials` block naming all four umpires and their position, so
`fetch_umpire_assignment` just reads that - no scraping needed.

There is genuinely no direct API for "how big is this umpire's strike
zone" - the spec calls this out explicitly. `compute_umpire_zone_history`
builds it ourselves: for every game we have on record where this umpire
worked home plate, pull that game's Statcast pitches and measure the
called-strike rate on pitches outside the rulebook zone. This requires
having already ingested games + umpire assignments for the lookback
window, so it degrades to `None` fields (not an exception) when there's
insufficient history rather than blocking the rest of the feature build.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from functools import lru_cache

import pandas as pd
from pybaseball import statcast
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, Umpire
from ingestion.mlb_stats_api import fetch_boxscore

log = logging.getLogger(__name__)

# Simplified rulebook strike zone in feet (Statcast plate_x/plate_z frame).
# Real batter-specific sz_top/sz_bot vary; this fixed box is what
# umpire-scorecard-style analyses commonly use for a first-order estimate.
ZONE_X_HALF_WIDTH = 0.83
ZONE_Z_BOTTOM = 1.5
ZONE_Z_TOP = 3.5

CALLED_STRIKE = "called_strike"
CALLED_BALL = "ball"


def fetch_umpire_assignment(mlb_game_id: int) -> list[dict]:
    """All four umpires for a game, from the boxscore's `officials` block."""
    box = fetch_boxscore(mlb_game_id)
    out = []
    for entry in box.get("officials", []):
        out.append(
            {
                "umpire_id": entry["official"]["id"],
                "name": entry["official"]["fullName"],
                "position": entry["officialType"],
                "home_plate": entry["officialType"] == "Home Plate",
            }
        )
    return out


def ingest_umpire_assignment(db: Session, mlb_game_id: int) -> int:
    game = db.execute(select(Game).where(Game.mlb_game_id == mlb_game_id)).scalar_one_or_none()
    if game is None:
        raise ValueError(f"Game {mlb_game_id} not ingested yet - run ingest_schedule_for_date first")

    written = 0
    for ump in fetch_umpire_assignment(mlb_game_id):
        row = db.execute(
            select(Umpire).where(Umpire.game_id == game.id, Umpire.umpire_name == ump["name"])
        ).scalar_one_or_none()
        if row is None:
            row = Umpire(game_id=game.id, umpire_name=ump["name"])
            db.add(row)
        row.home_plate = ump["home_plate"]
        written += 1
    return written


# Guards _season_league_pitches's actual computation (not just its cache
# dict) - functools.lru_cache only locks around the cache lookup/store,
# not around the wrapped call itself, so two threads racing on the same
# uncached season both start their own full league-wide statcast() pull
# instead of one waiting on the other. That's exactly what happened
# between api/main.py's startup warm-up thread and the first real
# request: both fired the same ~1-2 minute pull concurrently, doubling
# the load on Savant's endpoint and turning it into a 12-minute request.
# The lock makes the second caller block and then get a cheap cache hit
# instead of doing redundant, contending work.
_season_league_pitches_lock = threading.Lock()


@lru_cache(maxsize=8)
def _season_league_pitches_uncached(season: int) -> pd.DataFrame:
    """Whole-season, league-wide Statcast pitch log, fetched once and
    cached in-process - reused for every umpire/game evaluated in that
    season instead of a fresh ~45-day `statcast()` pull per (umpire,
    as_of_date) pair. The old per-call approach was so slow across
    hundreds of historical games that `build_training_matrix` disabled
    this feature for training entirely; this is the same tradeoff and fix
    pattern as `pitcher_features._season_pitcher_pitches`.

    A ~2-week league-wide pull was measured at ~7.7s / ~57k rows / ~66MB,
    so a full ~4.5-month season is a one-time cost of roughly a minute and
    several hundred MB per training run - trivial next to re-paying that
    per umpire-game. Call `_season_league_pitches` (below), not this
    directly, unless you specifically want the race documented above.
    """
    start = dt.date(season, 3, 1)
    end = dt.date(season, 12, 31)
    try:
        return statcast(start.isoformat(), end.isoformat())
    except Exception as exc:
        log.warning("Season-wide Statcast pull failed for %s (%s)", season, exc)
        return pd.DataFrame()


def _season_league_pitches(season: int) -> pd.DataFrame:
    with _season_league_pitches_lock:
        return _season_league_pitches_uncached(season)


def compute_umpire_zone_history(db: Session, umpire_name: str, as_of_date: dt.date, lookback_days: int = 45) -> dict:
    """Strike-zone-size / over-under lean / K-rate boost for a home-plate
    umpire, built from this umpire's prior games in our own DB joined to
    Statcast pitch calls.

    Returns None-filled fields (not an error) when we don't have enough
    history in the DB yet - e.g. a newly-seen call-up umpire, or the app
    hasn't ingested enough of the season.
    """
    start = as_of_date - dt.timedelta(days=lookback_days)
    empty = {"strike_zone_size_percentile": None, "over_under_lean": None, "k_rate_boost": None, "n_games": 0}

    rows = db.execute(
        select(Game.mlb_game_id, Game.date, Game.home_score, Game.away_score)
        .join(Umpire, Umpire.game_id == Game.id)
        .where(Umpire.umpire_name == umpire_name, Umpire.home_plate.is_(True))
        .where(Game.date >= start, Game.date < as_of_date)
        .where(Game.status == "final")
    ).all()
    if not rows:
        return empty

    game_pks = {r.mlb_game_id for r in rows}
    total_runs = [((r.home_score or 0) + (r.away_score or 0)) for r in rows]

    # Slice the cached season pull locally instead of a fresh network call.
    # Lookback windows that cross a calendar-year boundary (as_of_date early
    # enough in the season that `start` falls in the prior year) will miss
    # those prior-year pitches - same edge case already accepted by
    # pitcher_features' season cache, and rare in practice since it only
    # affects the first ~6 weeks of a season.
    season_pitches = _season_league_pitches(as_of_date.year)
    if season_pitches.empty:
        return {**empty, "n_games": len(rows), "over_under_lean": round(sum(total_runs) / len(total_runs), 2)}

    pitch_dates = pd.to_datetime(season_pitches["game_date"]).dt.date
    pitches = season_pitches[(pitch_dates >= start) & (pitch_dates < as_of_date)]
    if pitches.empty:
        return {**empty, "n_games": len(rows)}

    ump_pitches = pitches[pitches["game_pk"].isin(game_pks)]
    called = ump_pitches[ump_pitches["description"].isin([CALLED_STRIKE, CALLED_BALL])].copy()
    if called.empty:
        return {**empty, "n_games": len(rows)}

    called["in_zone"] = (
        called["plate_x"].abs().le(ZONE_X_HALF_WIDTH)
        & called["plate_z"].between(ZONE_Z_BOTTOM, ZONE_Z_TOP)
    )
    out_of_zone = called[~called["in_zone"]]
    generous_pct = (
        100 * (out_of_zone["description"] == CALLED_STRIKE).mean() if len(out_of_zone) else None
    )

    # Combined (both teams) strikeouts per game in this umpire's starts -
    # comparable directly to the ~16-17/game modern MLB average. Earlier
    # this multiplied by 100 (a leftover from treating it as a percentage
    # it never was), which inflated it into meaningless four-digit values.
    k_per_game = (ump_pitches["events"] == "strikeout").sum() / max(len(game_pks), 1)

    return {
        # Higher = umpire calls more strikes on pitches outside the zone
        # (a "pitcher-friendly" / larger effective zone).
        "strike_zone_size_percentile": round(generous_pct, 1) if generous_pct is not None else None,
        "over_under_lean": round(sum(total_runs) / len(total_runs), 2),
        "k_rate_boost": round(k_per_game, 2),
        "n_games": len(rows),
    }
