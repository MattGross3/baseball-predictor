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

    try:
        pitches = statcast(start.isoformat(), (as_of_date - dt.timedelta(days=1)).isoformat())
    except Exception as exc:
        log.warning("Statcast pull failed for umpire zone history (%s) - returning partial result", exc)
        return {**empty, "n_games": len(rows), "over_under_lean": round(sum(total_runs) / len(total_runs), 2)}

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

    k_pct = 100 * (ump_pitches["events"] == "strikeout").sum() / max(len(game_pks), 1)

    return {
        # Higher = umpire calls more strikes on pitches outside the zone
        # (a "pitcher-friendly" / larger effective zone).
        "strike_zone_size_percentile": round(generous_pct, 1) if generous_pct is not None else None,
        "over_under_lean": round(sum(total_runs) / len(total_runs), 2),
        "k_rate_boost": round(k_pct, 2),
        "n_games": len(rows),
    }
