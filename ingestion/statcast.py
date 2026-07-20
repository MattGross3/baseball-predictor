"""
Baseball Savant / Statcast ingestion (Section 4.2).

There's no official Statcast API - `pybaseball` scrapes the CSV export
endpoint Savant's own site uses, which is the standard approach (and the
one the spec calls out). pybaseball caches raw pulls to disk by default
(~/.pybaseball/cache), so repeated calls for overlapping date ranges are
cheap after the first hit.

This module returns pandas DataFrames / plain dicts - it does not write to
the DB directly. Feature modules (features/pitcher_features.py,
features/batter_features.py) call the `compute_*_summary` functions here
and fold the results into their feature dicts.
"""
from __future__ import annotations

import datetime as dt
import logging

import io

import pandas as pd
import requests
from pybaseball import statcast_batter, statcast_pitcher

log = logging.getLogger(__name__)

SAVANT_OAA_URL = "https://baseballsavant.mlb.com/leaderboard/outs_above_average"

# Statcast's own batted-ball quality classification (`launch_speed_angle`):
# 1 Weak, 2 Topped, 3 Under, 4 Flare/Burner, 5 Solid Contact, 6 Barrel.
BARREL_CODE = 6
HARD_HIT_MPH = 95.0
SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "missed_bunt",
    "foul_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


def fetch_pitcher_statcast(pitcher_id: int, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Pitch-level Statcast rows for one pitcher over a date range."""
    df = statcast_pitcher(start_date.isoformat(), end_date.isoformat(), player_id=pitcher_id)
    return df if df is not None else pd.DataFrame()


def fetch_batter_statcast(batter_id: int, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Pitch-level Statcast rows for one batter over a date range."""
    df = statcast_batter(start_date.isoformat(), end_date.isoformat(), player_id=batter_id)
    return df if df is not None else pd.DataFrame()


def compute_pitcher_statcast_summary(pitcher_id: int, as_of_date: dt.date, lookback_days: int = 30) -> dict:
    """Rolling velocity/spin/whiff/pitch-mix summary as of a date.

    Returns zeros/None-filled dict (never raises) if no pitches are found,
    so callers can build a full feature row even for pitchers with no
    recent Statcast history (e.g. rookies, long IL stints).
    """
    start = as_of_date - dt.timedelta(days=lookback_days)
    end = as_of_date - dt.timedelta(days=1)  # never include the target date itself - avoid leakage
    df = fetch_pitcher_statcast(pitcher_id, start, end)

    empty = {
        "avg_velo": None,
        "avg_spin_rate": None,
        "whiff_pct": None,
        "swing_pct": None,
        "primary_pitch_type": None,
        "pitch_mix": {},
        "n_pitches": 0,
    }
    if df.empty:
        return empty

    swings = df[df["description"].isin(SWING_DESCRIPTIONS)]
    whiffs = df[df["description"].isin(WHIFF_DESCRIPTIONS)]

    pitch_mix = (df["pitch_type"].value_counts(normalize=True) * 100).round(1).to_dict()
    primary = df["pitch_type"].value_counts().idxmax() if not df["pitch_type"].dropna().empty else None

    return {
        "avg_velo": _safe_round(df["release_speed"].mean()),
        "avg_spin_rate": _safe_round(df["release_spin_rate"].mean()) if "release_spin_rate" in df else None,
        "whiff_pct": _safe_round(100 * len(whiffs) / len(swings)) if len(swings) else None,
        "swing_pct": _safe_round(100 * len(swings) / len(df)),
        "primary_pitch_type": primary,
        "pitch_mix": pitch_mix,
        "n_pitches": int(len(df)),
    }


def compute_batter_statcast_summary(batter_id: int, as_of_date: dt.date, lookback_days: int = 30) -> dict:
    """Rolling exit-velo/barrel%/launch-angle summary as of a date."""
    start = as_of_date - dt.timedelta(days=lookback_days)
    end = as_of_date - dt.timedelta(days=1)
    df = fetch_batter_statcast(batter_id, start, end)

    empty = {
        "avg_exit_velo": None,
        "barrel_pct": None,
        "hard_hit_pct": None,
        "avg_launch_angle": None,
        "n_batted_balls": 0,
    }
    if df.empty:
        return empty

    batted = df.dropna(subset=["launch_speed"])
    if batted.empty:
        return empty

    barrels = batted[batted.get("launch_speed_angle") == BARREL_CODE] if "launch_speed_angle" in batted else batted.iloc[0:0]
    hard_hit = batted[batted["launch_speed"] >= HARD_HIT_MPH]

    return {
        "avg_exit_velo": _safe_round(batted["launch_speed"].mean()),
        "barrel_pct": _safe_round(100 * len(barrels) / len(batted)),
        "hard_hit_pct": _safe_round(100 * len(hard_hit) / len(batted)),
        "avg_launch_angle": _safe_round(batted["launch_angle"].mean()),
        "n_batted_balls": int(len(batted)),
    }


def fetch_team_defense_oaa(season: int) -> pd.DataFrame:
    """Team-level Outs Above Average for a season, straight from Baseball
    Savant's leaderboard CSV export. Unlike FanGraphs, Savant's leaderboard
    CSV export doesn't block unauthenticated requests (as of this build) -
    but treat that as fragile: catch failures and return an empty frame
    rather than letting one blocked/renamed endpoint take down feature
    building for every team.
    """
    try:
        resp = requests.get(
            SAVANT_OAA_URL,
            params={
                "type": "Fielding_Team",
                "startYear": season,
                "endYear": season,
                "split": "no",
                "team": "",
                "range": "year",
                "min": "q",
                "pos": "",
                "roles": "",
                "viz": "hide",
                "csv": "true",
            },
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        log.warning("Baseball Savant team OAA fetch failed for %s (%s)", season, exc)
        return pd.DataFrame()


def _safe_round(value, digits: int = 1):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)
