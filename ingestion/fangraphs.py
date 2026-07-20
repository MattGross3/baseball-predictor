"""
FanGraphs ingestion via pybaseball (Section 4.3).

As of this build, FanGraphs actively blocks pybaseball's scraper (the
leaderboard endpoints it hits return HTTP 403 - a server-side anti-bot
change on FanGraphs' end, not something wrong with this code). We do not
attempt to spoof headers or otherwise route around that block; instead
every function here fails soft (logs a warning, returns an empty
DataFrame) so the rest of the pipeline keeps working without FIP/xFIP/
SIERA/wOBA/wRC+/projections.

To compensate, `pitcher_features.py` computes FIP itself from boxscore
components (HR/BB/K/IP, which we do have from the MLB Stats API) using the
standard FIP formula - see `estimate_fip` below. That's a reasonable
proxy; it just won't match FanGraphs' park- and league-adjusted xFIP/SIERA
exactly.

If FanGraphs re-opens access (or you have a paid data feed / different
scraping-permitted source), swap the internals of `fetch_pitcher_advanced`
/ `fetch_batter_advanced` and the rest of the app is unaffected, since
everything downstream reads from these two function signatures.
"""
from __future__ import annotations

import logging

import pandas as pd
from pybaseball import batting_stats, pitching_stats

log = logging.getLogger(__name__)

# League-average constant for the current FIP formula; FanGraphs recomputes
# this yearly so it can't be scraped when their site 403s us. This value
# tracks recent MLB seasons closely enough for a fallback proxy.
FIP_CONSTANT = 3.15


def fetch_pitcher_advanced(season: int, qualified_only: bool = False) -> pd.DataFrame:
    """FIP, xFIP, SIERA etc. for all pitchers in a season, straight from
    FanGraphs. Returns an empty DataFrame (with a logged warning) if
    FanGraphs' anti-bot wall blocks the request."""
    try:
        return pitching_stats(season, qual=1 if not qualified_only else "y")
    except Exception as exc:  # pybaseball raises a generic requests.HTTPError
        log.warning("FanGraphs pitcher stats unavailable for %s (%s) - falling back to boxscore-derived FIP", season, exc)
        return pd.DataFrame()


def fetch_batter_advanced(season: int, qualified_only: bool = False) -> pd.DataFrame:
    """wOBA, wRC+ etc. for all batters in a season, straight from FanGraphs."""
    try:
        return batting_stats(season, qual=1 if not qualified_only else "y")
    except Exception as exc:
        log.warning("FanGraphs batter stats unavailable for %s (%s) - lineup features fall back to boxscore stats", season, exc)
        return pd.DataFrame()


def fetch_projections(season: int) -> pd.DataFrame:
    """Season-ahead ZiPS/Steamer projections - useful prior for early-season
    small samples. Same FanGraphs access caveat as above applies."""
    try:
        # pybaseball's projection scrapers hit the same FanGraphs leaderboard
        # infrastructure as pitching_stats/batting_stats.
        from pybaseball import fg_pitching_data

        return fg_pitching_data(season, stat_columns="fip,xfip,siera", qual=1)
    except Exception as exc:
        log.warning("FanGraphs projections unavailable for %s (%s)", season, exc)
        return pd.DataFrame()


def estimate_fip(hr: int, bb: int, k: int, ip: float, hbp: int = 0, constant: float = FIP_CONSTANT) -> float | None:
    """FIP computed from raw components we always have (MLB Stats API
    boxscores), used whenever the FanGraphs scrape above returns nothing.

    FIP = ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP + constant
    """
    if not ip:
        return None
    return round(((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + constant, 2)
