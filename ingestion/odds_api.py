"""
Odds ingestion (Section 4.4), via The Odds API (https://the-odds-api.com/).

Requires ODDS_API_KEY in .env - the free tier is 500 requests/month, a hard
account-wide cap covering every endpoint below, not per-endpoint. Every
function that actually calls out enforces `ingestion.api_budget.within_budget`
first and records the call via `record_call` right at the request site (not
some approximation after the fact), so the count can never drift from what
The Odds API itself sees - see api_budget.py's docstring, and
scheduler/daily_jobs.py's ODDS_POLL_INTERVAL_MINUTES for how the default
polling cadence stays well under the cap by design rather than leaning on
this hard stop to save you.

Without a key, every function here returns an empty result rather than
raising, so the rest of the app (predictions, dashboard) still works -
edge-vs-market columns just show "N/A" instead of a real number.
"""
from __future__ import annotations

import datetime as dt
import logging

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from database.models import Game, OddsSnapshot
from ingestion.api_budget import record_call, within_budget

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
REQUEST_TIMEOUT = 15
API_NAME = "the-odds-api"


def fetch_current_lines(db: Session, bookmaker_region: str = "us") -> list[dict]:
    """Moneyline / run line / total for every upcoming MLB game - one
    request against the account-wide monthly budget, regardless of how
    many games are in the slate (The Odds API returns the whole slate in
    one call).

    Returns The Odds API's raw event list (one dict per game, each carrying
    a `bookmakers` list) - `ingest_current_lines` is what maps that onto our
    per-game `odds_snapshots` rows. Takes `db` (unlike a "pure" fetch
    wrapper elsewhere in ingestion/) because the budget check needs it -
    there is no way to call this function and skip the budget accounting.
    """
    if not settings.has_odds_key:
        log.debug("ODDS_API_KEY not set - skipping odds fetch")
        return []
    if not within_budget(db, API_NAME, settings.odds_api_monthly_limit, settings.odds_api_safety_buffer):
        return []

    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": settings.odds_api_key,
            "regions": bookmaker_region,
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        },
        timeout=REQUEST_TIMEOUT,
    )
    record_call(db, API_NAME)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        log.info("The Odds API requests remaining this period (per their own header): %s", remaining)
    return resp.json()


def fetch_line_history(db: Session, event_id: str, timestamp: dt.datetime) -> dict | None:
    """Odds as of a specific point in time, for closing-line-value tracking.

    Requires The Odds API's paid "historical odds" add-on
    (`/v4/historical/sports/.../odds`) - the free tier only exposes current
    lines. Returns None on any non-2xx response (e.g. 422 "not on your
    plan") so CLV tracking degrades to "insufficient data" instead of
    crashing the backtest. Budget-checked the same as fetch_current_lines -
    this endpoint counts against the same account-wide monthly cap.
    """
    if not settings.has_odds_key:
        return None
    if not within_budget(db, API_NAME, settings.odds_api_monthly_limit, settings.odds_api_safety_buffer):
        return None

    resp = requests.get(
        f"{BASE_URL}/historical/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": settings.odds_api_key,
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "american",
            "date": timestamp.astimezone(dt.timezone.utc).isoformat(),
        },
        timeout=REQUEST_TIMEOUT,
    )
    record_call(db, API_NAME)
    if resp.status_code != 200:
        log.warning("Historical odds unavailable (status %s) - is your plan the free tier?", resp.status_code)
        return None
    data = resp.json().get("data", [])
    return next((e for e in data if e.get("id") == event_id), None)


def fetch_public_bet_percentages(event_id: str) -> dict | None:
    """Bet/money split (% of tickets and % of handle on each side).

    The Odds API does not provide this - it's typically sourced from a
    dedicated consensus-odds provider (e.g. Action Network, VSiN, Sports
    Insights). Left as an explicit stub: wire in that provider's client
    here if/when you have access, and callers (features/, backtest/) that
    read the `predicted_probability` vs. `public_pct` gap already handle a
    None result by skipping the "fade the public" feature rather than
    erroring.
    """
    log.debug("fetch_public_bet_percentages: no configured provider (event %s)", event_id)
    return None


def _american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def _extract_best_lines(event: dict) -> dict:
    """Collapse a multi-bookmaker event payload down to one representative
    line per market (first bookmaker returned - good enough for a single
    reference price; CLV tracking cares about movement over time, not
    which specific book)."""
    out = {
        "moneyline_home": None,
        "moneyline_away": None,
        "run_line": None,
        "run_line_odds": None,
        "total": None,
        "over_odds": None,
        "under_odds": None,
    }
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return out

    home_team = event.get("home_team")
    away_team = event.get("away_team")
    markets = {m["key"]: m for m in bookmakers[0].get("markets", [])}

    if "h2h" in markets:
        for outcome in markets["h2h"]["outcomes"]:
            if outcome["name"] == home_team:
                out["moneyline_home"] = outcome["price"]
            elif outcome["name"] == away_team:
                out["moneyline_away"] = outcome["price"]

    if "spreads" in markets:
        for outcome in markets["spreads"]["outcomes"]:
            if outcome["name"] == home_team:
                out["run_line"] = outcome["point"]
                out["run_line_odds"] = outcome["price"]

    if "totals" in markets:
        for outcome in markets["totals"]["outcomes"]:
            out["total"] = outcome.get("point")
            if outcome["name"] == "Over":
                out["over_odds"] = outcome["price"]
            elif outcome["name"] == "Under":
                out["under_odds"] = outcome["price"]

    return out


def ingest_current_lines(db: Session) -> int:
    """Pull current lines for all upcoming games and write an `odds_snapshots`
    row per game (one row per poll, so line movement over time is queryable -
    see backtest/clv_tracker.py)."""
    events = fetch_current_lines(db)
    written = 0
    now = dt.datetime.now(dt.timezone.utc)

    for event in events:
        commence = dt.datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        game = db.execute(
            select(Game).where(
                Game.home_team.has(name=event["home_team"]),
                Game.away_team.has(name=event["away_team"]),
                Game.date == commence.date(),
            )
        ).scalar_one_or_none()
        if game is None:
            log.debug("No matching game in DB for odds event %s @ %s on %s", event["away_team"], event["home_team"], commence.date())
            continue

        lines = _extract_best_lines(event)
        db.add(OddsSnapshot(game_id=game.id, timestamp=now, source="the-odds-api", **lines))
        written += 1

    return written
