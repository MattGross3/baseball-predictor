"""FastAPI app entrypoint (Section 8). Run with:

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import datetime as dt
import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import backtest, games, models, odds, predictions
from api.schemas import HealthConfigOut
from config import settings
from ingestion.umpire_scorecards import _season_league_pitches

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Baseball Prediction API",
    description="MLB moneyline / totals / NRFI / player-prop predictions, features, odds and backtest results.",
    version="0.1.0",
)


def _warm_umpire_season_cache() -> None:
    """`_season_league_pitches` is an in-process lru_cache - fast for every
    request after the first, but the first caller pays a real ~1-2 minute
    league-wide Statcast pull (day-by-day, even against pybaseball's own
    disk cache - deserializing/concatenating a full season of pitch-level
    data isn't free). Left lazy, that first caller is whichever user hits
    /backtest/results first after a server restart - discovered when the
    dashboard's Backtest page timed out on a fresh restart. Pre-warming in
    a background thread at startup means the cost is paid once, during
    boot, off the request path, instead of by whoever's unlucky enough to
    load the page first.
    """
    try:
        _season_league_pitches(dt.date.today().year)
    except Exception:
        log.exception("Background warm-up of umpire season Statcast cache failed - will retry lazily on first request")


@app.on_event("startup")
def _on_startup() -> None:
    threading.Thread(target=_warm_umpire_season_cache, daemon=True).start()

# Wide open by default since this is a single-tenant personal-project API
# behind no auth for its read endpoints; tighten before exposing publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (games.router, predictions.router, backtest.router, models.router, odds.router):
    app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/config", response_model=HealthConfigOut)
def health_config():
    """Which optional API keys are configured, booleans only - never the
    key values themselves. The frontend uses this to tell "no odds key
    configured, every game is blank by design" apart from "key configured,
    this specific game just has no odds snapshot yet" - two very
    different states that used to render as the same unexplained '—'."""
    return HealthConfigOut(
        odds_api_key_configured=settings.has_odds_key,
        weather_api_key_configured=settings.has_weather_key,
    )
