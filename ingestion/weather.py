"""
Weather ingestion (Section 4.5), via OpenWeatherMap, keyed to stadium lat/lon
(already captured on the `venues` table by ingestion/mlb_stats_api.py).

Requires WEATHER_API_KEY in .env. Without one, `fetch_game_weather` returns
an all-None reading (not an exception) and callers fall back to
park-factor-only features - see features/park_weather_features.py.

Free-tier caveat: OpenWeatherMap's no-cost plan only offers current
conditions + a 5-day/3-hour forecast, not historical data. That means live
"today's slate" games get real forecasted weather, but backfilling weather
for older games (for backtesting) isn't possible on the free tier - those
rows simply stay null. A paid One Call "timemachine" subscription would
close that gap without any code changes here.
"""
from __future__ import annotations

import datetime as dt
import logging

import requests

from config import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT = 15

_EMPTY_READING = {
    "temp_f": None,
    "wind_speed_mph": None,
    "wind_direction": None,
    "humidity": None,
    "condition": None,
}


def fetch_game_weather(lat: float, lon: float, game_datetime: dt.datetime) -> dict:
    """Best-available weather reading for a venue near a game's start time.

    Uses the closest 3-hour forecast bucket to `game_datetime` if it falls
    within the next 5 days, otherwise the current-conditions endpoint as a
    rough same-day fallback. Returns an all-None dict if no API key is
    configured or the request fails, rather than raising - weather is a
    nice-to-have feature, not a pipeline-blocking dependency.
    """
    if not settings.has_weather_key:
        log.debug("WEATHER_API_KEY not set - skipping weather fetch")
        return dict(_EMPTY_READING)

    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=5)
    game_dt_utc = game_datetime.astimezone(dt.timezone.utc) if game_datetime.tzinfo else game_datetime.replace(tzinfo=dt.timezone.utc)

    try:
        if now <= game_dt_utc <= horizon:
            return _fetch_forecast(lat, lon, game_dt_utc)
        return _fetch_current(lat, lon)
    except requests.RequestException as exc:
        log.warning("Weather fetch failed for (%s, %s): %s", lat, lon, exc)
        return dict(_EMPTY_READING)


def _fetch_current(lat: float, lon: float) -> dict:
    resp = requests.get(
        f"{BASE_URL}/weather",
        params={"lat": lat, "lon": lon, "appid": settings.weather_api_key, "units": "imperial"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return _parse_reading(resp.json())


def _fetch_forecast(lat: float, lon: float, target: dt.datetime) -> dict:
    resp = requests.get(
        f"{BASE_URL}/forecast",
        params={"lat": lat, "lon": lon, "appid": settings.weather_api_key, "units": "imperial"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    buckets = resp.json().get("list", [])
    if not buckets:
        return dict(_EMPTY_READING)

    def _delta(bucket: dict) -> float:
        bucket_dt = dt.datetime.fromtimestamp(bucket["dt"], tz=dt.timezone.utc)
        return abs((bucket_dt - target).total_seconds())

    closest = min(buckets, key=_delta)
    return _parse_reading(closest)


def _parse_reading(payload: dict) -> dict:
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    weather_list = payload.get("weather", [])
    return {
        "temp_f": main.get("temp"),
        "wind_speed_mph": wind.get("speed"),
        "wind_direction": _degrees_to_compass(wind.get("deg")),
        "humidity": main.get("humidity"),
        "condition": weather_list[0]["main"] if weather_list else None,
    }


def _degrees_to_compass(deg: float | None) -> str | None:
    if deg is None:
        return None
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return directions[idx]
