"""
Park + weather features (Section 6).
"""
from __future__ import annotations

import datetime as dt

from database.models import Venue
from ingestion.weather import fetch_game_weather


def compute_park_weather_features(venue: Venue, game_datetime: dt.datetime) -> dict:
    """Note the parameter is the `Venue` row itself, not an id + a DB
    session - the caller (build_game_feature_row) already has it loaded
    via the game's relationship, and every field this needs (lat/lon/park
    factors/roof type) lives on that one row."""
    weather = {"temp_f": None, "wind_speed_mph": None, "wind_direction": None, "humidity": None}
    roof_closed = venue.roof_type is not None and venue.roof_type.lower() not in ("open", "retractable")

    if venue.roof_type is None or venue.roof_type.lower() != "dome":
        if venue.lat is not None and venue.lon is not None:
            reading = fetch_game_weather(venue.lat, venue.lon, game_datetime)
            weather = {k: reading[k] for k in weather}

    wind_out_mph = _signed_wind_out(weather["wind_speed_mph"], weather["wind_direction"])

    return {
        "park_factor_runs": venue.park_factor_runs,
        "park_factor_hr": venue.park_factor_hr,
        "temp_f": weather["temp_f"],
        "wind_out_mph": wind_out_mph,
        "roof_closed": roof_closed,
    }


def _signed_wind_out(speed_mph: float | None, direction: str | None) -> float | None:
    """Positive = blowing out (toward the outfield), per the spec. Without
    each park's actual orientation (azimuth) we can't know "out" precisely
    per-stadium, so this uses a simple heuristic: southerly components
    (S/SW/SSW/SSE/SE) are treated as blowing out for a typical home-plate
    orientation, everything else as blowing in/across. This is a rough
    proxy - wiring in real per-park azimuths (Savant's venue fieldInfo has
    an `azimuthAngle`) would make it exact."""
    if speed_mph is None or direction is None:
        return None
    out_directions = {"S", "SSW", "SSE", "SW", "SE"}
    in_directions = {"N", "NNW", "NNE", "NW", "NE"}
    if direction in out_directions:
        return round(speed_mph, 1)
    if direction in in_directions:
        return round(-speed_mph, 1)
    return 0.0
