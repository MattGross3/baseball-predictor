"""
Umpire features (Section 6) - thin wrapper around
ingestion/umpire_scorecards.py's zone-history computation, reshaped to the
spec's field names.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from ingestion.umpire_scorecards import compute_umpire_zone_history


def compute_umpire_features(db: Session, umpire_name: str, as_of_date: dt.date, include_zone_history: bool = True) -> dict:
    """`include_zone_history=False` skips `compute_umpire_zone_history`,
    which pulls Statcast pitch-level data for the umpire's entire game
    history (a full-league `statcast()` date-range query, not a per-pitcher
    one) - cheap for one live prediction, far too slow to repeat for every
    game in a bulk training-matrix build. See build_game_feature_row.
    """
    if not include_zone_history:
        return {"strike_zone_size_percentile": None, "over_under_lean": None, "k_rate_boost": None}
    history = compute_umpire_zone_history(db, umpire_name, as_of_date)
    return {
        "strike_zone_size_percentile": history["strike_zone_size_percentile"],
        "over_under_lean": history["over_under_lean"],
        "k_rate_boost": history["k_rate_boost"],
    }
