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
    """`include_zone_history=False` skips `compute_umpire_zone_history`.
    That used to mean a fresh full-league `statcast()` date-range pull per
    (umpire, game) - far too slow to repeat across a bulk training-matrix
    build - but it now draws from a per-season cache
    (`_season_league_pitches`), so this stays as an opt-out escape hatch
    rather than something `build_training_matrix` needs to lean on. See
    build_game_feature_row.
    """
    if not include_zone_history:
        return {"strike_zone_size_percentile": None, "over_under_lean": None, "k_rate_boost": None}
    history = compute_umpire_zone_history(db, umpire_name, as_of_date)
    return {
        "strike_zone_size_percentile": history["strike_zone_size_percentile"],
        "over_under_lean": history["over_under_lean"],
        "k_rate_boost": history["k_rate_boost"],
    }
