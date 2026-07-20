"""
Combines every feature module into one row per game (Section 6).

`build_game_feature_row` returns a nested dict - handy for the API's
"why does the model like this side" explain view. `build_training_matrix`
flattens that into a numeric pd.DataFrame suitable for scikit-learn/XGBoost,
computing home-minus-away differential features in addition to the raw
per-side values (per the spec: "combines all of the above into one row per
game with differential features (home - away) in addition to raw values").

Leakage discipline: every sub-feature call is passed `game.date` as
`as_of_date`, so nothing here ever reads same-day-or-later data for the
*inputs*. Labels (win/loss, total runs, NRFI) come from the game's own
final score/linescore, which is exactly what a label should be.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, Umpire
from features.batter_features import compute_lineup_features
from features.bullpen_features import compute_bullpen_features
from features.park_weather_features import compute_park_weather_features
from features.pitcher_features import compute_starter_features
from features.team_features import compute_team_features
from features.umpire_features import compute_umpire_features

log = logging.getLogger(__name__)

_EMPTY_STARTER = {
    "era_season": None, "fip_season": None, "siera_season": None,
    "era_last_3_starts": None, "k_pct_rolling": None, "bb_pct_rolling": None,
    "velo_trend_last_3": None, "days_rest": None, "pitch_count_last_start": None,
    "home_away_split_era": {"home": None, "away": None},
    "vs_opponent_career_era": None, "handedness": None,
}

# Scalar fields pulled straight into the model matrix from each feature
# dict. List/dict-valued fields (pitch_mix, hot_streak_players, home/away
# splits) are kept in the nested `build_game_feature_row` output for the
# API's explain view but intentionally left out of the flat training
# matrix - they're not directly usable by sklearn/XGBoost without further
# encoding, which is future work, not this pass.
_STARTER_SCALAR_FIELDS = [
    "era_season", "fip_season", "era_last_3_starts", "k_pct_rolling", "bb_pct_rolling",
    "velo_trend_last_3", "days_rest", "pitch_count_last_start", "vs_opponent_career_era",
]
_BULLPEN_SCALAR_FIELDS = ["bullpen_era_rolling_7d", "bullpen_era_rolling_14d", "innings_thrown_last_3_games"]
_TEAM_SCALAR_FIELDS = ["win_pct_season", "win_pct_last_10", "run_diff_season", "pythag_win_pct", "oaa_defense_rating"]
_LINEUP_SCALAR_FIELDS = ["lineup_wOBA_weighted_by_order", "platoon_advantage_count"]
_PARK_WEATHER_SCALAR_FIELDS = ["park_factor_runs", "park_factor_hr", "temp_f", "wind_out_mph"]
_UMPIRE_SCALAR_FIELDS = ["strike_zone_size_percentile", "over_under_lean", "k_rate_boost"]


def build_game_feature_row(db: Session, game_id: int, include_statcast_trend: bool = True) -> dict:
    """Full nested feature dict for one game (internal `games.id`, not the
    MLB gamePk). `include_statcast_trend=False` skips every feature that
    needs a live Statcast network pull done per-game: the per-starter
    velocity-trend lookup (pitcher_features.compute_starter_features) and
    the umpire zone-history lookup (umpire_features.compute_umpire_features,
    which pulls a full-league statcast() date range - the single most
    expensive call in the whole feature layer). Cheap for one live
    prediction; far too slow to repeat hundreds of times, which is exactly
    what build_training_matrix does, so it always passes False here.
    """
    game = db.get(Game, game_id)
    if game is None:
        raise ValueError(f"No game with id={game_id}")

    home_starter = (
        compute_starter_features(db, game.home_starter_id, game.date, opponent_team_id=game.away_team_id, include_statcast_trend=include_statcast_trend)
        if game.home_starter_id else dict(_EMPTY_STARTER)
    )
    away_starter = (
        compute_starter_features(db, game.away_starter_id, game.date, opponent_team_id=game.home_team_id, include_statcast_trend=include_statcast_trend)
        if game.away_starter_id else dict(_EMPTY_STARTER)
    )

    home_hand = home_starter.get("handedness")
    away_hand = away_starter.get("handedness")

    umpire_row = db.execute(
        select(Umpire.umpire_name).where(Umpire.game_id == game.id, Umpire.home_plate.is_(True))
    ).scalar_one_or_none()

    return {
        "game_id": game.id,
        "mlb_game_id": game.mlb_game_id,
        "date": game.date,
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "home_starter": home_starter,
        "away_starter": away_starter,
        "home_bullpen": compute_bullpen_features(db, game.home_team_id, game.date),
        "away_bullpen": compute_bullpen_features(db, game.away_team_id, game.date),
        "home_team": compute_team_features(db, game.home_team_id, game.date),
        "away_team": compute_team_features(db, game.away_team_id, game.date),
        "home_lineup": compute_lineup_features(db, game.home_team_id, game.id, game.date, opposing_pitcher_hand=away_hand),
        "away_lineup": compute_lineup_features(db, game.away_team_id, game.id, game.date, opposing_pitcher_hand=home_hand),
        "park_weather": compute_park_weather_features(game.venue, game.start_time) if game.venue else {},
        "umpire": compute_umpire_features(db, umpire_row, game.date, include_zone_history=include_statcast_trend) if umpire_row else {
            "strike_zone_size_percentile": None, "over_under_lean": None, "k_rate_boost": None
        },
    }


def flatten_feature_row(nested: dict) -> dict:
    flat: dict = {"game_id": nested["game_id"], "date": nested["date"]}

    def _pull(prefix: str, source: dict, fields: list[str]):
        for f in fields:
            flat[f"{prefix}_{f}"] = source.get(f)

    _pull("home_starter", nested["home_starter"], _STARTER_SCALAR_FIELDS)
    _pull("away_starter", nested["away_starter"], _STARTER_SCALAR_FIELDS)
    _pull("home_bullpen", nested["home_bullpen"], _BULLPEN_SCALAR_FIELDS)
    _pull("away_bullpen", nested["away_bullpen"], _BULLPEN_SCALAR_FIELDS)
    _pull("home_team", nested["home_team"], _TEAM_SCALAR_FIELDS)
    _pull("away_team", nested["away_team"], _TEAM_SCALAR_FIELDS)
    _pull("home_lineup", nested["home_lineup"], _LINEUP_SCALAR_FIELDS)
    _pull("away_lineup", nested["away_lineup"], _LINEUP_SCALAR_FIELDS)
    _pull("park", nested.get("park_weather", {}), _PARK_WEATHER_SCALAR_FIELDS)
    _pull("umpire", nested.get("umpire", {}), _UMPIRE_SCALAR_FIELDS)

    flat["home_closer_available"] = nested["home_bullpen"].get("closer_available")
    flat["away_closer_available"] = nested["away_bullpen"].get("closer_available")
    flat["home_lineup_confirmed"] = nested["home_lineup"].get("lineup_confirmed")
    flat["away_lineup_confirmed"] = nested["away_lineup"].get("lineup_confirmed")
    flat["roof_closed"] = nested.get("park_weather", {}).get("roof_closed")

    # Differential features: home minus away, for every scalar pair we have both sides of.
    diff_pairs = [
        ("era_season", "home_starter_era_season", "away_starter_era_season"),
        ("fip_season", "home_starter_fip_season", "away_starter_fip_season"),
        ("k_pct_rolling", "home_starter_k_pct_rolling", "away_starter_k_pct_rolling"),
        ("bb_pct_rolling", "home_starter_bb_pct_rolling", "away_starter_bb_pct_rolling"),
        ("days_rest", "home_starter_days_rest", "away_starter_days_rest"),
        ("bullpen_era_7d", "home_bullpen_bullpen_era_rolling_7d", "away_bullpen_bullpen_era_rolling_7d"),
        ("win_pct_season", "home_team_win_pct_season", "away_team_win_pct_season"),
        ("win_pct_last_10", "home_team_win_pct_last_10", "away_team_win_pct_last_10"),
        ("pythag_win_pct", "home_team_pythag_win_pct", "away_team_pythag_win_pct"),
        ("oaa_defense_rating", "home_team_oaa_defense_rating", "away_team_oaa_defense_rating"),
        ("lineup_wOBA", "home_lineup_lineup_wOBA_weighted_by_order", "away_lineup_lineup_wOBA_weighted_by_order"),
    ]
    for name, home_key, away_key in diff_pairs:
        h, a = flat.get(home_key), flat.get(away_key)
        flat[f"diff_{name}"] = (h - a) if (h is not None and a is not None) else None

    return flat


def build_training_matrix(db: Session, start_date: dt.date, end_date: dt.date, target: str) -> pd.DataFrame:
    """One row per completed game in [start_date, end_date) with features
    + a label column for `target` in {"moneyline", "total", "nrfi"}.

    Always call this with disjoint train/test date ranges - never shuffle
    rows randomly across dates before splitting (Section 11's explicit
    warning against leaking future team form into past predictions).
    """
    if target not in {"moneyline", "total", "nrfi"}:
        raise ValueError(f"Unknown target '{target}' - expected moneyline, total, or nrfi")

    games = db.execute(
        select(Game)
        .where(Game.date >= start_date, Game.date < end_date)
        .where(Game.status == "final")
        .order_by(Game.date)
    ).scalars().all()

    rows = []
    for i, game in enumerate(games):
        if game.home_score is None or game.away_score is None:
            continue
        if target == "nrfi" and (game.first_inning_home_runs is None or game.first_inning_away_runs is None):
            continue

        try:
            # include_statcast_trend=False: see build_game_feature_row's
            # docstring - bulk historical builds skip the per-starter live
            # Statcast trend lookups to stay tractable across many games.
            nested = build_game_feature_row(db, game.id, include_statcast_trend=False)
        except Exception as exc:
            log.warning("Skipping game %s while building training matrix: %s", game.id, exc)
            continue

        if (i + 1) % 50 == 0:
            log.info("build_training_matrix: processed %d/%d games", i + 1, len(games))

        flat = flatten_feature_row(nested)
        # Always carried alongside `label` so callers that need a different
        # view of the same game (e.g. train_totals.py's per-team Poisson
        # baseline needs home/away runs separately, not just the summed
        # total) don't have to rebuild the feature row a second time.
        flat["home_score"] = game.home_score
        flat["away_score"] = game.away_score
        flat["first_inning_home_runs"] = game.first_inning_home_runs
        flat["first_inning_away_runs"] = game.first_inning_away_runs

        if target == "moneyline":
            flat["label"] = int(game.home_score > game.away_score)
        elif target == "total":
            flat["label"] = game.home_score + game.away_score
        else:
            flat["label"] = int(game.first_inning_home_runs == 0 and game.first_inning_away_runs == 0)

        rows.append(flat)

    return pd.DataFrame(rows)
