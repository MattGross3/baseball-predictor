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

from backtest.clv_tracker import american_to_implied_prob
from database.models import Game, OddsSnapshot, Umpire
from features.batter_features import compute_leadoff_obp, compute_lineup_features
from features.bullpen_features import compute_bullpen_features
from features.injury_features import compute_injury_features
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
_LINEUP_SCALAR_FIELDS = ["lineup_wOBA_weighted_by_order", "platoon_advantage_count", "leadoff_obp"]
_PARK_WEATHER_SCALAR_FIELDS = ["park_factor_runs", "park_factor_hr", "temp_f", "wind_out_mph"]
_UMPIRE_SCALAR_FIELDS = ["strike_zone_size_percentile", "over_under_lean", "k_rate_boost"]
_INJURY_SCALAR_FIELDS = ["injured_count", "key_regulars_injured"]


def build_game_feature_row(
    db: Session, game_id: int, include_statcast_trend: bool = True, include_live_oaa: bool | None = None
) -> dict:
    """Full nested feature dict for one game (internal `games.id`, not the
    MLB gamePk).

    `include_statcast_trend=False` skips the per-starter velocity-trend
    lookup (pitcher_features.compute_starter_features) and the umpire
    zone-history lookup (umpire_features.compute_umpire_features) - this
    used to gate raw per-call Statcast network cost, but both are now
    backed by an in-process per-season cache (see
    pitcher_features._season_pitcher_pitches and
    umpire_scorecards._season_league_pitches), so `build_training_matrix`
    leaves this on by default too.

    `include_live_oaa` is a *separate* knob, defaulting to whatever
    `include_statcast_trend` is unless given explicitly. Don't just tie it
    to `include_statcast_trend`, though: unlike the two lookups above,
    team OAA isn't a performance trade-off, it's a leakage one - Savant's
    leaderboard can't be bounded to as_of_date (team_features._oaa_rating),
    so bulk historical builds must keep this off regardless of how fast
    the other two get. build_training_matrix passes it explicitly False.
    """
    if include_live_oaa is None:
        include_live_oaa = include_statcast_trend
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

    # Bounded to snapshots at-or-before first pitch even though
    # ingestion/odds_api.py only ever polls upcoming (not-yet-started)
    # games in practice - the leakage boundary belongs here, in the
    # feature layer, not as an assumption about an external API's
    # undocumented behavior. Matches the "as of" discipline every other
    # feature in this file follows.
    odds_query = select(OddsSnapshot).where(OddsSnapshot.game_id == game.id)
    if game.start_time is not None:
        odds_query = odds_query.where(OddsSnapshot.timestamp <= game.start_time)
    latest_odds = db.execute(odds_query.order_by(OddsSnapshot.timestamp.desc())).scalar_one_or_none()

    market_implied_probability_home = None
    market_implied_probability_away = None
    if latest_odds is not None:
        if latest_odds.moneyline_home is not None:
            market_implied_probability_home = round(american_to_implied_prob(latest_odds.moneyline_home), 4)
        if latest_odds.moneyline_away is not None:
            market_implied_probability_away = round(american_to_implied_prob(latest_odds.moneyline_away), 4)

    return {
        "game_id": game.id,
        "mlb_game_id": game.mlb_game_id,
        "date": game.date,
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        # Ingested and exposed via the API since Section 1, but never
        # actually fed to any model until now - doubleheaders strain a
        # bullpen (both games draw from the same relievers) and shuffle
        # rosters (a fresh 26th man for game two) in ways the rest of the
        # feature set doesn't otherwise capture.
        "is_doubleheader": game.is_doubleheader,
        "game_number_in_series": game.game_number_in_series,
        "home_starter": home_starter,
        "away_starter": away_starter,
        "home_bullpen": compute_bullpen_features(db, game.home_team_id, game.date),
        "away_bullpen": compute_bullpen_features(db, game.away_team_id, game.date),
        "home_team": compute_team_features(db, game.home_team_id, game.date, include_live_oaa=include_live_oaa),
        "away_team": compute_team_features(db, game.away_team_id, game.date, include_live_oaa=include_live_oaa),
        "home_lineup": {
            **compute_lineup_features(db, game.home_team_id, game.id, game.date, opposing_pitcher_hand=away_hand),
            **compute_leadoff_obp(db, game.home_team_id, game.id, game.date),
        },
        "away_lineup": {
            **compute_lineup_features(db, game.away_team_id, game.id, game.date, opposing_pitcher_hand=home_hand),
            **compute_leadoff_obp(db, game.away_team_id, game.id, game.date),
        },
        # Unlike the Statcast-trend/umpire lookups this flag otherwise
        # gates, injury features are reconstructed from dated transaction
        # history (see injury_features.py), so they're cheap and safe to
        # compute during bulk training too - no gating needed.
        "home_injuries": compute_injury_features(db, game.home_team_id, game.date),
        "away_injuries": compute_injury_features(db, game.away_team_id, game.date),
        "park_weather": compute_park_weather_features(game.venue, game.start_time) if game.venue else {},
        "umpire": compute_umpire_features(db, umpire_row, game.date, include_zone_history=include_statcast_trend) if umpire_row else {
            "strike_zone_size_percentile": None, "over_under_lean": None, "k_rate_boost": None
        },
        "market_implied_probability_home": market_implied_probability_home,
        "market_implied_probability_away": market_implied_probability_away,
    }


def flatten_feature_row(nested: dict) -> dict:
    flat: dict = {"game_id": nested["game_id"], "date": nested["date"]}
    flat["is_doubleheader"] = nested.get("is_doubleheader")
    flat["game_number_in_series"] = nested.get("game_number_in_series")

    def _pull(prefix: str, source: dict, fields: list[str]):
        for f in fields:
            flat[f"{prefix}_{f}"] = source.get(f)

    _pull("home_starter", nested["home_starter"], _STARTER_SCALAR_FIELDS)
    _pull("away_starter", nested["away_starter"], _STARTER_SCALAR_FIELDS)
    _pull("home_bullpen", nested["home_bullpen"], _BULLPEN_SCALAR_FIELDS)
    _pull("away_bullpen", nested["away_bullpen"], _BULLPEN_SCALAR_FIELDS)
    # .get() with a default (unlike the direct nested[...] access above) -
    # this key was added after those, and the tests construct nested dicts
    # by hand without it; a bare KeyError here would be a surprising way
    # for an otherwise-unrelated test to break.
    _pull("home_injuries", nested.get("home_injuries", {}), _INJURY_SCALAR_FIELDS)
    _pull("away_injuries", nested.get("away_injuries", {}), _INJURY_SCALAR_FIELDS)
    _pull("home_team", nested["home_team"], _TEAM_SCALAR_FIELDS)
    _pull("away_team", nested["away_team"], _TEAM_SCALAR_FIELDS)
    _pull("home_lineup", nested["home_lineup"], _LINEUP_SCALAR_FIELDS)
    _pull("away_lineup", nested["away_lineup"], _LINEUP_SCALAR_FIELDS)
    _pull("park", nested.get("park_weather", {}), _PARK_WEATHER_SCALAR_FIELDS)
    _pull("umpire", nested.get("umpire", {}), _UMPIRE_SCALAR_FIELDS)

    flat["home_starter_home_away_split_era_home"] = nested["home_starter"].get("home_away_split_era", {}).get("home")
    flat["home_starter_home_away_split_era_away"] = nested["home_starter"].get("home_away_split_era", {}).get("away")
    flat["away_starter_home_away_split_era_home"] = nested["away_starter"].get("home_away_split_era", {}).get("home")
    flat["away_starter_home_away_split_era_away"] = nested["away_starter"].get("home_away_split_era", {}).get("away")

    flat["home_team_home_away_win_pct_home"] = nested["home_team"].get("home_away_win_pct", {}).get("home")
    flat["home_team_home_away_win_pct_away"] = nested["home_team"].get("home_away_win_pct", {}).get("away")
    flat["away_team_home_away_win_pct_home"] = nested["away_team"].get("home_away_win_pct", {}).get("home")
    flat["away_team_home_away_win_pct_away"] = nested["away_team"].get("home_away_win_pct", {}).get("away")

    bullpen_h_home = nested["home_bullpen"].get("bullpen_hand_distribution", {})
    bullpen_h_away = nested["away_bullpen"].get("bullpen_hand_distribution", {})
    flat["home_bullpen_hand_distribution_L"] = bullpen_h_home.get("L")
    flat["home_bullpen_hand_distribution_R"] = bullpen_h_home.get("R")
    flat["away_bullpen_hand_distribution_L"] = bullpen_h_away.get("L")
    flat["away_bullpen_hand_distribution_R"] = bullpen_h_away.get("R")

    flat["home_lineup_hot_streak_count"] = len(nested["home_lineup"].get("hot_streak_players", []))
    flat["away_lineup_hot_streak_count"] = len(nested["away_lineup"].get("hot_streak_players", []))

    flat["market_implied_probability_home"] = nested.get("market_implied_probability_home")
    flat["market_implied_probability_away"] = nested.get("market_implied_probability_away")

    flat["home_closer_available"] = nested["home_bullpen"].get("closer_available")
    flat["away_closer_available"] = nested["away_bullpen"].get("closer_available")
    flat["home_lineup_confirmed"] = nested["home_lineup"].get("lineup_confirmed")
    flat["away_lineup_confirmed"] = nested["away_lineup"].get("lineup_confirmed")
    flat["roof_closed"] = nested.get("park_weather", {}).get("roof_closed")

    # Differential features: (minuend - subtrahend), for every scalar pair
    # we have both sides of. For every row except the injuries one, that's
    # literally home minus away. The injuries row deliberately flips which
    # side is which so the *sign* stays consistent with every other row
    # (positive = favors home) rather than the *key order* - the away
    # team having more injured regulars is good news for the home team,
    # so away's count is the minuend here.
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
        ("leadoff_obp", "home_lineup_leadoff_obp", "away_lineup_leadoff_obp"),
        ("market_implied_probability", "market_implied_probability_home", "market_implied_probability_away"),
        ("key_regulars_injured", "away_injuries_key_regulars_injured", "home_injuries_key_regulars_injured"),
    ]
    for name, minuend_key, subtrahend_key in diff_pairs:
        m, s = flat.get(minuend_key), flat.get(subtrahend_key)
        flat[f"diff_{name}"] = (m - s) if (m is not None and s is not None) else None

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
            # include_statcast_trend=True: both the per-starter velo-trend
            # and umpire zone-history lookups are now backed by per-season
            # caches (see build_game_feature_row's docstring), so bulk
            # training no longer needs to drop them. include_live_oaa stays
            # explicitly False regardless - that one's a leakage guard, not
            # a performance trade-off, and must never turn on for
            # historical rows.
            nested = build_game_feature_row(db, game.id, include_statcast_trend=True, include_live_oaa=False)
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
