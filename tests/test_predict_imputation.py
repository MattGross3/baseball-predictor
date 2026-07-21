"""
Regression coverage for the train/serve imputation mismatch: every
training script median-imputes missing feature values, but the live-serving
path (models/predict.py's _score_with_model, and the totals distribution
functions it calls into) used to fill with 0 instead - a nonsense value for
stats like ERA, which reads to the model as "elite" rather than "unknown".
"""
from __future__ import annotations

import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

from database.models import ModelRegistryEntry
from features.build_feature_matrix import flatten_feature_row
from models.model_utils import prepare_xy
from models.predict import _score_with_model

# A known, deliberately non-zero median so a 0-fill and a median-fill are
# guaranteed to disagree - away_starter_era_season values chosen so their
# median is exactly 4.5.
_ERA_VALUES = [3.0, 4.0, 5.0, 6.0]  # median = 4.5


def _nested_row(era_season: float | None) -> dict:
    return {
        "game_id": 1,
        "date": "2026-07-19",
        "home_starter": {"era_season": 3.4, "fip_season": 3.2, "era_last_3_starts": None, "k_pct_rolling": None, "bb_pct_rolling": None, "velo_trend_last_3": None, "days_rest": 4, "pitch_count_last_start": 92, "vs_opponent_career_era": None, "home_away_split_era": {"home": None, "away": None}},
        "away_starter": {"era_season": era_season, "fip_season": 3.5, "era_last_3_starts": None, "k_pct_rolling": None, "bb_pct_rolling": None, "velo_trend_last_3": None, "days_rest": 5, "pitch_count_last_start": 88, "vs_opponent_career_era": None, "home_away_split_era": {"home": None, "away": None}},
        "home_bullpen": {"bullpen_era_rolling_7d": 3.1, "bullpen_era_rolling_14d": 3.4, "innings_thrown_last_3_games": 10, "closer_available": True, "bullpen_hand_distribution": {"L": 40.0, "R": 60.0}},
        "away_bullpen": {"bullpen_era_rolling_7d": 3.3, "bullpen_era_rolling_14d": 3.6, "innings_thrown_last_3_games": 11, "closer_available": False, "bullpen_hand_distribution": {"L": 50.0, "R": 50.0}},
        "home_team": {"win_pct_season": 0.570, "win_pct_last_10": 0.600, "run_diff_season": 13, "pythag_win_pct": 0.560, "oaa_defense_rating": 4.2},
        "away_team": {"win_pct_season": 0.520, "win_pct_last_10": 0.500, "run_diff_season": -5, "pythag_win_pct": 0.510, "oaa_defense_rating": 1.8},
        "home_lineup": {"lineup_wOBA_weighted_by_order": 0.340, "platoon_advantage_count": 2, "hot_streak_players": [], "lineup_confirmed": True},
        "away_lineup": {"lineup_wOBA_weighted_by_order": 0.310, "platoon_advantage_count": 1, "hot_streak_players": [], "lineup_confirmed": False},
        "park_weather": {"park_factor_runs": 1.05, "park_factor_hr": 1.12, "temp_f": 78.0, "wind_out_mph": 8.0, "roof_closed": False},
        "umpire": {"strike_zone_size_percentile": 55.0, "over_under_lean": 1.0, "k_rate_boost": 0.02},
        "market_implied_probability_home": 0.56,
        "market_implied_probability_away": 0.44,
    }


def _train_bundle(tmp_path, with_medians: bool) -> ModelRegistryEntry:
    """Trains a real LogisticRegression on synthetic data (4 known
    away_starter_era_season values, median 4.5) via the same prepare_xy
    path train_moneyline.py uses, then pickles a bundle exactly like
    model_utils.save_model would - optionally omitting feature_medians to
    simulate a legacy pre-fix model artifact."""
    rows = []
    for i, era in enumerate(_ERA_VALUES):
        flat = flatten_feature_row(_nested_row(era))
        flat["label"] = i % 2
        rows.append(flat)
    df = pd.DataFrame(rows)

    X_train, y_train, medians = prepare_xy(df)
    assert medians["away_starter_era_season"] == 4.5  # sanity check on the known median

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    bundle_path = tmp_path / f"test_model_{'with' if with_medians else 'without'}_medians.pkl"
    bundle = {"model": model, "feature_columns": list(X_train.columns)}
    if with_medians:
        bundle["feature_medians"] = medians
    with bundle_path.open("wb") as f:
        pickle.dump(bundle, f)

    return ModelRegistryEntry(model_name="test_moneyline", target_type="moneyline", version="v1", file_path=str(bundle_path))


class TestScoreWithModelImputation:
    def test_missing_feature_matches_explicit_training_median(self, tmp_path):
        """The core promise of the fix, in the task's own words: a
        live-style prediction call with a missing feature must produce
        the same output as one where that feature is explicitly set to
        the training median - not 0. era_season=4.5 is this synthetic
        model's real training-set median (asserted in _train_bundle), and
        home_starter_era_season is held constant across every synthetic
        row, so diff_era_season's own median works out to exactly
        3.4 - 4.5 = -1.1 too - both the raw and the derived feature agree
        with a plain median substitution, not just era_season alone."""
        entry = _train_bundle(tmp_path, with_medians=True)

        missing = _nested_row(era_season=None)
        explicit_median = _nested_row(era_season=4.5)

        _, prob_missing, _, _ = _score_with_model(entry, "moneyline", missing)
        _, prob_median, _, _ = _score_with_model(entry, "moneyline", explicit_median)

        assert prob_missing == prob_median

    def test_median_fill_differs_from_legacy_zero_fill(self, tmp_path):
        """Same trained model (LogisticRegression is deterministic, and
        both bundles are trained on identical data), same missing-feature
        input - only whether a feature_medians dict was persisted differs
        (simulating a legacy pre-fix bundle vs. one saved after the fix).
        The two must disagree, or the fix isn't actually changing
        anything: 0 is nowhere near the training median (4.5) for
        era_season, so filling with the wrong value has to move the
        model's output."""
        with_medians_entry = _train_bundle(tmp_path, with_medians=True)
        legacy_entry = _train_bundle(tmp_path, with_medians=False)

        missing = _nested_row(era_season=None)

        _, prob_correct, _, _ = _score_with_model(with_medians_entry, "moneyline", missing)
        _, prob_legacy, _, _ = _score_with_model(legacy_entry, "moneyline", missing)

        assert prob_correct != prob_legacy
