from features.batter_features import _woba_proxy, _z_score
from features.build_feature_matrix import flatten_feature_row
from features.park_weather_features import _signed_wind_out


class TestWobaProxy:
    def test_no_plate_appearances_returns_none(self):
        assert _woba_proxy(bb=0, h=0, hr=0, ab=0) is None

    def test_more_hr_scores_higher_than_singles_only(self):
        # Same H total, but one profile is all home runs
        low = _woba_proxy(bb=0, h=4, hr=0, ab=10)
        high = _woba_proxy(bb=0, h=4, hr=4, ab=10)
        assert high > low

    def test_walks_contribute_positively(self):
        no_walks = _woba_proxy(bb=0, h=2, hr=0, ab=10)
        with_walks = _woba_proxy(bb=3, h=2, hr=0, ab=10)
        assert with_walks > no_walks


class TestZScore:
    def test_above_season_average_is_positive(self):
        z = _z_score(recent_woba=0.400, season_woba=0.320, assumed_league_std=0.08)
        assert z == 1.0

    def test_below_season_average_is_negative(self):
        z = _z_score(recent_woba=0.240, season_woba=0.320, assumed_league_std=0.08)
        assert z == -1.0

    def test_zero_std_returns_none(self):
        assert _z_score(0.3, 0.3, assumed_league_std=0) is None


class TestSignedWindOut:
    def test_missing_data_returns_none(self):
        assert _signed_wind_out(None, "S") is None
        assert _signed_wind_out(10, None) is None

    def test_blowing_out_is_positive(self):
        assert _signed_wind_out(10, "S") == 10.0

    def test_blowing_in_is_negative(self):
        assert _signed_wind_out(10, "N") == -10.0

    def test_crosswind_is_zero(self):
        assert _signed_wind_out(10, "E") == 0.0


class TestFlattenFeatureRow:
    def test_market_implied_probability_is_added_when_odds_exist(self):
        nested = {
            "game_id": 1,
            "date": "2026-07-19",
            "home_starter": {"era_season": 3.4, "fip_season": 3.2, "era_last_3_starts": None, "k_pct_rolling": None, "bb_pct_rolling": None, "velo_trend_last_3": None, "days_rest": 4, "pitch_count_last_start": 92, "vs_opponent_career_era": None, "home_away_split_era": {"home": None, "away": None}},
            "away_starter": {"era_season": 3.8, "fip_season": 3.5, "era_last_3_starts": None, "k_pct_rolling": None, "bb_pct_rolling": None, "velo_trend_last_3": None, "days_rest": 5, "pitch_count_last_start": 88, "vs_opponent_career_era": None, "home_away_split_era": {"home": None, "away": None}},
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

        flat = flatten_feature_row(nested)

        assert flat["market_implied_probability_home"] == 0.56
        assert flat["market_implied_probability_away"] == 0.44
