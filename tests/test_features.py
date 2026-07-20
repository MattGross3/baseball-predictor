from features.batter_features import _woba_proxy, _z_score
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
