from ingestion.fangraphs import estimate_fip
from ingestion.mlb_stats_api import _parse_innings_pitched


class TestParseInningsPitched:
    def test_whole_innings(self):
        assert _parse_innings_pitched("6.0") == 6.0

    def test_one_third(self):
        assert _parse_innings_pitched("6.1") == 6.0 + 1 / 3

    def test_two_thirds(self):
        assert _parse_innings_pitched("6.2") == 6.0 + 2 / 3

    def test_none_input(self):
        assert _parse_innings_pitched(None) is None

    def test_empty_string(self):
        assert _parse_innings_pitched("") is None


class TestEstimateFip:
    def test_known_values(self):
        # FIP = ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP + constant
        fip = estimate_fip(hr=1, bb=2, k=8, ip=6.0, constant=3.15)
        expected = ((13 * 1) + (3 * 2) - (2 * 8)) / 6.0 + 3.15
        assert fip == round(expected, 2)

    def test_zero_ip_returns_none(self):
        assert estimate_fip(hr=1, bb=1, k=1, ip=0) is None

    def test_dominant_start_has_low_fip(self):
        # No baserunners allowed, lots of Ks -> well below league-average FIP
        fip = estimate_fip(hr=0, bb=0, k=10, ip=7.0, constant=3.15)
        assert fip < 3.15
