from backtest.clv_tracker import american_to_implied_prob


class TestAmericanToImpliedProb:
    def test_even_money(self):
        assert american_to_implied_prob(100) == 0.5
        assert american_to_implied_prob(-100) == 0.5

    def test_underdog_has_lower_implied_prob_than_favorite(self):
        underdog = american_to_implied_prob(150)
        favorite = american_to_implied_prob(-150)
        assert underdog < 0.5 < favorite

    def test_known_value(self):
        # -200 favorite: implied prob = 200 / (200 + 100) = 0.6667
        assert round(american_to_implied_prob(-200), 4) == 0.6667
