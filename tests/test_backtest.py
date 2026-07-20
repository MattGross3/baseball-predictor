import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backtest.clv_tracker import american_to_implied_prob, compute_clv, devig_two_way
from database.models import Base, Game, OddsSnapshot, Prediction, Team


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


class TestDevigTwoWay:
    def test_sums_to_one(self):
        # -150/+130 is a real, vig-laden two-way market: raw implied
        # probabilities sum to ~103.5%, not 100%.
        raw_home = american_to_implied_prob(-150)
        raw_away = american_to_implied_prob(130)
        assert round(raw_home + raw_away, 4) != 1.0  # sanity check there's vig to remove

        fair_home, fair_away = devig_two_way(raw_home, raw_away)
        assert round(fair_home + fair_away, 8) == 1.0

    def test_devigged_probability_is_lower_than_raw_for_each_side(self):
        # Removing the vig can only ever shrink each side's share (dividing
        # by a total > 1), never grow it.
        raw_home = american_to_implied_prob(-150)
        raw_away = american_to_implied_prob(130)
        fair_home, fair_away = devig_two_way(raw_home, raw_away)
        assert fair_home < raw_home
        assert fair_away < raw_away

    def test_no_vig_is_a_no_op(self):
        # A hypothetical zero-vig market (50/50 at even money both sides)
        # should devig to exactly itself.
        raw_home = american_to_implied_prob(100)
        raw_away = american_to_implied_prob(100)
        fair_home, fair_away = devig_two_way(raw_home, raw_away)
        assert round(fair_home, 6) == 0.5
        assert round(fair_away, 6) == 0.5


@pytest.fixture
def db_session():
    # In-memory SQLite, not the real Postgres DB - compute_clv only needs
    # plain SELECT/ORDER BY over OddsSnapshot, which every model here
    # supports portably, so this is a real, fast, isolated regression test
    # rather than a mock of the query layer.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make_game(db: Session, game_id: int = 1) -> Game:
    db.add_all([
        Team(id=1, mlb_team_id=100, name="Home Team", abbreviation="HOM", league="AL", division="AL East"),
        Team(id=2, mlb_team_id=200, name="Away Team", abbreviation="AWY", league="AL", division="AL East"),
    ])
    game = Game(
        id=game_id, mlb_game_id=game_id, date=dt.date(2026, 6, 1),
        home_team_id=1, away_team_id=2, status="final", home_score=4, away_score=2,
    )
    db.add(game)
    db.flush()
    return game


class TestComputeClv:
    """Regression coverage for the compute_clv() NameError: it called
    `_american_to_implied_prob` (leading underscore) when only
    `american_to_implied_prob` is defined in this module - untested until
    now since tests/test_backtest.py previously only exercised the
    underscore-less helper directly, never compute_clv() itself. Any of
    these tests would have raised NameError on the buggy version."""

    def test_none_with_fewer_than_two_snapshots(self, db_session):
        game = _make_game(db_session)
        db_session.add(OddsSnapshot(game_id=game.id, timestamp=dt.datetime(2026, 6, 1, 10), moneyline_home=-150, moneyline_away=130))
        db_session.flush()

        assert compute_clv(db_session, game.id, bet_odds=-150, bet_side="home") is None

    def test_positive_clv_when_line_moves_toward_your_side(self, db_session):
        game = _make_game(db_session)
        # Bet home at -150; closing line moved to -180 (home even more
        # favored) - you got the better number, so this should be positive CLV.
        db_session.add_all([
            OddsSnapshot(game_id=game.id, timestamp=dt.datetime(2026, 6, 1, 10), moneyline_home=-150, moneyline_away=130),
            OddsSnapshot(game_id=game.id, timestamp=dt.datetime(2026, 6, 1, 18), moneyline_home=-180, moneyline_away=160),
        ])
        db_session.flush()

        result = compute_clv(db_session, game.id, bet_odds=-150, bet_side="home")
        assert result is not None
        assert result["clv_pct"] > 0

    def test_negative_clv_when_line_moves_away_from_your_side(self, db_session):
        game = _make_game(db_session)
        # Bet home at -150; closing line drifted to -120 (home less
        # favored) - you got the worse number, so this should be negative CLV.
        db_session.add_all([
            OddsSnapshot(game_id=game.id, timestamp=dt.datetime(2026, 6, 1, 10), moneyline_home=-150, moneyline_away=130),
            OddsSnapshot(game_id=game.id, timestamp=dt.datetime(2026, 6, 1, 18), moneyline_home=-120, moneyline_away=100),
        ])
        db_session.flush()

        result = compute_clv(db_session, game.id, bet_odds=-150, bet_side="home")
        assert result is not None
        assert result["clv_pct"] < 0


class TestComputeEdgeVsMarket:
    """Regression coverage for the missing-devig bug: api/routers/games.py's
    _compute_edge_vs_market() used to subtract the model's probability from
    one side's *raw* (vig-inflated) implied probability instead of the
    devigged fair probability, which systematically understates edge.
    Fixed example: -150/+130 (raw implied probs sum to ~103.5%) with a
    0.62 model probability on the home side.
    """

    def test_edge_uses_fair_not_raw_implied_probability(self, db_session):
        from api.routers.games import _compute_edge_vs_market

        game = _make_game(db_session)
        db_session.add(OddsSnapshot(game_id=game.id, timestamp=dt.datetime(2026, 6, 1, 10), moneyline_home=-150, moneyline_away=130))
        db_session.add(Prediction(game_id=game.id, model_name="moneyline_xgboost", model_version="v1", target_type="moneyline", predicted_probability=0.62))
        db_session.flush()

        predictions = db_session.query(Prediction).filter_by(game_id=game.id).all()
        result = _compute_edge_vs_market(db_session, game.id, predictions)

        assert result is not None
        raw_home_implied = american_to_implied_prob(-150)
        # The old, buggy behavior: edge = 0.62 - 0.6 = 0.02 (raw implied).
        old_raw_edge = round(0.62 - raw_home_implied, 4)
        assert result["market_implied_probability_home"] != round(raw_home_implied, 4)
        assert result["market_implied_probability_home"] < round(raw_home_implied, 4)
        # Devigged edge should be strictly more favorable (larger) than the
        # old raw-implied-probability edge would have been.
        assert result["edge"] > old_raw_edge
