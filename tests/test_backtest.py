import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import backtest.backtest_engine as backtest_engine
import models.train_totals as train_totals
from backtest.backtest_engine import _total_over_probability, high_confidence_accuracy
from backtest.clv_tracker import american_to_implied_prob, compute_clv, devig_two_way
from database.models import Base, Game, ModelRegistryEntry, OddsSnapshot, Prediction, Team


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


class TestTotalOverProbability:
    def test_sums_only_totals_strictly_above_the_line(self):
        dist = {8: 0.25, 9: 0.75}
        assert _total_over_probability(dist, 8.5) == 0.75

    def test_totals_at_or_below_the_line_are_excluded(self):
        dist = {8: 0.5, 9: 0.5}
        assert _total_over_probability(dist, 9) == 0  # 9 is not > 9


def _fake_registry_entry(db: Session, model_name: str, target_type: str) -> None:
    db.add(ModelRegistryEntry(model_name=model_name, target_type=target_type, version="v1", file_path="unused.pkl"))
    db.flush()


class TestHighConfidenceAccuracyClassification:
    """Moneyline/NRFI path: confidence is just the model's own predicted
    probability - >=threshold or <=1-threshold counts, everything in
    between is excluded as "not confident", not scored as wrong."""

    def _make_four_games(self, db: Session) -> None:
        db.add_all([
            Team(id=1, mlb_team_id=100, name="Home Team", abbreviation="HOM", league="AL", division="AL East"),
            Team(id=2, mlb_team_id=200, name="Away Team", abbreviation="AWY", league="AL", division="AL East"),
        ])
        for i in range(1, 5):
            db.add(Game(id=i, mlb_game_id=i, date=dt.date(2026, 6, i), home_team_id=1, away_team_id=2, status="final", home_score=4, away_score=2))
        db.flush()

    def test_excludes_coin_flip_games_and_scores_only_confident_ones(self, db_session, monkeypatch):
        self._make_four_games(db_session)
        _fake_registry_entry(db_session, "moneyline_xgboost", "moneyline")

        # game 1: p=0.90 (confident home), label=1 (home won)   -> correct
        # game 2: p=0.15 (confident away), label=0 (away won)   -> correct
        # game 3: p=0.50 (coin flip)                             -> excluded
        # game 4: p=0.65 (confident home), label=0 (away won)   -> wrong
        df = pd.DataFrame({
            "game_id": [1, 2, 3, 4],
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3), dt.date(2026, 6, 4)],
            "label": [1, 0, 1, 0],
        })
        probs = [0.9, 0.15, 0.5, 0.65]

        class FakeModel:
            def predict_proba(self, X):
                return np.array([[1 - p, p] for p in probs])

        monkeypatch.setattr(backtest_engine, "build_training_matrix", lambda db, s, e, target: df)
        monkeypatch.setattr(backtest_engine, "load_model", lambda path: {"model": FakeModel(), "feature_columns": ["f1"]})

        result = high_confidence_accuracy(db_session, "moneyline_xgboost", dt.date(2026, 6, 1), dt.date(2026, 6, 5))

        assert result["n_considered"] == 4
        assert result["n"] == 3  # games 1, 2, 4 - game 3 excluded
        assert round(result["accuracy"], 4) == round(2 / 3, 4)

    def test_no_confident_games_returns_null_metrics_not_zero(self, db_session, monkeypatch):
        self._make_four_games(db_session)
        _fake_registry_entry(db_session, "nrfi_logistic", "nrfi")

        df = pd.DataFrame({
            "game_id": [1, 2],
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
            "label": [1, 0],
        })

        class FakeModel:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5], [0.45, 0.55]])  # neither clears 60%/40%

        monkeypatch.setattr(backtest_engine, "build_training_matrix", lambda db, s, e, target: df)
        monkeypatch.setattr(backtest_engine, "load_model", lambda path: {"model": FakeModel(), "feature_columns": ["f1"]})

        result = high_confidence_accuracy(db_session, "nrfi_logistic", dt.date(2026, 6, 1), dt.date(2026, 6, 3))

        assert result["n_considered"] == 2
        assert result["n"] == 0
        assert result["accuracy"] is None


class TestHighConfidenceAccuracyTotals:
    """Totals path: no native probability (it's a regression target), so
    confidence comes from the model's own implied P(over)/P(under) versus
    the market total line - which requires a real odds_snapshot per game,
    not just a prediction."""

    def _make_games_with_odds(self, db: Session, totals: dict[int, float]) -> None:
        db.add_all([
            Team(id=1, mlb_team_id=100, name="Home Team", abbreviation="HOM", league="AL", division="AL East"),
            Team(id=2, mlb_team_id=200, name="Away Team", abbreviation="AWY", league="AL", division="AL East"),
        ])
        for game_id, total_line in totals.items():
            db.add(Game(id=game_id, mlb_game_id=game_id, date=dt.date(2026, 6, game_id), home_team_id=1, away_team_id=2, status="final", home_score=5, away_score=4))
            db.add(OddsSnapshot(game_id=game_id, timestamp=dt.datetime(2026, 6, game_id, 10), total=total_line))
        db.flush()

    def test_only_odds_covered_games_with_high_confidence_are_scored(self, db_session, monkeypatch):
        # game A (f1=1): model says 75% over 8.5, actual total 9 (over)  -> correct, confident
        # game B (f1=2): model says 95% under 8.5, actual total 9 (over) -> wrong, confident
        # game C (f1=3): model says 50/50 on 8.5                        -> excluded, not confident
        self._make_games_with_odds(db_session, {1: 8.5, 2: 8.5, 3: 8.5})
        _fake_registry_entry(db_session, "totals_poisson", "total")

        df = pd.DataFrame({
            "game_id": [1, 2, 3],
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3)],
            "label": [9, 9, 8],
            "f1": [1, 2, 3],
        })
        distributions = {
            1: {8: 0.25, 9: 0.75},
            2: {8: 0.95, 9: 0.05},
            3: {8: 0.5, 9: 0.5},
        }

        def fake_poisson_run_distribution(poisson_models, feature_row):
            f1 = int(feature_row["f1"].iloc[0])
            return {"distribution_over_totals": distributions[f1]}

        monkeypatch.setattr(backtest_engine, "build_training_matrix", lambda db, s, e, target: df)
        monkeypatch.setattr(
            backtest_engine, "load_model",
            lambda path: {"model": {"home": "fake", "away": "fake"}, "feature_columns": ["f1"]},
        )
        monkeypatch.setattr(train_totals, "poisson_run_distribution", fake_poisson_run_distribution)

        result = high_confidence_accuracy(db_session, "totals_poisson", dt.date(2026, 6, 1), dt.date(2026, 6, 4))

        assert result["n_considered"] == 3  # all 3 games had a market total line
        assert result["n"] == 2  # only A and B cleared 60% confidence
        assert round(result["accuracy"], 4) == 0.5  # A correct, B wrong

    def test_games_without_a_market_line_are_excluded_from_n_considered(self, db_session, monkeypatch):
        self._make_games_with_odds(db_session, {1: 8.5})  # only game 1 has odds
        _fake_registry_entry(db_session, "totals_poisson", "total")

        df = pd.DataFrame({
            "game_id": [1, 2],
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
            "label": [9, 9],
            "f1": [1, 1],
        })

        monkeypatch.setattr(backtest_engine, "build_training_matrix", lambda db, s, e, target: df)
        monkeypatch.setattr(
            backtest_engine, "load_model",
            lambda path: {"model": {"home": "fake", "away": "fake"}, "feature_columns": ["f1"]},
        )
        monkeypatch.setattr(train_totals, "poisson_run_distribution", lambda pm, fr: {"distribution_over_totals": {8: 0.25, 9: 0.75}})

        result = high_confidence_accuracy(db_session, "totals_poisson", dt.date(2026, 6, 1), dt.date(2026, 6, 3))

        assert result["n_considered"] == 1  # game 2 has no odds snapshot, excluded entirely
