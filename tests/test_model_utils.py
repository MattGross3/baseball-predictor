import datetime as dt

import pandas as pd

from api.schemas import PredictionOut
from models.model_utils import classification_metrics, date_split, feature_columns, regression_metrics


class TestDateSplit:
    def test_splits_by_date_not_row_count(self):
        df = pd.DataFrame(
            {
                "date": [dt.date(2025, 4, 1), dt.date(2025, 4, 2), dt.date(2025, 4, 10), dt.date(2025, 4, 11)],
                "label": [1, 0, 1, 0],
            }
        )
        train, test = date_split(df, test_start=dt.date(2025, 4, 10))
        assert len(train) == 2
        assert len(test) == 2
        assert train["date"].max() < dt.date(2025, 4, 10)
        assert test["date"].min() >= dt.date(2025, 4, 10)

    def test_no_row_leaks_across_the_boundary(self):
        df = pd.DataFrame({"date": pd.date_range("2025-04-01", periods=10), "label": range(10)})
        train, test = date_split(df, test_start=dt.date(2025, 4, 6))
        # Every label value appears on exactly one side of the split - no
        # row counted as both train and test.
        assert set(train["label"]).isdisjoint(set(test["label"]))
        assert len(train) + len(test) == len(df)
        assert (pd.to_datetime(train["date"]).dt.date < dt.date(2025, 4, 6)).all()


class TestFeatureColumns:
    def test_excludes_identifiers_and_label(self):
        df = pd.DataFrame(
            {
                "game_id": [1], "date": [dt.date(2025, 4, 1)], "label": [1],
                "home_score": [4], "away_score": [2],
                "first_inning_home_runs": [0], "first_inning_away_runs": [0],
                "diff_win_pct_season": [0.1],
            }
        )
        cols = feature_columns(df)
        assert cols == ["diff_win_pct_season"]


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [1, 0, 1, 0]
        y_prob = [0.99, 0.01, 0.99, 0.01]
        m = classification_metrics(y_true, y_prob)
        assert m["accuracy"] == 1.0
        assert m["n"] == 4

    def test_worst_case_predictions_score_lower_than_best_case(self):
        y_true = [1, 0, 1, 0]
        best = classification_metrics(y_true, [0.99, 0.01, 0.99, 0.01])
        worst = classification_metrics(y_true, [0.01, 0.99, 0.01, 0.99])
        assert worst["log_loss"] > best["log_loss"]
        assert worst["accuracy"] < best["accuracy"]


class TestRegressionMetrics:
    def test_zero_error_for_exact_predictions(self):
        m = regression_metrics([4, 5, 6], [4, 5, 6])
        assert m["mae"] == 0
        assert m["rmse"] == 0

    def test_mae_matches_hand_calculation(self):
        m = regression_metrics([4, 6], [5, 5])
        assert m["mae"] == 1.0


class TestPredictionSchema:
    def test_exposes_richer_prediction_fields(self):
        payload = PredictionOut.model_validate(
            {
                "id": 1,
                "game_id": 2,
                "model_name": "moneyline_xgboost",
                "model_version": "moneyline_xgboost_v1",
                "target_type": "moneyline",
                "predicted_value": 4.5,
                "predicted_probability": 0.62,
                "predicted_side": "home",
                "home_probability": 0.62,
                "away_probability": 0.38,
                "market_home_probability": 0.58,
                "market_away_probability": 0.42,
                "confidence": 0.12,
                "actual_outcome": "home_win",
                "target_unit": "win_probability",
                "created_at": "2026-07-20T12:00:00+00:00",
            }
        )

        data = payload.model_dump()
        assert data["predicted_side"] == "home"
        assert data["home_probability"] == 0.62
        assert data["away_probability"] == 0.38
        assert data["market_home_probability"] == 0.58
        assert data["market_away_probability"] == 0.42
        assert data["confidence"] == 0.12
        assert data["actual_outcome"] == "home_win"
        assert data["target_unit"] == "win_probability"
