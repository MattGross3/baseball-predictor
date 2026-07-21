import datetime as dt

import pandas as pd

from api.schemas import PredictionOut
from models.model_utils import (
    classification_metrics,
    date_split,
    feature_columns,
    regression_metrics,
    seasonal_walk_forward_splits,
    summarize_walk_forward,
    walk_forward_splits,
    walk_forward_splits_by_games,
)


def _multi_season_df(games_per_day: int = 15, seasons: tuple[int, ...] = (2023, 2024, 2025, 2026)) -> pd.DataFrame:
    """Synthetic data spanning multiple real MLB seasons with a realistic
    Nov-Mar off-season gap between each (regular season modeled as
    March 28 - September 30) - what exposes the calendar-based
    walk_forward_splits' off-season bug and confirms the game-count-based
    replacement and the seasonal splitter are both immune to it."""
    rows = []
    label = 0
    for season in seasons:
        for date in pd.date_range(f"{season}-03-28", f"{season}-09-30", freq="D"):
            for _ in range(games_per_day):
                rows.append({"date": date.date(), "label": label})  # unique per row, so leakage checks are meaningful
                label += 1
    return pd.DataFrame(rows)


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


class TestWalkForwardSplits:
    def _df(self, n_days=40):
        return pd.DataFrame({"date": pd.date_range("2025-04-01", periods=n_days), "label": range(n_days)})

    def test_returns_requested_number_of_folds_when_data_allows(self):
        df = self._df(n_days=40)
        folds = walk_forward_splits(df, n_splits=3, test_window_days=10)
        assert len(folds) == 3

    def test_folds_are_chronological_and_expanding(self):
        df = self._df(n_days=40)
        folds = walk_forward_splits(df, n_splits=3, test_window_days=10)
        # Expanding window: each successive fold's train set is a superset
        # of (strictly larger than) the previous fold's, since the cutoff
        # only ever advances forward in time.
        train_sizes = [len(train) for train, _ in folds]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] < train_sizes[-1]

        # Chronological order: fold i's test window ends before fold i+1's.
        test_max_dates = [pd.to_datetime(test["date"]).max() for _, test in folds]
        assert test_max_dates == sorted(test_max_dates)

    def test_no_row_leaks_between_train_and_test_within_a_fold(self):
        df = self._df(n_days=40)
        folds = walk_forward_splits(df, n_splits=3, test_window_days=10)
        for train_df, test_df in folds:
            assert set(train_df["label"]).isdisjoint(set(test_df["label"]))
            assert (pd.to_datetime(train_df["date"]).dt.date < pd.to_datetime(test_df["date"]).dt.date.min()).all()

    def test_last_fold_test_window_ends_at_most_recent_data(self):
        df = self._df(n_days=40)
        folds = walk_forward_splits(df, n_splits=3, test_window_days=10)
        last_test = folds[-1][1]
        assert pd.to_datetime(last_test["date"]).max().date() == df["date"].max().date()

    def test_drops_folds_that_would_have_no_training_history(self):
        # Only 15 days of data but asking for 3 folds of 10-day test
        # windows (30 days) - most folds have nowhere to draw a non-empty
        # training set from and should be silently dropped, not raise.
        df = self._df(n_days=15)
        folds = walk_forward_splits(df, n_splits=3, test_window_days=10)
        assert len(folds) < 3
        for train_df, test_df in folds:
            assert not train_df.empty
            assert not test_df.empty


class TestWalkForwardSplitsByGames:
    def test_folds_never_empty_across_the_offseason_gap(self):
        # The whole point: a calendar-day version (walk_forward_splits)
        # would silently drop folds whose window lands in the Nov-Mar gap
        # between seasons. Game-count folds are drawn from real rows, so
        # there's no such thing as an empty one here.
        df = _multi_season_df()
        folds = walk_forward_splits_by_games(df, n_splits=5, test_size=150)
        assert len(folds) == 5
        for train_df, test_df in folds:
            assert not train_df.empty
            assert len(test_df) == 150

    def test_test_block_never_spans_the_offseason_gap(self):
        # A calendar window straddling the ~5-month off-season would show
        # a huge gap between consecutive dates inside the block. Game-count
        # blocks only ever contain real games, so the largest gap between
        # consecutive dates in any block should look like normal in-season
        # scheduling (at most a few days), nowhere near the real gap.
        df = _multi_season_df()
        folds = walk_forward_splits_by_games(df, n_splits=5, test_size=150)
        for _, test_df in folds:
            dates = sorted(pd.to_datetime(test_df["date"]).dt.date.unique())
            gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
            assert max(gaps, default=0) < 30

    def test_expanding_window_and_chronological_no_leakage(self):
        df = _multi_season_df()
        folds = walk_forward_splits_by_games(df, n_splits=5, test_size=150)
        train_sizes = [len(train) for train, _ in folds]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] < train_sizes[-1]
        for train_df, test_df in folds:
            assert set(train_df["label"]).isdisjoint(set(test_df["label"]))
            assert pd.to_datetime(train_df["date"]).max() < pd.to_datetime(test_df["date"]).min()

    def test_last_fold_ends_at_the_newest_game(self):
        df = _multi_season_df()
        folds = walk_forward_splits_by_games(df, n_splits=5, test_size=150)
        last_test = folds[-1][1]
        assert pd.to_datetime(last_test["date"]).max().date() == df["date"].max()

    def test_drops_folds_without_enough_training_history(self):
        df = _multi_season_df(seasons=(2026,))  # one short season - not much history
        folds = walk_forward_splits_by_games(df, n_splits=20, test_size=150, min_train_size=200)
        assert len(folds) < 20
        for train_df, _ in folds:
            assert len(train_df) >= 200

    def test_min_train_size_defaults_to_at_least_test_size(self):
        # Only barely enough games for one fold's test block plus a
        # trivial training set - default min_train_size (max(test_size,
        # 200)) should reject it rather than fit a near-useless model.
        df = _multi_season_df(seasons=(2026,), games_per_day=1)
        n = len(df)
        folds = walk_forward_splits_by_games(df, n_splits=1, test_size=n - 5)
        assert folds == []

    def test_stable_sort_preserves_same_day_row_order(self):
        # Two dates, three rows each, with each date's rows deliberately
        # NOT in label order - a stable sort-by-date must preserve each
        # date group's pre-sort relative order rather than reordering ties
        # some other way (e.g. falling back to sorting by label too).
        df = pd.DataFrame(
            {"date": [dt.date(2025, 4, 1)] * 3 + [dt.date(2025, 4, 2)] * 3, "label": [12, 10, 11, 22, 20, 21]}
        )
        folds = walk_forward_splits_by_games(df, n_splits=1, test_size=3, min_train_size=3)
        train_df, test_df = folds[0]
        assert list(train_df["label"]) == [12, 10, 11]
        assert list(test_df["label"]) == [22, 20, 21]


class TestSeasonalWalkForwardSplits:
    def test_one_fold_per_season_after_the_first(self):
        df = _multi_season_df(seasons=(2023, 2024, 2025, 2026))
        folds = seasonal_walk_forward_splits(df)
        assert len(folds) == 3  # 2024, 2025, 2026 each tested once - 2023 never tested

    def test_folds_are_oldest_tested_season_first(self):
        df = _multi_season_df(seasons=(2023, 2024, 2025, 2026))
        folds = seasonal_walk_forward_splits(df)
        test_years = [pd.to_datetime(test_df["date"]).dt.year.iloc[0] for _, test_df in folds]
        assert test_years == [2024, 2025, 2026]

    def test_trains_only_on_strictly_earlier_seasons(self):
        df = _multi_season_df(seasons=(2023, 2024, 2025, 2026))
        folds = seasonal_walk_forward_splits(df)
        for train_df, test_df in folds:
            test_year = pd.to_datetime(test_df["date"]).dt.year.iloc[0]
            train_years = pd.to_datetime(train_df["date"]).dt.year.unique()
            assert all(y < test_year for y in train_years)

    def test_no_leakage_within_a_fold(self):
        df = _multi_season_df(seasons=(2023, 2024, 2025, 2026))
        folds = seasonal_walk_forward_splits(df)
        for train_df, test_df in folds:
            assert set(train_df["label"]).isdisjoint(set(test_df["label"]))

    def test_no_folds_with_only_one_season(self):
        df = _multi_season_df(seasons=(2026,))
        assert seasonal_walk_forward_splits(df) == []


class TestSummarizeWalkForward:
    def test_empty_folds_returns_empty_dict(self):
        assert summarize_walk_forward([]) == {}

    def test_averages_and_sums_across_folds(self):
        folds = [
            {"accuracy": 0.5, "log_loss": 0.70, "brier_score": 0.25, "n": 10},
            {"accuracy": 0.6, "log_loss": 0.68, "brier_score": 0.24, "n": 20},
        ]
        summary = summarize_walk_forward(folds)
        assert summary["n_folds"] == 2
        assert summary["n_total"] == 30  # summed, not averaged
        assert summary["accuracy_mean"] == 0.55
        assert summary["log_loss_mean"] == 0.69

    def test_identical_folds_have_zero_std(self):
        folds = [{"accuracy": 0.5, "n": 10}, {"accuracy": 0.5, "n": 10}]
        summary = summarize_walk_forward(folds)
        assert summary["accuracy_std"] == 0.0

    def test_varying_folds_have_nonzero_std(self):
        folds = [{"accuracy": 0.4, "n": 10}, {"accuracy": 0.6, "n": 10}]
        summary = summarize_walk_forward(folds)
        assert summary["accuracy_std"] > 0


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
