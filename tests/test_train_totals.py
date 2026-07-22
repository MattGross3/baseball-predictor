"""
Regression coverage for the totals-model collinearity fix: every diff_X
column build_feature_matrix.py adds is an EXACT linear combination of the
two raw home_X/away_X columns it's derived from (both are also present as
standalone features), which is a classic dummy-variable-trap
collinearity. sklearn's L2-regularized logistic model and tree-based
XGBoost tolerate this fine, but statsmodels' unregularized Poisson GLM
does not - confirmed in practice on a real seasonal fold, where one
diff_/home_/away_ trio's fitted coefficients exceeded |150| and produced a
predicted game total in the billions on live data.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from models.train_totals import _prep, train_poisson_baseline


def _synthetic_totals_df(n: int = 40) -> pd.DataFrame:
    """A small synthetic training frame with one exactly-collinear trio
    (home_win_pct, away_win_pct, diff_win_pct = home - away) - the same
    structural pattern build_feature_matrix.py produces for all 14 of its
    diff_ pairs."""
    rows = []
    for i in range(n):
        home_win_pct = 0.3 + 0.01 * (i % 20)
        away_win_pct = 0.4 + 0.01 * ((i + 5) % 20)
        rows.append(
            {
                "date": dt.date(2025, 4, 1) + dt.timedelta(days=i),
                "label": 8 + (i % 5),
                "home_score": 4 + (i % 3),
                "away_score": 3 + (i % 4),
                "home_win_pct_season": home_win_pct,
                "away_win_pct_season": away_win_pct,
                "diff_win_pct_season": home_win_pct - away_win_pct,
            }
        )
    return pd.DataFrame(rows)


class TestPrepExcludesDiffColumns:
    def test_no_diff_prefixed_column_reaches_the_model(self):
        df = _synthetic_totals_df()
        X, medians = _prep(df)
        assert not any(c.startswith("diff_") for c in X.columns)
        assert not any(k.startswith("diff_") for k in medians)

    def test_raw_home_away_columns_are_kept(self):
        # Dropping the redundant diff_ column shouldn't lose real
        # information - the raw home/away pair it was derived from must
        # still reach the model.
        df = _synthetic_totals_df()
        X, _ = _prep(df)
        assert "home_win_pct_season" in X.columns
        assert "away_win_pct_season" in X.columns


class TestTrainPoissonBaselineAvoidsCollinearity:
    def test_fitted_model_has_no_diff_column_in_its_params(self):
        """The real regression check: even when the training data contains
        an exactly-collinear diff_/home_/away_ trio, the fitted Poisson
        GLM's own parameter index must never include a diff_-prefixed
        name - proving the collinearity can never reach the unregularized
        model in the first place, rather than hoping the solver happens to
        land on a tame coefficient split."""
        df = _synthetic_totals_df()
        poisson_models = train_poisson_baseline(df)
        assert not any(c.startswith("diff_") for c in poisson_models["columns"])
        assert not any(str(p).startswith("diff_") for p in poisson_models["home"].params.index)
