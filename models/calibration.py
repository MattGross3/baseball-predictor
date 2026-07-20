"""
Calibration diagnostics (Section 7.5): reliability diagrams and Brier score
tracking over time. Pure computation here - the Streamlit dashboard's
Model Comparison view (Section 9) is what plots these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def reliability_diagram(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Bins predictions into `n_bins` equal-width buckets and compares mean
    predicted probability to actual observed frequency in each bin - the
    standard reliability-diagram table. A well-calibrated model has
    `mean_predicted` ~= `observed_frequency` in every bin, which is exactly
    what matters when comparing a predicted win probability to the market's
    implied probability (Section 7.1's whole point of calibrating at all).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            rows.append(
                {"bin_low": bins[b], "bin_high": bins[b + 1], "mean_predicted": None, "observed_frequency": None, "n": 0}
            )
            continue
        rows.append(
            {
                "bin_low": round(bins[b], 2),
                "bin_high": round(bins[b + 1], 2),
                "mean_predicted": round(float(y_prob[mask].mean()), 4),
                "observed_frequency": round(float(y_true[mask].mean()), 4),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def brier_score_over_time(df: pd.DataFrame, date_col: str = "date", y_true_col: str = "y_true", y_prob_col: str = "y_prob", freq: str = "W") -> pd.DataFrame:
    """Rolling Brier score aggregated by calendar period (default weekly),
    so calibration drift over a season is visible at a glance rather than
    buried in one aggregate number."""
    working = df.copy()
    working[date_col] = pd.to_datetime(working[date_col])
    working = working.set_index(date_col)

    rows = []
    for period, group in working.resample(freq):
        if group.empty:
            continue
        score = brier_score_loss(group[y_true_col], group[y_prob_col])
        rows.append({"period": period, "brier_score": round(score, 4), "n": len(group)})
    return pd.DataFrame(rows)
