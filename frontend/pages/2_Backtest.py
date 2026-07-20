"""Backtest view (Section 9): accuracy/ROI/CLV over a date range, and a
trend of accuracy/Brier score broken into weekly buckets so calibration
drift across a season is visible rather than buried in one aggregate."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend import api_client

st.set_page_config(page_title="Backtest", page_icon="📊", layout="wide")
st.title("📊 Backtest Results")

KNOWN_MODELS = [
    "moneyline_xgboost", "moneyline_logistic",
    "totals_xgboost", "totals_poisson",
    "nrfi_logistic", "nrfi_xgboost",
]

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    model = st.selectbox("Model", KNOWN_MODELS)
with col2:
    start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=28))
with col3:
    end_date = st.date_input("End date", value=dt.date.today())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

date_range = f"{start_date.isoformat()},{end_date.isoformat()}"

try:
    overall = api_client.backtest_results(model, date_range)
except Exception as exc:
    st.error(f"Backtest failed - is '{model}' trained yet? ({exc})")
    st.stop()

st.subheader("Overall")
metric_cols = st.columns(6)
metric_cols[0].metric("Accuracy", f"{overall['accuracy']:.1%}" if overall.get("accuracy") is not None else "—")
metric_cols[1].metric("Log loss", f"{overall['log_loss']:.4f}" if overall.get("log_loss") is not None else "—")
metric_cols[2].metric("Brier score", f"{overall['brier_score']:.4f}" if overall.get("brier_score") is not None else "—")
metric_cols[3].metric("ROI (flat bet)", f"{overall['roi_flat_bet']:.1%}" if overall.get("roi_flat_bet") is not None else "N/A")
metric_cols[4].metric("ROI (Kelly)", f"{overall['roi_kelly']:.1%}" if overall.get("roi_kelly") is not None else "N/A")
metric_cols[5].metric("Avg CLV", f"{overall['clv_avg']:+.2f}%" if overall.get("clv_avg") is not None else "N/A")
st.caption(f"{overall['n_bets']} bets simulated over {overall['date_range']}. ROI/CLV need odds_snapshots (ODDS_API_KEY) to be non-null.")

st.subheader("Weekly trend")
weeks = pd.date_range(start_date, end_date, freq="7D")
rows = []
for i in range(len(weeks) - 1):
    w_start, w_end = weeks[i].date(), weeks[i + 1].date()
    try:
        r = api_client.backtest_results(model, f"{w_start.isoformat()},{w_end.isoformat()}")
        if r.get("n_bets", 0) or r.get("accuracy") is not None:
            rows.append({"week_start": w_start, "accuracy": r.get("accuracy"), "brier_score": r.get("brier_score"), "log_loss": r.get("log_loss")})
    except Exception:
        continue

if not rows:
    st.info("Not enough weekly data to plot a trend over this range.")
else:
    trend_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend_df["week_start"], y=trend_df["accuracy"], mode="lines+markers", name="Accuracy", line=dict(color="#3B82F6", width=2)))
    fig.update_layout(title="Accuracy by week", xaxis_title="Week", yaxis_title="Accuracy", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=trend_df["week_start"], y=trend_df["brier_score"], mode="lines+markers", name="Brier score", line=dict(color="#F97316", width=2)))
    fig2.update_layout(title="Brier score by week (lower = better calibrated)", xaxis_title="Week", yaxis_title="Brier score")
    st.plotly_chart(fig2, width='stretch')

    st.dataframe(trend_df, width='stretch', hide_index=True)
