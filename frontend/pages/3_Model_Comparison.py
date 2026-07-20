"""Model comparison view (Section 9): logistic baseline vs. XGBoost vs. a
simple blend (mean of the two probabilities), side by side."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend import api_client

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Comparison")

TARGET_MODEL_PAIRS = {
    "moneyline": ("moneyline_logistic", "moneyline_xgboost"),
    "nrfi": ("nrfi_logistic", "nrfi_xgboost"),
}

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    target = st.selectbox("Target", list(TARGET_MODEL_PAIRS.keys()))
with col2:
    start_date = st.date_input("Start date", value=dt.date.today() - dt.timedelta(days=28))
with col3:
    end_date = st.date_input("End date", value=dt.date.today())

baseline_name, xgb_name = TARGET_MODEL_PAIRS[target]
date_range = f"{start_date.isoformat()},{end_date.isoformat()}"

st.subheader("Backtest metrics side by side")
cols = st.columns(2)
results = {}
for col, name in zip(cols, (baseline_name, xgb_name)):
    with col:
        st.markdown(f"**{name}**")
        try:
            r = api_client.backtest_results(name, date_range)
            results[name] = r
            st.metric("Accuracy", f"{r['accuracy']:.1%}" if r.get("accuracy") is not None else "—")
            st.metric("Log loss", f"{r['log_loss']:.4f}" if r.get("log_loss") is not None else "—")
            st.metric("Brier score", f"{r['brier_score']:.4f}" if r.get("brier_score") is not None else "—")
        except Exception as exc:
            st.warning(f"Not trained yet or no data: {exc}")

st.divider()
st.subheader("Blended (mean of both models' probabilities)")
st.caption("Pulls each model's per-game prediction history and the game's actual result, then scores a simple 50/50 average of the two probabilities against outcomes.")

try:
    baseline_preds = api_client.prediction_history(date_range, target_type=target)
    baseline_preds = [p for p in baseline_preds if p["model_version"].startswith(baseline_name)]
    xgb_preds = api_client.prediction_history(date_range, target_type=target)
    xgb_preds = [p for p in xgb_preds if p["model_version"].startswith(xgb_name)]
except Exception as exc:
    st.error(f"Couldn't load prediction history: {exc}")
    baseline_preds, xgb_preds = [], []

by_game_baseline = {p["game_id"]: p["predicted_probability"] for p in baseline_preds if p["predicted_probability"] is not None}
by_game_xgb = {p["game_id"]: p["predicted_probability"] for p in xgb_preds if p["predicted_probability"] is not None}
common_games = sorted(set(by_game_baseline) & set(by_game_xgb))

if not common_games:
    st.info(
        "No overlapping predictions from both models yet - run POST /models/retrain (or models/predict.py) "
        "for both models over this range to populate prediction history, then revisit this page."
    )
else:
    rows = []
    for game_id in common_games:
        try:
            game = api_client.get_game(game_id)
        except Exception:
            continue
        if game["home_score"] is None or game["away_score"] is None:
            continue
        label = int(game["home_score"] > game["away_score"])
        blended = (by_game_baseline[game_id] + by_game_xgb[game_id]) / 2
        rows.append({"game_id": game_id, "baseline_prob": by_game_baseline[game_id], "xgb_prob": by_game_xgb[game_id], "blended_prob": blended, "label": label})

    if rows:
        df = pd.DataFrame(rows)
        blend_metrics = {
            "accuracy": accuracy_score(df["label"], (df["blended_prob"] >= 0.5).astype(int)),
            "log_loss": log_loss(df["label"], df["blended_prob"], labels=[0, 1]),
            "brier_score": brier_score_loss(df["label"], df["blended_prob"]),
        }
        c1, c2, c3 = st.columns(3)
        c1.metric("Blended accuracy", f"{blend_metrics['accuracy']:.1%}")
        c2.metric("Blended log loss", f"{blend_metrics['log_loss']:.4f}")
        c3.metric("Blended Brier score", f"{blend_metrics['brier_score']:.4f}")
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("No completed games among the overlapping predictions yet.")
