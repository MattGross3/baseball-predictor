"""
Streamlit dashboard v1 (Section 9) - today's slate view.

Run with:
    streamlit run frontend/app.py

Game detail / backtest / model comparison live in frontend/pages/ (Streamlit's
multi-page convention - they show up in the sidebar automatically).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `config`/`frontend.*` imports resolve when run via `streamlit run`

from frontend import api_client

st.set_page_config(page_title="Baseball Predictor - Today's Slate", page_icon="⚾", layout="wide")

st.title("⚾ Today's Slate")
st.caption("Each model's expected win probability and predicted run total for each game.")

selected_date = st.date_input("Slate date", value=dt.date.today())

try:
    games = api_client.games_today(date=selected_date.isoformat())
except Exception as exc:
    st.error(f"Couldn't reach the API at the configured API_BASE_URL: {exc}")
    st.stop()

if not games:
    st.info(f"No games found for {selected_date}. Try a date with ingested data (see README for the demo backfill range).")
    st.stop()

for game in games:
    with st.container(border=True):
        cols = st.columns([3, 2, 2, 2, 2])

        with cols[0]:
            st.markdown(f"**{game['away_team']['abbreviation']} @ {game['home_team']['abbreviation']}**")
            start = game.get("start_time")
            st.caption(f"{start[11:16] if start else 'TBD'} · {game['venue']['name'] if game['venue'] else 'TBD'} · {game['status']}")
            if game["status"] == "final":
                st.caption(f"Final: {game['away_team']['abbreviation']} {game['away_score']} - {game['home_team']['abbreviation']} {game['home_score']}")

        try:
            pred_data = api_client.get_game_predictions(game["id"])
        except Exception:
            pred_data = {"predictions": []}

        preds_by_target = {p["target_type"]: p for p in pred_data["predictions"]}
        moneyline = preds_by_target.get("moneyline")
        total = preds_by_target.get("total")

        with cols[1]:
            if moneyline and moneyline["predicted_probability"] is not None:
                home_prob = moneyline["predicted_probability"]
                st.metric(f"{game['home_team']['abbreviation']} expected win %", f"{home_prob:.1%}")
            else:
                st.metric("Expected win %", "—")
                st.caption("No prediction yet")

        with cols[2]:
            if moneyline and moneyline["predicted_probability"] is not None:
                away_prob = 1 - moneyline["predicted_probability"]
                st.metric(f"{game['away_team']['abbreviation']} expected win %", f"{away_prob:.1%}")
            else:
                st.metric("Expected win %", "—")

        with cols[3]:
            if total and total["predicted_value"] is not None:
                st.metric("Predicted total", f"{total['predicted_value']:.1f}")
            else:
                st.metric("Predicted total", "—")

        with cols[4]:
            if st.button("View detail →", key=f"detail_{game['id']}"):
                st.session_state["selected_game_id"] = game["id"]
                st.switch_page("pages/1_Game_Detail.py")

st.divider()
st.caption("Click \"View detail\" on any game, or enter a game id directly on the Game Detail page (sidebar).")
