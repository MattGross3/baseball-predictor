"""Game detail view (Section 9): expected win probability, predictions,
and feature breakdown - why the model likes this side - for one game."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from frontend import api_client

st.set_page_config(page_title="Game Detail", page_icon="🔎", layout="wide")
st.title("🔎 Game Detail")

default_id = st.session_state.get("selected_game_id", 1)
game_id = st.number_input("Game id", min_value=1, value=default_id, step=1)

try:
    game = api_client.get_game(int(game_id))
except Exception as exc:
    st.error(f"Couldn't load game {game_id}: {exc}")
    st.stop()

st.subheader(f"{game['away_team']['name']} @ {game['home_team']['name']}")
st.caption(f"{game['date']} · {game['venue']['name'] if game['venue'] else 'TBD'} · {game['status']}")
if game["status"] == "final":
    st.caption(f"Final score: {game['away_team']['abbreviation']} {game['away_score']} - {game['home_team']['abbreviation']} {game['home_score']}")

try:
    pred_data = api_client.get_game_predictions(int(game_id))
except Exception as exc:
    st.error(f"Couldn't load predictions: {exc}")
    pred_data = {"predictions": []}

preds_by_target = {p["target_type"]: p for p in pred_data["predictions"]}
moneyline = preds_by_target.get("moneyline")
total = preds_by_target.get("total")
nrfi = preds_by_target.get("nrfi")

st.subheader("Expected outcome")
c1, c2, c3 = st.columns(3)
if moneyline and moneyline["predicted_probability"] is not None:
    home_prob = moneyline["predicted_probability"]
    c1.metric(f"{game['home_team']['abbreviation']} expected win %", f"{home_prob:.1%}")
    c2.metric(f"{game['away_team']['abbreviation']} expected win %", f"{1 - home_prob:.1%}")
else:
    c1.metric("Expected win %", "—")
    c2.metric("Expected win %", "—")
c3.metric("Predicted total runs", f"{total['predicted_value']:.1f}" if total and total["predicted_value"] is not None else "—")

if nrfi and nrfi["predicted_probability"] is not None:
    st.caption(f"NRFI (no runs first inning) probability: {nrfi['predicted_probability']:.1%}")

tab_predictions, tab_features = st.tabs(["Predictions", "Feature breakdown"])

with tab_predictions:
    if not pred_data["predictions"]:
        st.info("No predictions generated yet for this game. Run models/predict.py or POST /models/retrain first.")
    else:
        df = pd.DataFrame(pred_data["predictions"])[["target_type", "model_version", "predicted_probability", "predicted_value", "created_at"]]
        st.dataframe(df, width='stretch', hide_index=True)

with tab_features:
    try:
        feat_data = api_client.get_game_features(int(game_id))
        features = feat_data["features"]
    except Exception as exc:
        st.error(f"Couldn't load features: {exc}")
        features = None

    if features:
        col_home, col_away = st.columns(2)
        for col, side, label in ((col_home, "home", game["home_team"]["abbreviation"]), (col_away, "away", game["away_team"]["abbreviation"])):
            with col:
                st.markdown(f"**{label} (starter)**")
                st.json(features[f"{side}_starter"], expanded=False)
                st.markdown(f"**{label} (bullpen)**")
                st.json(features[f"{side}_bullpen"], expanded=False)
                st.markdown(f"**{label} (team form)**")
                st.json(features[f"{side}_team"], expanded=False)
                st.markdown(f"**{label} (lineup)**")
                st.json(features[f"{side}_lineup"], expanded=False)

        park_weather = features.get("park_weather", {})
        st.markdown("**Park factors**")
        st.json(
            {
                "park_factor_runs": park_weather.get("park_factor_runs"),
                "park_factor_hr": park_weather.get("park_factor_hr"),
                "roof_closed": park_weather.get("roof_closed"),
            },
            expanded=False,
        )
        st.markdown("**Umpire**")
        st.json(features.get("umpire", {}), expanded=False)
