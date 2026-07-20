"""Thin requests wrapper around the FastAPI backend, shared by every
Streamlit page. Every function returns parsed JSON (a dict/list) or raises
`requests.HTTPError` - pages are responsible for wrapping calls in a
try/except and showing `st.error` (see app.py's pattern), not this module.
"""
from __future__ import annotations

import requests

from config import settings

TIMEOUT = 30


def _get(path: str, params: dict | None = None):
    resp = requests.get(f"{settings.api_base_url}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def games_today(date: str | None = None) -> list[dict]:
    return _get("/games/today", {"date": date} if date else None)


def get_game(game_id: int) -> dict:
    return _get(f"/games/{game_id}")


def get_game_features(game_id: int) -> dict:
    return _get(f"/games/{game_id}/features")


def get_game_predictions(game_id: int) -> dict:
    return _get(f"/games/{game_id}/predictions")


def prediction_history(date_range: str, target_type: str | None = None) -> list[dict]:
    params = {"date_range": date_range}
    if target_type:
        params["target_type"] = target_type
    return _get("/predictions/history", params)


def backtest_results(model: str, date_range: str) -> dict:
    return _get("/backtest/results", {"model": model, "date_range": date_range})
