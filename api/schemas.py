"""Pydantic response models for the API (Section 8)."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    abbreviation: str
    league: str
    division: str


class VenueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    city: str | None = None
    park_factor_runs: float
    park_factor_hr: float
    roof_type: str | None = None


class GameOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    mlb_game_id: int
    date: dt.date
    start_time: dt.datetime | None = None
    status: str
    home_team: TeamOut
    away_team: TeamOut
    venue: VenueOut | None = None
    home_score: int | None = None
    away_score: int | None = None
    is_doubleheader: bool
    game_number_in_series: int


class GameFeaturesOut(BaseModel):
    game_id: int
    features: dict[str, Any]


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    game_id: int
    model_name: str
    model_version: str
    target_type: str
    predicted_value: float | None = None
    predicted_probability: float | None = None
    created_at: dt.datetime


class GamePredictionsOut(BaseModel):
    game_id: int
    predictions: list[PredictionOut]
    edge_vs_market: dict[str, Any] | None = None


class OddsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    game_id: int
    timestamp: dt.datetime
    moneyline_home: int | None = None
    moneyline_away: int | None = None
    run_line: float | None = None
    run_line_odds: int | None = None
    total: float | None = None
    over_odds: int | None = None
    under_odds: int | None = None
    source: str


class BacktestResultOut(BaseModel):
    model: str
    target_type: str
    date_range: str
    # Classification targets (moneyline, nrfi):
    accuracy: float | None = None
    log_loss: float | None = None
    brier_score: float | None = None
    # Regression target (total) - backtest_engine.run_backtest returns
    # these via regression_metrics() instead of the three above.
    mae: float | None = None
    rmse: float | None = None
    roi_flat_bet: float | None = None
    roi_kelly: float | None = None
    clv_avg: float | None = None
    n_bets: int


class RetrainRequest(BaseModel):
    target: str  # "moneyline" | "total" | "nrfi"
    train_start: dt.date
    test_start: dt.date
    test_end: dt.date


class RetrainResponse(BaseModel):
    status: str
    target: str
    detail: str
