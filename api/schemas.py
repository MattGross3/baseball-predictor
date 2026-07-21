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
    computed_at: dt.datetime | None = None


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    game_id: int
    model_name: str
    model_version: str
    target_type: str
    predicted_value: float | None = None
    predicted_probability: float | None = None
    predicted_side: str | None = None
    predicted_home_value: float | None = None
    predicted_away_value: float | None = None
    home_probability: float | None = None
    away_probability: float | None = None
    market_home_probability: float | None = None
    market_away_probability: float | None = None
    confidence: float | None = None
    actual_outcome: str | None = None
    target_unit: str | None = None
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


class GameSlateSummaryOut(BaseModel):
    game_id: int
    moneyline_probability: float | None = None
    total_prediction: float | None = None
    total_home_prediction: float | None = None
    total_away_prediction: float | None = None
    nrfi_probability: float | None = None
    pick_type: str | None = None
    pick_side: str | None = None
    projected_value: float | None = None
    market_value: float | None = None
    confidence: float | None = None
    edge: float | None = None
    # Raw odds (moneyline/run line/total prices), not just the computed
    # pick above - the dashboard shows these directly on Today's Slate so
    # a user can see the real market number next to the model's take, not
    # just "the model likes the over."
    latest_odds: OddsOut | None = None
    run_line_pick_side: str | None = None
    run_line_edge: float | None = None
    # Pitching matchup - season ERA/WHIP for each side's starter, computed
    # the same way features/pitcher_features.py does for the model (not a
    # separate stat source), so this always agrees with what the model
    # actually saw.
    home_starter_name: str | None = None
    home_starter_era: float | None = None
    home_starter_whip: float | None = None
    away_starter_name: str | None = None
    away_starter_era: float | None = None
    away_starter_whip: float | None = None


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
    # Raw win/loss counts (classification targets only) - the ROI tab's
    # fallback for date ranges with no odds coverage to compute ROI from.
    n_games: int | None = None
    wins: int | None = None
    losses: int | None = None
    computed_at: dt.datetime | None = None


class SpreadResultOut(BaseModel):
    date_range: str
    roi_flat_bet: float | None = None
    n_bets: int
    n_games: int
    wins: int
    losses: int
    computed_at: dt.datetime | None = None


class OddsRefreshOut(BaseModel):
    written: int
    calls_used_this_month: int
    calls_remaining: int
    message: str


class ModelInfoOut(BaseModel):
    model_name: str
    target_type: str
    version: str
    trained_at: dt.datetime
    metrics: dict[str, Any]


class RetrainRequest(BaseModel):
    target: str  # "moneyline" | "total" | "nrfi"
    train_start: dt.date
    test_start: dt.date
    test_end: dt.date


class RetrainResponse(BaseModel):
    status: str
    target: str
    detail: str
