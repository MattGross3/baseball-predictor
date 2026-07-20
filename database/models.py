"""
SQLAlchemy ORM models implementing the schema in Section 5 of the spec.

Design notes:
- Every table that mirrors an external entity (team, player, venue, game)
  carries an `mlb_*_id` column holding the MLB Stats API's own integer ID.
  Ingestion code upserts on that column so daily re-runs are idempotent
  instead of creating duplicates.
- Timestamps are timezone-aware UTC; display-layer code converts to the
  configured local timezone (see config.settings.timezone).
"""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class GameStatus(str, enum.Enum):
    scheduled = "scheduled"
    live = "live"
    final = "final"
    postponed = "postponed"
    cancelled = "cancelled"


class TargetType(str, enum.Enum):
    moneyline = "moneyline"
    total = "total"
    nrfi = "nrfi"
    prop_hr = "prop_hr"
    prop_hits = "prop_hits"
    prop_strikeouts = "prop_strikeouts"


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_team_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    abbreviation: Mapped[str] = mapped_column(String(10))
    league: Mapped[str] = mapped_column(String(30))  # e.g. "American League"
    division: Mapped[str] = mapped_column(String(40))  # e.g. "American League Central"

    players: Mapped[list["Player"]] = relationship(back_populates="team")


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_venue_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(150))
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    surface_type: Mapped[str] = mapped_column(String(30), nullable=True)
    roof_type: Mapped[str] = mapped_column(String(30), nullable=True)  # open / retractable / dome
    park_factor_runs: Mapped[float] = mapped_column(Float, default=100.0)  # 100 = league avg
    park_factor_hr: Mapped[float] = mapped_column(Float, default=100.0)
    lat: Mapped[float] = mapped_column(Float, nullable=True)
    lon: Mapped[float] = mapped_column(Float, nullable=True)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_player_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(150))
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"), nullable=True)
    position: Mapped[str] = mapped_column(String(10), nullable=True)
    bats: Mapped[str] = mapped_column(String(1), nullable=True)  # L/R/S
    throws: Mapped[str] = mapped_column(String(1), nullable=True)  # L/R

    team: Mapped[Team | None] = relationship(back_populates="players")


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        UniqueConstraint("mlb_game_id", name="uq_games_mlb_game_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    mlb_game_id: Mapped[int] = mapped_column(Integer, index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    start_time: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    venue_id: Mapped[int | None] = mapped_column(ForeignKey("venues.id"))
    status: Mapped[str] = mapped_column(String(20), default=GameStatus.scheduled.value)
    home_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_doubleheader: Mapped[bool] = mapped_column(Boolean, default=False)
    game_number_in_series: Mapped[int] = mapped_column(Integer, default=1)

    # Not in the spec's literal Section 5 schema, but required for the
    # feature/model layers in Section 6/7: without a pointer to the
    # starting pitcher, `build_game_feature_row` has no way to know whose
    # stats to pull. Populated from the probable pitcher at schedule-ingest
    # time and overwritten with the confirmed starter once the boxscore
    # is available (probables change ~10-15% of the time before first pitch).
    home_starter_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    away_starter_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True)

    # Also not in the spec's literal schema: needed to label the NRFI/YRFI
    # target (Section 7.3) - whether any run scored in the top+bottom 1st.
    # Populated post-game from the schedule endpoint's linescore hydrate.
    first_inning_home_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_inning_away_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    venue: Mapped[Venue | None] = relationship()
    home_starter: Mapped["Player | None"] = relationship(foreign_keys=[home_starter_id])
    away_starter: Mapped["Player | None"] = relationship(foreign_keys=[away_starter_id])


class PitcherGameLog(Base):
    __tablename__ = "pitcher_game_logs"
    __table_args__ = (UniqueConstraint("player_id", "game_id", name="uq_pitcher_game"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    is_starter: Mapped[bool] = mapped_column(Boolean, default=False)
    ip: Mapped[float] = mapped_column(Float, nullable=True)
    er: Mapped[int] = mapped_column(Integer, nullable=True)
    k: Mapped[int] = mapped_column(Integer, nullable=True)
    bb: Mapped[int] = mapped_column(Integer, nullable=True)
    h: Mapped[int] = mapped_column(Integer, nullable=True)
    hr: Mapped[int] = mapped_column(Integer, nullable=True)
    pitch_count: Mapped[int] = mapped_column(Integer, nullable=True)
    avg_velo: Mapped[float] = mapped_column(Float, nullable=True)
    days_rest: Mapped[int] = mapped_column(Integer, nullable=True)


class BatterGameLog(Base):
    __tablename__ = "batter_game_logs"
    __table_args__ = (UniqueConstraint("player_id", "game_id", name="uq_batter_game"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    batting_order_position: Mapped[int] = mapped_column(Integer, nullable=True)
    at_bats: Mapped[int] = mapped_column(Integer, nullable=True)
    hits: Mapped[int] = mapped_column(Integer, nullable=True)
    hr: Mapped[int] = mapped_column(Integer, nullable=True)
    bb: Mapped[int] = mapped_column(Integer, nullable=True)
    k: Mapped[int] = mapped_column(Integer, nullable=True)
    vs_pitcher_hand: Mapped[str] = mapped_column(String(1), nullable=True)


class Lineup(Base):
    __tablename__ = "lineups"
    __table_args__ = (UniqueConstraint("game_id", "team_id", "player_id", name="uq_lineup_slot"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    batting_order_position: Mapped[int] = mapped_column(Integer)
    confirmed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BullpenUsage(Base):
    __tablename__ = "bullpen_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    pitcher_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    pitches_thrown: Mapped[int] = mapped_column(Integer, nullable=True)
    back_to_back_flag: Mapped[bool] = mapped_column(Boolean, default=False)


class WeatherSnapshot(Base):
    __tablename__ = "weather_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    temp_f: Mapped[float] = mapped_column(Float, nullable=True)
    wind_speed_mph: Mapped[float] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[str] = mapped_column(String(10), nullable=True)
    humidity: Mapped[float] = mapped_column(Float, nullable=True)
    condition: Mapped[str] = mapped_column(String(50), nullable=True)


class Umpire(Base):
    __tablename__ = "umpires"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    umpire_name: Mapped[str] = mapped_column(String(100))
    home_plate: Mapped[bool] = mapped_column(Boolean, default=True)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    moneyline_home: Mapped[int] = mapped_column(Integer, nullable=True)
    moneyline_away: Mapped[int] = mapped_column(Integer, nullable=True)
    run_line: Mapped[float] = mapped_column(Float, nullable=True)
    run_line_odds: Mapped[int] = mapped_column(Integer, nullable=True)
    total: Mapped[float] = mapped_column(Float, nullable=True)
    over_odds: Mapped[int] = mapped_column(Integer, nullable=True)
    under_odds: Mapped[int] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="the-odds-api")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        # One stored prediction per game per target *per model family* - a
        # game is played once, so a given model should predict it once
        # (computed, then served from this row from then on, not
        # re-inserted as a growing duplicate log every time
        # generate_prediction() is called - see models/predict.py's
        # upsert). Keying on model_name (not model_version, which bumps
        # every retrain) means re-training the same model family updates
        # its row in place, while *different* model families (e.g.
        # moneyline_logistic vs moneyline_xgboost) each get their own row -
        # required for the Model Comparison page's blended view, which
        # needs both models' predictions for the same game at once.
        UniqueConstraint("game_id", "target_type", "model_name", name="uq_prediction_game_target_model"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(50))
    model_version: Mapped[str] = mapped_column(String(50))
    target_type: Mapped[str] = mapped_column(String(30))
    predicted_value: Mapped[float] = mapped_column(Float, nullable=True)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100))
    target_type: Mapped[str] = mapped_column(String(30))
    version: Mapped[str] = mapped_column(String(50))
    trained_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    file_path: Mapped[str] = mapped_column(String(300))


class ApiCallLog(Base):
    """One row per outbound call to a metered external API - currently just
    The Odds API's free tier (500 requests/month). See
    ingestion/api_budget.py, which is what actually reads/writes this."""

    __tablename__ = "api_call_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_name: Mapped[str] = mapped_column(String(50), index=True)
    called_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
