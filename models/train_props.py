"""
Player props (Section 7.4) - stretch goal per Section 1's build order, so
this is intentionally leaner than the game-level models: one shared
feature-building function plus three targets.

- HR (binary, XGBClassifier): did this batter hit a home run in this game?
- Hits (regression, XGBRegressor): how many hits did this batter get?
- Strikeouts (regression, XGBRegressor): how many Ks did the *starting
  pitcher* record? Betting markets overwhelmingly offer pitcher-strikeout
  props, not batter-strikeout props, so "strikeouts" here means the
  pitcher side - flagged explicitly since the spec's wording ("HR, hits,
  strikeouts") could be read either way.

Features lean on Statcast batted-ball quality (barrel%, hard-hit%) per the
spec's note that these targets rely more heavily on that data than the
game-level models do, plus the opposing starter's rate stats and platoon
handedness - all pulled from the feature modules already built for the
game-level targets, not reimplemented here.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from xgboost import XGBClassifier, XGBRegressor

from database.db import session_scope
from database.models import BatterGameLog, Game, PitcherGameLog, Player
from features.pitcher_features import compute_starter_features
from ingestion.statcast import compute_batter_statcast_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _batter_prop_row(db: Session, log_row: BatterGameLog, game: Game) -> dict | None:
    player = db.get(Player, log_row.player_id)
    if player is None:
        return None
    is_home = player.team_id == game.home_team_id
    opponent_pitcher_id = game.away_starter_id if is_home else game.home_starter_id
    opponent_hand = None
    if opponent_pitcher_id:
        opp = db.get(Player, opponent_pitcher_id)
        opponent_hand = opp.throws if opp else None

    statcast = compute_batter_statcast_summary(player.mlb_player_id, game.date, lookback_days=30)
    venue = game.venue

    return {
        "avg_exit_velo": statcast["avg_exit_velo"],
        "barrel_pct": statcast["barrel_pct"],
        "hard_hit_pct": statcast["hard_hit_pct"],
        "platoon_favorable": int(
            player.bats == "S" or (player.bats == "L" and opponent_hand == "R") or (player.bats == "R" and opponent_hand == "L")
        ) if player.bats and opponent_hand else None,
        "park_factor_hr": venue.park_factor_hr if venue else None,
        "park_factor_runs": venue.park_factor_runs if venue else None,
        "hr_label": int((log_row.hr or 0) >= 1),
        "hits_label": log_row.hits or 0,
    }


def build_batter_prop_matrix(db: Session, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    logs = db.execute(
        select(BatterGameLog, Game)
        .join(Game, Game.id == BatterGameLog.game_id)
        .where(Game.date >= start_date, Game.date < end_date, Game.status == "final")
    ).all()

    rows = []
    for log_row, game in logs:
        if not log_row.at_bats:  # skip pinch-hit-only/no-AB appearances
            continue
        row = _batter_prop_row(db, log_row, game)
        if row:
            row["date"] = game.date
            rows.append(row)
    return pd.DataFrame(rows)


def build_pitcher_strikeout_matrix(db: Session, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    logs = db.execute(
        select(PitcherGameLog, Game)
        .join(Game, Game.id == PitcherGameLog.game_id)
        .where(PitcherGameLog.is_starter.is_(True))
        .where(Game.date >= start_date, Game.date < end_date, Game.status == "final")
    ).all()

    rows = []
    for log_row, game in logs:
        player = db.get(Player, log_row.player_id)
        if player is None:
            continue
        opponent_team_id = game.away_team_id if player.team_id == game.home_team_id else game.home_team_id
        features = compute_starter_features(db, log_row.player_id, game.date, opponent_team_id=opponent_team_id)
        rows.append(
            {
                "date": game.date,
                "era_season": features["era_season"],
                "k_pct_rolling": features["k_pct_rolling"],
                "days_rest": features["days_rest"],
                "k_label": log_row.k or 0,
            }
        )
    return pd.DataFrame(rows)


def _prep(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median(numeric_only=True))


def train_hr_classifier(df: pd.DataFrame) -> XGBClassifier:
    X = _prep(df, ["date", "hr_label", "hits_label"])
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric="logloss")
    model.fit(X, df["hr_label"])
    return model


def train_hits_regressor(df: pd.DataFrame) -> XGBRegressor:
    X = _prep(df, ["date", "hr_label", "hits_label"])
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X, df["hits_label"])
    return model


def train_strikeouts_regressor(df: pd.DataFrame) -> XGBRegressor:
    X = _prep(df, ["date", "k_label"])
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(X, df["k_label"])
    return model


def run(start_date: dt.date, end_date: dt.date) -> None:
    with session_scope() as db:
        batter_df = build_batter_prop_matrix(db, start_date, end_date)
        log.info("Batter prop rows: %d", len(batter_df))
        if not batter_df.empty:
            train_hr_classifier(batter_df)
            train_hits_regressor(batter_df)
            log.info("Trained HR classifier and hits regressor (HR rate in data: %.1f%%)", 100 * batter_df["hr_label"].mean())

        pitcher_df = build_pitcher_strikeout_matrix(db, start_date, end_date)
        log.info("Pitcher strikeout prop rows: %d", len(pitcher_df))
        if not pitcher_df.empty:
            train_strikeouts_regressor(pitcher_df)
            log.info("Trained strikeouts regressor (mean Ks in data: %.1f)", pitcher_df["k_label"].mean())


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        run(dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2]))
    else:
        print("Usage: python -m models.train_props START_DATE END_DATE")
