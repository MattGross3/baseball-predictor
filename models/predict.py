"""
Generates predictions for a single game using the latest registered model
per target, and writes them to the `predictions` table.

This is the piece Section 10's "1 hr pre-game - final feature build +
generate predictions" job calls, and it's also what api/routers/models.py's
retrain endpoint and manual testing use to populate real prediction rows.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, ModelRegistryEntry, Prediction
from features.build_feature_matrix import build_game_feature_row, flatten_feature_row
from models.model_utils import load_model

log = logging.getLogger(__name__)

# Which registered model wins for each target when multiple are trained
# (e.g. moneyline_logistic vs moneyline_xgboost) - prefers the XGBoost/
# production model per Section 7's baseline-vs-production framing, falling
# back to the baseline if XGBoost hasn't been trained yet.
PREFERRED_MODEL_BY_TARGET = {
    "moneyline": ["moneyline_xgboost", "moneyline_logistic"],
    "nrfi": ["nrfi_logistic", "nrfi_xgboost"],  # spec: logistic is the default for NRFI unless XGBoost shows real lift
    "total": ["totals_xgboost", "totals_poisson"],
}


def _latest_entry(db: Session, model_name: str) -> ModelRegistryEntry | None:
    return db.execute(
        select(ModelRegistryEntry)
        .where(ModelRegistryEntry.model_name == model_name)
        .order_by(ModelRegistryEntry.trained_at.desc())
    ).scalars().first()


def _pick_model(db: Session, target: str) -> ModelRegistryEntry | None:
    for name in PREFERRED_MODEL_BY_TARGET.get(target, []):
        entry = _latest_entry(db, name)
        if entry is not None:
            return entry
    return None


def generate_prediction(db: Session, game_id: int, target: str, include_statcast_trend: bool = True) -> Prediction | None:
    """Builds a feature row and runs it through the latest model for
    `target`. Returns None (logs a warning) rather than raising if there's
    no trained model yet, or if the game is missing a starter (too early
    pre-game for the feature layer to have anything to work with).

    `include_statcast_trend` defaults True (full live Statcast trend +
    umpire history) - fine for a single live prediction. Bulk-populating
    predictions across many games (e.g. `generate_predictions_for_date`
    called in a loop, or a demo backfill) should pass False, both to stay
    fast and because that's the same feature set those models were
    trained on (build_training_matrix always uses False - see its
    docstring) - matching it here avoids train/serve skew.
    """
    entry = _pick_model(db, target)
    if entry is None:
        log.warning("No trained model registered for target '%s' - skipping prediction for game %s", target, game_id)
        return None

    game = db.get(Game, game_id)
    if game is None:
        raise ValueError(f"No game with id={game_id}")

    bundle = load_model(entry.file_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]

    nested = build_game_feature_row(db, game_id, include_statcast_trend=include_statcast_trend)
    flat = flatten_feature_row(nested)
    X = pd.DataFrame([flat]).reindex(columns=feature_cols, fill_value=0)
    # See train_totals._prep for why the explicit float cast matters here.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    predicted_value = None
    predicted_probability = None

    if target in ("moneyline", "nrfi"):
        predicted_probability = float(model.predict_proba(X)[:, 1][0])
    else:  # total - same compound-model dispatch as backtest_engine.run_backtest
        from models.train_totals import poisson_run_distribution, xgb_run_distribution

        dist = poisson_run_distribution(model, X) if ("home" in model and "away" in model) else xgb_run_distribution(model, X)
        predicted_value = dist["mean"]

    # Upsert on (game_id, target_type) - a game is predicted once and
    # served from that stored row from then on. Calling this again (e.g.
    # the scheduler's pregame job re-running, or a manual refresh after
    # retraining) updates that same row in place rather than piling up
    # duplicates - see the UniqueConstraint on Prediction.
    prediction = db.execute(
        select(Prediction).where(Prediction.game_id == game_id, Prediction.target_type == target)
    ).scalar_one_or_none()
    if prediction is None:
        prediction = Prediction(game_id=game_id, target_type=target)
        db.add(prediction)

    prediction.model_version = f"{entry.model_name}_{entry.version}"
    prediction.predicted_value = predicted_value
    prediction.predicted_probability = predicted_probability
    prediction.created_at = dt.datetime.now(dt.timezone.utc)
    db.flush()
    return prediction


def generate_predictions_for_date(
    db: Session,
    date: dt.date,
    targets: tuple[str, ...] = ("moneyline", "total", "nrfi"),
    include_statcast_trend: bool = True,
) -> int:
    games = db.execute(select(Game).where(Game.date == date)).scalars().all()
    written = 0
    for game in games:
        for target in targets:
            try:
                if generate_prediction(db, game.id, target, include_statcast_trend=include_statcast_trend) is not None:
                    written += 1
            except Exception:
                log.exception("Failed generating %s prediction for game %s", target, game.id)
    return written
