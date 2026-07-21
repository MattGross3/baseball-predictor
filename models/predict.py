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
from fastapi.encoders import jsonable_encoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, GameFeatureCache, ModelRegistryEntry, Prediction
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


def _cached_feature_row(db: Session, game_id: int, include_statcast_trend: bool = True, include_live_oaa: bool | None = None) -> dict:
    """Serve a cached feature row for a game when possible, and write it back
    to `game_feature_cache` the first time we have to build it. This cuts the
    repeat cost of a full feature build across multiple model calls."""
    cached = db.get(GameFeatureCache, game_id)
    if cached is not None:
        return cached.features_json

    nested = build_game_feature_row(db, game_id, include_statcast_trend=include_statcast_trend, include_live_oaa=include_live_oaa)
    if cached is None:
        cached = GameFeatureCache(game_id=game_id)
        db.add(cached)
    cached.features_json = jsonable_encoder(nested)
    cached.computed_at = dt.datetime.now(dt.timezone.utc)
    db.flush()
    return nested


def _score_with_model(entry: ModelRegistryEntry, target: str, nested: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """Runs one already-loaded model bundle against an already-built
    feature row. Split out from generate_prediction so multiple models can
    share one (expensive) feature build - see generate_all_model_predictions."""
    bundle = load_model(entry.file_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]
    medians = bundle.get("feature_medians", {})

    flat = flatten_feature_row(nested)
    X = pd.DataFrame([flat]).reindex(columns=feature_cols)
    # See train_totals._prep for why the explicit float cast matters here.
    # Missing values are filled with this model's training-set median
    # (persisted by model_utils.save_model), not 0 - 0 reads to the model
    # as "elite" for stats like ERA/win_pct, not "unknown", which distorts
    # exactly the thin-data cases (rookies, call-ups, an ingestion hiccup)
    # where a neutral fill matters most. Falls back to 0 only for legacy
    # model bundles saved before feature_medians was persisted.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(pd.Series(medians)).fillna(0).astype(float)

    if target in ("moneyline", "nrfi"):
        return None, float(model.predict_proba(X)[:, 1][0]), None, None

    # total - same compound-model dispatch as backtest_engine.run_backtest
    from models.train_totals import poisson_run_distribution, xgb_run_distribution

    dist = poisson_run_distribution(model, X) if ("home" in model and "away" in model) else xgb_run_distribution(model, X)
    return dist["mean"], None, dist.get("lambda_home"), dist.get("lambda_away")


def _upsert_prediction(
    db: Session,
    game_id: int,
    target: str,
    entry: ModelRegistryEntry,
    predicted_value: float | None,
    predicted_probability: float | None,
    predicted_home_value: float | None = None,
    predicted_away_value: float | None = None,
    nested: dict | None = None,
) -> Prediction:
    # Upsert on (game_id, target_type, model_name) - a given model family
    # predicts a game once and is served from that stored row from then on;
    # calling this again for the *same* model (e.g. the scheduler's pregame
    # job re-running, or a refresh after retraining) updates that row in
    # place rather than piling up duplicates. Different model families
    # (moneyline_logistic vs moneyline_xgboost) intentionally get their own
    # rows - Model Comparison's blended view needs both at once.
    prediction = db.execute(
        select(Prediction).where(
            Prediction.game_id == game_id, Prediction.target_type == target, Prediction.model_name == entry.model_name
        )
    ).scalar_one_or_none()
    if prediction is None:
        prediction = Prediction(game_id=game_id, target_type=target, model_name=entry.model_name)
        db.add(prediction)

    game = db.get(Game, game_id)
    home_prob = None
    away_prob = None
    market_home = nested.get("market_implied_probability_home") if nested is not None else None
    market_away = nested.get("market_implied_probability_away") if nested is not None else None

    if predicted_probability is not None:
        home_prob = round(predicted_probability, 6)
        away_prob = round(1.0 - predicted_probability, 6)

    prediction.model_version = f"{entry.model_name}_{entry.version}"
    prediction.predicted_value = predicted_value
    prediction.predicted_probability = predicted_probability
    prediction.predicted_side = "home" if (predicted_probability is not None and predicted_probability >= 0.5) else "away" if predicted_probability is not None else None
    prediction.predicted_home_value = predicted_home_value
    prediction.predicted_away_value = predicted_away_value
    prediction.home_probability = home_prob
    prediction.away_probability = away_prob
    prediction.market_home_probability = market_home
    prediction.market_away_probability = market_away
    prediction.confidence = round(abs((predicted_probability or 0.5) - 0.5) * 2, 6) if predicted_probability is not None else None
    prediction.target_unit = {
        "moneyline": "win_probability",
        "nrfi": "win_probability",
        "total": "runs",
    }.get(target, None)

    if game is not None and game.status == "final":
        if target == "moneyline":
            prediction.actual_outcome = "home_win" if game.home_score is not None and game.away_score is not None and game.home_score > game.away_score else "away_win" if game.home_score is not None and game.away_score is not None else None
        elif target == "nrfi":
            if game.first_inning_home_runs is not None and game.first_inning_away_runs is not None:
                prediction.actual_outcome = "nrfi" if game.first_inning_home_runs == 0 and game.first_inning_away_runs == 0 else "yrfi"
        else:
            prediction.actual_outcome = None

    prediction.created_at = dt.datetime.now(dt.timezone.utc)
    db.flush()
    return prediction


def generate_prediction(db: Session, game_id: int, target: str, include_statcast_trend: bool = True) -> Prediction | None:
    """Builds a feature row and runs it through the *preferred* model for
    `target` (see PREFERRED_MODEL_BY_TARGET) - the one Today's Slate/Game
    Detail display. Returns None (logs a warning) rather than raising if
    there's no trained model yet, or if the game is missing a starter (too
    early pre-game for the feature layer to have anything to work with).

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

    if db.get(Game, game_id) is None:
        raise ValueError(f"No game with id={game_id}")

    nested = _cached_feature_row(db, game_id, include_statcast_trend=include_statcast_trend)
    value, prob, home_value, away_value = _score_with_model(entry, target, nested)
    return _upsert_prediction(db, game_id, target, entry, value, prob, predicted_home_value=home_value, predicted_away_value=away_value, nested=nested)


def generate_all_model_predictions(db: Session, game_id: int, target: str, include_statcast_trend: bool = True) -> list[Prediction]:
    """Like generate_prediction, but scores *every* trained model family
    for `target` (not just the preferred one), sharing a single feature
    build across all of them. This is what populates enough data for the
    Model Comparison page's blended view to have something to blend -
    generate_prediction alone only ever stores one model's opinion per
    game, by design.
    """
    candidates = PREFERRED_MODEL_BY_TARGET.get(target, [])
    entries = [e for e in (_latest_entry(db, name) for name in candidates) if e is not None]
    if not entries:
        log.warning("No trained models registered for target '%s' - skipping game %s", target, game_id)
        return []

    if db.get(Game, game_id) is None:
        raise ValueError(f"No game with id={game_id}")

    nested = _cached_feature_row(db, game_id, include_statcast_trend=include_statcast_trend)

    predictions = []
    for entry in entries:
        value, prob, home_value, away_value = _score_with_model(entry, target, nested)
        predictions.append(_upsert_prediction(db, game_id, target, entry, value, prob, predicted_home_value=home_value, predicted_away_value=away_value, nested=nested))
    return predictions


def generate_predictions_for_date(
    db: Session,
    date: dt.date,
    targets: tuple[str, ...] = ("moneyline", "total", "nrfi"),
    include_statcast_trend: bool = True,
    all_models: bool = False,
) -> int:
    """`all_models=True` scores every trained model family per game (via
    generate_all_model_predictions) instead of just the preferred one -
    needed to seed data for Model Comparison's blended view. The
    scheduler's real pregame job leaves this False: it only needs the one
    prediction Today's Slate/Game Detail actually display.
    """
    games = db.execute(select(Game).where(Game.date == date)).scalars().all()
    written = 0
    for game in games:
        for target in targets:
            try:
                if all_models:
                    written += len(generate_all_model_predictions(db, game.id, target, include_statcast_trend=include_statcast_trend))
                elif generate_prediction(db, game.id, target, include_statcast_trend=include_statcast_trend) is not None:
                    written += 1
            except Exception:
                log.exception("Failed generating %s prediction for game %s", target, game.id)
    return written
