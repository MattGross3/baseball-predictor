"""GET /backtest/results (Section 8)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from sqlalchemy import extract, select
from sqlalchemy.orm import Session

from api.schemas import BacktestResultOut, HighConfidenceResultOut, SpreadResultOut
from backtest.backtest_engine import CONFIDENCE_THRESHOLD_DEFAULT, high_confidence_accuracy, run_backtest, simulate_run_line_bets
from database.db import get_db
from database.models import BacktestCache, Game

router = APIRouter(prefix="/backtest", tags=["backtest"])

# Synthetic "model name" for caching the run-line/spread simulation in the
# same backtest_cache table /backtest/results uses - spread betting isn't
# a registered model (it rides on totals_poisson's home/away split, see
# backtest_engine.simulate_run_line_bets), so there's no real model_name
# to key on, but reusing the existing cache table avoids a second one.
SPREAD_CACHE_KEY = "run_line_spread"


@router.get("/seasons", response_model=list[int])
def backtest_seasons(db: Session = Depends(get_db)):
    """Distinct season years with at least one final game - backs the ROI
    tab's per-season breakdown so it only ever asks about seasons that
    actually have data, rather than a hardcoded list."""
    years = db.execute(
        select(extract("year", Game.date)).where(Game.status == "final").distinct().order_by(extract("year", Game.date))
    ).scalars().all()
    return [int(y) for y in years]


@router.get("/results", response_model=BacktestResultOut)
def backtest_results(
    model: str = Query(..., description="Registered model name, e.g. moneyline_xgboost"),
    date_range: str = Query(..., description="YYYY-MM-DD,YYYY-MM-DD - should be a date range the model was NOT trained on"),
    refresh: bool = Query(False, description="Recompute instead of serving a cached result"),
    db: Session = Depends(get_db),
):
    """A backtest rebuilds the full feature set for every game in the
    range from scratch - genuinely slow (tens of seconds), and identical
    repeat queries (e.g. Model Comparison calling this for both models in
    a pair, or just revisiting the same page) used to pay that cost every
    single time. A finished date range's result is deterministic until
    the model is retrained, so it's cached in `backtest_cache` and served
    from there unless `refresh=true` is passed.
    """
    try:
        start_str, end_str = date_range.split(",")
        start_date, end_date = dt.date.fromisoformat(start_str), dt.date.fromisoformat(end_str)
    except ValueError as exc:
        raise HTTPException(400, "date_range must be 'YYYY-MM-DD,YYYY-MM-DD'") from exc

    cached = db.execute(
        select(BacktestCache).where(
            BacktestCache.model_name == model,
            BacktestCache.start_date == start_date,
            BacktestCache.end_date == end_date,
        )
    ).scalar_one_or_none()
    if cached is not None and not refresh:
        return BacktestResultOut(**cached.result_json, computed_at=cached.computed_at)

    try:
        result = run_backtest(db, model, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    if cached is None:
        cached = BacktestCache(model_name=model, start_date=start_date, end_date=end_date)
        db.add(cached)
    # jsonable_encoder - run_backtest's metrics can include numpy scalar
    # types that stdlib json.dumps (what the JSON column type uses) can't
    # serialize directly, unlike FastAPI's own response encoding.
    cached.result_json = jsonable_encoder(result)
    cached.computed_at = dt.datetime.now(dt.timezone.utc)
    db.commit()

    return BacktestResultOut(**result, computed_at=cached.computed_at)


def _high_confidence_cache_key(model: str, threshold: float) -> str:
    # Threshold is a second cache dimension /backtest/results doesn't have
    # (that endpoint's result only ever varies by model+date_range) - fold
    # it into a synthetic model_name the same way SPREAD_CACHE_KEY reuses
    # this table for a non-model result, rather than adding a column.
    return f"{model}_conf{int(round(threshold * 100))}"


@router.get("/high-confidence", response_model=HighConfidenceResultOut)
def high_confidence_results(
    model: str = Query(..., description="Registered model name, e.g. moneyline_xgboost"),
    date_range: str = Query(..., description="YYYY-MM-DD,YYYY-MM-DD"),
    threshold: float = Query(CONFIDENCE_THRESHOLD_DEFAULT, ge=0.5, lt=1.0, description="Confidence cutoff, e.g. 0.6 for 60%"),
    refresh: bool = Query(False, description="Recompute instead of serving a cached result"),
    db: Session = Depends(get_db),
):
    """Accuracy restricted to the model's own high-confidence picks (see
    backtest_engine.high_confidence_accuracy) - a different, harder
    question than /backtest/results' plain accuracy: is the model actually
    right more often specifically on the games it claims to be confident
    about, not just right on average across everything including the
    genuine coin-flip games where "confidence" isn't meaningful. Cached
    the same way /backtest/results is, under a synthetic model name that
    folds in the threshold.
    """
    try:
        start_str, end_str = date_range.split(",")
        start_date, end_date = dt.date.fromisoformat(start_str), dt.date.fromisoformat(end_str)
    except ValueError as exc:
        raise HTTPException(400, "date_range must be 'YYYY-MM-DD,YYYY-MM-DD'") from exc

    cache_key = _high_confidence_cache_key(model, threshold)
    cached = db.execute(
        select(BacktestCache).where(
            BacktestCache.model_name == cache_key,
            BacktestCache.start_date == start_date,
            BacktestCache.end_date == end_date,
        )
    ).scalar_one_or_none()
    if cached is not None and not refresh:
        return HighConfidenceResultOut(**cached.result_json, computed_at=cached.computed_at)

    try:
        result = high_confidence_accuracy(db, model, start_date, end_date, threshold)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    if cached is None:
        cached = BacktestCache(model_name=cache_key, start_date=start_date, end_date=end_date)
        db.add(cached)
    cached.result_json = jsonable_encoder(result)
    cached.computed_at = dt.datetime.now(dt.timezone.utc)
    db.commit()

    return HighConfidenceResultOut(**result, computed_at=cached.computed_at)


@router.get("/spread-results", response_model=SpreadResultOut)
def spread_results(
    date_range: str = Query(..., description="YYYY-MM-DD,YYYY-MM-DD"),
    refresh: bool = Query(False, description="Recompute instead of serving a cached result"),
    db: Session = Depends(get_db),
):
    """Run-line ("spread") backtest - see backtest_engine.simulate_run_line_bets
    for why this isn't just another /backtest/results?model=... call: a
    spread pick needs a predicted home/away split, which only the Poisson
    totals baseline produces, not a registered model of its own. Cached
    the same way /backtest/results is, under a synthetic model name.
    """
    try:
        start_str, end_str = date_range.split(",")
        start_date, end_date = dt.date.fromisoformat(start_str), dt.date.fromisoformat(end_str)
    except ValueError as exc:
        raise HTTPException(400, "date_range must be 'YYYY-MM-DD,YYYY-MM-DD'") from exc

    cached = db.execute(
        select(BacktestCache).where(
            BacktestCache.model_name == SPREAD_CACHE_KEY,
            BacktestCache.start_date == start_date,
            BacktestCache.end_date == end_date,
        )
    ).scalar_one_or_none()
    if cached is not None and not refresh:
        return SpreadResultOut(**cached.result_json, computed_at=cached.computed_at)

    result = simulate_run_line_bets(db, start_date, end_date)
    result["date_range"] = f"{start_date}..{end_date}"

    if cached is None:
        cached = BacktestCache(model_name=SPREAD_CACHE_KEY, start_date=start_date, end_date=end_date)
        db.add(cached)
    cached.result_json = jsonable_encoder(result)
    cached.computed_at = dt.datetime.now(dt.timezone.utc)
    db.commit()

    return SpreadResultOut(**result, computed_at=cached.computed_at)
