"""GET /backtest/results (Section 8)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from sqlalchemy import extract, select
from sqlalchemy.orm import Session

from api.schemas import BacktestResultOut
from backtest.backtest_engine import run_backtest
from database.db import get_db
from database.models import BacktestCache, Game

router = APIRouter(prefix="/backtest", tags=["backtest"])


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
