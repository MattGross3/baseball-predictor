"""GET /backtest/results (Section 8)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.schemas import BacktestResultOut
from backtest.backtest_engine import run_backtest
from database.db import get_db

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.get("/results", response_model=BacktestResultOut)
def backtest_results(
    model: str = Query(..., description="Registered model name, e.g. moneyline_xgboost"),
    date_range: str = Query(..., description="YYYY-MM-DD,YYYY-MM-DD - should be a date range the model was NOT trained on"),
    db: Session = Depends(get_db),
):
    try:
        start_str, end_str = date_range.split(",")
        start_date, end_date = dt.date.fromisoformat(start_str), dt.date.fromisoformat(end_str)
    except ValueError as exc:
        raise HTTPException(400, "date_range must be 'YYYY-MM-DD,YYYY-MM-DD'") from exc

    try:
        result = run_backtest(db, model, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    return BacktestResultOut(**result)
