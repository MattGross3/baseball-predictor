"""GET /predictions/history (Section 8)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.schemas import PredictionOut
from database.db import get_db
from database.models import Game, Prediction

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/history", response_model=list[PredictionOut])
def prediction_history(
    date_range: str = Query(..., description="YYYY-MM-DD,YYYY-MM-DD (inclusive start, exclusive end)"),
    target_type: str | None = Query(None, description="Filter to moneyline | total | nrfi | prop_hr | prop_hits | prop_strikeouts"),
    db: Session = Depends(get_db),
):
    try:
        start_str, end_str = date_range.split(",")
        start_date, end_date = dt.date.fromisoformat(start_str), dt.date.fromisoformat(end_str)
    except ValueError as exc:
        raise HTTPException(400, "date_range must be 'YYYY-MM-DD,YYYY-MM-DD'") from exc

    stmt = (
        select(Prediction)
        .join(Game, Game.id == Prediction.game_id)
        .where(Game.date >= start_date, Game.date < end_date)
        .order_by(Prediction.created_at)
    )
    if target_type:
        stmt = stmt.where(Prediction.target_type == target_type)

    return db.execute(stmt).scalars().all()
