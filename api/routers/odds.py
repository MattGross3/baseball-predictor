"""POST /odds/refresh - manual odds poll (Section 4.4)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas import OddsRefreshOut
from config import settings
from database.db import get_db
from database.models import Game
from ingestion.api_budget import calls_this_month
from ingestion.odds_api import ingest_current_lines

router = APIRouter(prefix="/odds", tags=["odds"])


@router.post("/refresh", response_model=OddsRefreshOut)
def refresh_odds(db: Session = Depends(get_db)):
    """On-demand odds poll, for whenever a user doesn't want to wait for
    the next scheduled slot (scheduler/daily_jobs.py's ODDS_POLL_HOURS_ET).
    Still budget-checked the same way the scheduled job is -
    ingest_current_lines -> fetch_current_lines already refuses to call
    out once the monthly cap is hit, so this can't be used to bypass it,
    only to trigger the same call early.
    """
    limit = settings.odds_api_monthly_limit
    buffer = settings.odds_api_safety_buffer

    if not settings.has_odds_key:
        return OddsRefreshOut(
            written=0, calls_used_this_month=0, calls_remaining=limit,
            message="No ODDS_API_KEY configured - odds refresh is a no-op.",
        )

    used_before = calls_this_month(db, "the-odds-api")
    if used_before >= limit - buffer:
        return OddsRefreshOut(
            written=0,
            calls_used_this_month=used_before,
            calls_remaining=max(limit - buffer - used_before, 0),
            message=f"Monthly odds API budget exhausted ({used_before}/{limit} used) - try again next month.",
        )

    written = ingest_current_lines(db)
    db.commit()
    used_after = calls_this_month(db, "the-odds-api")

    # Compare against today's scheduled game count, not just "written > 0" -
    # a book not having posted a line yet for an 9:40pm game is completely
    # normal (lines for late-slate/West Coast games often post later in
    # the day) and shouldn't read as "the refresh silently did nothing" the
    # way a bare bare game count would.
    today_scheduled = db.execute(
        select(func.count()).select_from(Game).where(Game.date == dt.date.today(), Game.status == "scheduled")
    ).scalar_one()

    if written and today_scheduled and written < today_scheduled:
        message = (
            f"Refreshed - lines posted for {written} of {today_scheduled} of today's games; "
            "the rest haven't been posted by sportsbooks yet (normal for later start times)."
        )
    elif written:
        message = f"Refreshed - {written} game{'s' if written != 1 else ''} updated."
    else:
        message = "Refreshed, but no odds available yet - books may not have posted lines for later games."

    return OddsRefreshOut(
        written=written,
        calls_used_this_month=used_after,
        calls_remaining=max(limit - buffer - used_after, 0),
        message=message,
    )
