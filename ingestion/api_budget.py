"""
Monthly call-budget tracking for metered external APIs - built for The
Odds API's free tier (500 requests/month), since going over it either
costs money or stops working entirely.

Deliberately simple: not a token bucket or sliding window, just "how many
calls have we logged since the 1st of this calendar month" - which is what
these free tiers actually reset against. `within_budget` is a pure read
(safe to call as often as you like to check before doing work);
`record_call` should only be called once a request has actually gone out,
right next to the `requests.get(...)` call it's tracking, so the count
can never drift from reality.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import ApiCallLog

log = logging.getLogger(__name__)


def _month_start(now: dt.datetime | None = None) -> dt.datetime:
    now = now or dt.datetime.now(dt.timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def calls_this_month(db: Session, api_name: str) -> int:
    return db.execute(
        select(func.count())
        .select_from(ApiCallLog)
        .where(ApiCallLog.api_name == api_name, ApiCallLog.called_at >= _month_start())
    ).scalar_one()


def record_call(db: Session, api_name: str) -> None:
    db.add(ApiCallLog(api_name=api_name))
    db.flush()


def within_budget(db: Session, api_name: str, monthly_limit: int, safety_buffer: int = 0) -> bool:
    """False once `monthly_limit - safety_buffer` calls have been used this
    month. The buffer exists so a few calls stay available for manual
    debugging/testing near the end of the month without a scheduled job
    tipping the account over the hard cap."""
    used = calls_this_month(db, api_name)
    remaining = monthly_limit - safety_buffer - used
    if remaining <= 0:
        log.warning(
            "%s: monthly call budget exhausted (%d/%d used this month, %d held back as a safety buffer) - skipping call",
            api_name, used, monthly_limit, safety_buffer,
        )
        return False
    return True
