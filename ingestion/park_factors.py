"""
Park factors (Section 4.7): a static reference table, refreshed yearly.

There's no reliable free API for this - FanGraphs' park factor pages block
scraping the same way their leaderboards do (see ingestion/fangraphs.py),
and Baseball Savant's park factor leaderboard is a JS-rendered page without
a public JSON endpoint. So this is seeded from a checked-in CSV of
approximate, publicly-known multi-year park factor values
(100 = league average; ingestion/reference_data/park_factors.csv).

These are reasonable defaults, not exact single-season figures - refresh
the CSV once a year from FanGraphs' or Savant's park factor pages (copy the
numbers by hand; both block programmatic scraping) if you want current-year
precision.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Venue

log = logging.getLogger(__name__)

CSV_PATH = Path(__file__).parent / "reference_data" / "park_factors.csv"


def load_park_factors_csv() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def seed_park_factors(db: Session) -> int:
    """Match CSV rows to `venues` by name and set park_factor_runs/hr.

    Only updates venues we already know about (i.e. that have appeared in
    an ingested schedule) - run this after `ingest_schedule_for_date` has
    populated the venues table, and re-run any time you add new venues.
    """
    rows = load_park_factors_csv()
    by_name = {r["venue_name"]: r for r in rows}

    updated = 0
    for venue in db.execute(select(Venue)).scalars():
        row = by_name.get(venue.name)
        if row is None:
            continue
        venue.park_factor_runs = float(row["park_factor_runs"])
        venue.park_factor_hr = float(row["park_factor_hr"])
        updated += 1

    if updated < len(rows):
        log.info(
            "seed_park_factors: matched %d/%d CSV rows to known venues (ingest more schedules to pick up the rest)",
            updated, len(rows),
        )
    return updated
