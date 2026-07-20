"""
One-off / catch-up backfill: ingest schedule, boxscores, lineups, linescores
and umpire assignments for every day in a date range.

Usage:
    python -m scripts.backfill_data 2025-04-01 2025-05-01

This is what you'd run once to seed enough history for the feature layer's
rolling windows (7d/14d bullpen ERA, season stats, etc.) to have something
to compute over, and again periodically to catch up on days the scheduler
missed. Commits once per date so an interrupted run can be resumed from
roughly where it left off instead of losing everything.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys

from database.db import session_scope
from ingestion import mlb_stats_api as mlb
from ingestion.umpire_scorecards import ingest_umpire_assignment
from ingestion.park_factors import seed_park_factors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def backfill_date(date: dt.date) -> dict:
    stats = {"games": 0, "boxscores": 0, "lineups": 0, "linescores": 0, "umpires": 0, "errors": 0}
    with session_scope() as db:
        games = mlb.ingest_schedule_for_date(db, date)
        stats["games"] = len(games)

        for game in games:
            if game.status != "final":
                continue
            try:
                mlb.ingest_boxscore(db, game.mlb_game_id)
                stats["boxscores"] += 1
                mlb.ingest_confirmed_lineup(db, game.mlb_game_id)
                stats["lineups"] += 1
                if mlb.ingest_linescore(db, game.mlb_game_id):
                    stats["linescores"] += 1
                ingest_umpire_assignment(db, game.mlb_game_id)
                stats["umpires"] += 1
            except Exception:
                log.exception("Failed ingesting game %s on %s", game.mlb_game_id, date)
                stats["errors"] += 1

        seed_park_factors(db)
    return stats


def backfill_range(start_date: dt.date, end_date: dt.date) -> None:
    day = start_date
    while day < end_date:
        stats = backfill_date(day)
        log.info("%s: %s", day, stats)
        day += dt.timedelta(days=1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.backfill_data YYYY-MM-DD YYYY-MM-DD")
        sys.exit(1)
    start = dt.date.fromisoformat(sys.argv[1])
    end = dt.date.fromisoformat(sys.argv[2])
    backfill_range(start, end)
