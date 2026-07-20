"""
Scheduling & automation (Section 10).

Run standalone as a long-lived process:

    python -m scheduler.daily_jobs

(docker-compose's `scheduler` service does exactly this.)

Design note on the two "N hours pre-game" jobs (confirmed lineups, final
predictions): the spec phrases these as if you'd schedule a one-off job per
game at exactly T-minus-3h / T-minus-1h. APScheduler could do that with a
dynamically added per-game trigger, but that adds real complexity (jobs to
add/cancel as games get postponed, doubleheaders added, etc.) for little
benefit. Instead both run on a short fixed interval and, each time they
fire, scan today's games and act on whichever ones currently fall inside
the target window - simpler, self-healing if a run is missed, and
idempotent (safe to fire more often than strictly necessary).
"""
from __future__ import annotations

import datetime as dt
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select

from config import settings
from database.db import session_scope
from database.models import Game, ModelRegistryEntry, Prediction
from ingestion import mlb_stats_api as mlb
from ingestion import odds_api
from ingestion.park_factors import seed_park_factors
from models.predict import generate_prediction
from scripts.backfill_data import backfill_date

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

LINEUP_POLL_WINDOW_HOURS = 3
PREDICTION_WINDOW_MINUTES = (50, 70)  # fire once a game's start_time is 50-70 min out
RETRAIN_MIN_INTERVAL_DAYS = 7  # spec: retrain weekly, not daily, to avoid overfitting to noise
RETRAIN_TRAIN_WINDOW_DAYS = 90


def job_morning_schedule() -> None:
    """06:00 ET - fetch_daily_schedule() + probable pitchers (included in
    ingest_schedule_for_date) for today's slate."""
    today = dt.date.today()
    with session_scope() as db:
        games = mlb.ingest_schedule_for_date(db, today)
        seed_park_factors(db)
    log.info("job_morning_schedule: ingested %d games for %s", len(games), today)


def job_poll_lineups() -> None:
    """Every 30 min from 3 hrs pre-game - poll for confirmed lineups."""
    now = dt.datetime.now(dt.timezone.utc)
    today = dt.date.today()
    with session_scope() as db:
        games = db.execute(
            select(Game).where(Game.date == today, Game.status.in_(["scheduled", "live"]))
        ).scalars().all()

        polled = 0
        for game in games:
            if game.start_time is None:
                continue
            hours_to_start = (game.start_time - now).total_seconds() / 3600
            if 0 <= hours_to_start <= LINEUP_POLL_WINDOW_HOURS:
                try:
                    mlb.ingest_confirmed_lineup(db, game.mlb_game_id)
                    polled += 1
                except Exception:
                    log.exception("Failed polling lineup for game %s", game.mlb_game_id)
    log.info("job_poll_lineups: polled %d games within %dh of first pitch", polled, LINEUP_POLL_WINDOW_HOURS)


def job_poll_odds() -> None:
    """Every 15 min - poll odds_api for line movement. No-ops (logs once)
    if ODDS_API_KEY isn't configured."""
    if not settings.has_odds_key:
        log.debug("job_poll_odds: ODDS_API_KEY not set, skipping")
        return
    with session_scope() as db:
        written = odds_api.ingest_current_lines(db)
    log.info("job_poll_odds: wrote %d odds snapshots", written)


def job_pregame_predictions() -> None:
    """~1 hr pre-game - final feature build + generate predictions."""
    now = dt.datetime.now(dt.timezone.utc)
    today = dt.date.today()
    low, high = PREDICTION_WINDOW_MINUTES

    with session_scope() as db:
        games = db.execute(
            select(Game).where(Game.date == today, Game.status == "scheduled")
        ).scalars().all()

        generated = 0
        for game in games:
            if game.start_time is None:
                continue
            minutes_to_start = (game.start_time - now).total_seconds() / 60
            if not (low <= minutes_to_start <= high):
                continue

            already_done = db.execute(
                select(Prediction.id).where(
                    Prediction.game_id == game.id,
                    Prediction.target_type == "moneyline",
                    Prediction.created_at >= dt.datetime.combine(today, dt.time.min, tzinfo=dt.timezone.utc),
                )
            ).first()
            if already_done:
                continue

            for target in ("moneyline", "total", "nrfi"):
                try:
                    if generate_prediction(db, game.id, target) is not None:
                        generated += 1
                except Exception:
                    log.exception("Failed generating %s prediction for game %s", target, game.id)
    log.info("job_pregame_predictions: generated %d predictions", generated)


def job_postgame_results() -> None:
    """Post-game - fetch_boxscore() + linescore/umpire/lineup, write
    actuals for backtesting. Reuses scripts/backfill_data.py's per-date
    ingest (idempotent upserts), which is exactly what's needed to pick up
    newly-finished games."""
    stats = backfill_date(dt.date.today())
    log.info("job_postgame_results: %s", stats)


def job_nightly_retrain_check() -> None:
    """02:00 ET nightly - only actually retrain if it's been >= 7 days
    since a model's last training run, per the spec's explicit
    weekly-not-daily rule (retraining on every day's noise overfits)."""
    from models.train_moneyline import run as run_moneyline
    from models.train_nrfi import run as run_nrfi
    from models.train_totals import run as run_totals

    today = dt.date.today()
    train_start = today - dt.timedelta(days=RETRAIN_TRAIN_WINDOW_DAYS)
    test_start = today - dt.timedelta(days=7)

    with session_scope() as db:
        for target, runner, model_names in (
            ("moneyline", run_moneyline, ["moneyline_logistic", "moneyline_xgboost"]),
            ("total", run_totals, ["totals_poisson", "totals_xgboost"]),
            ("nrfi", run_nrfi, ["nrfi_logistic", "nrfi_xgboost"]),
        ):
            last_trained = db.execute(
                select(ModelRegistryEntry.trained_at)
                .where(ModelRegistryEntry.model_name.in_(model_names))
                .order_by(ModelRegistryEntry.trained_at.desc())
            ).scalars().first()

            if last_trained and (dt.datetime.now(dt.timezone.utc) - last_trained).days < RETRAIN_MIN_INTERVAL_DAYS:
                log.info("job_nightly_retrain_check: %s trained %s ago, skipping (< %d days)", target, dt.datetime.now(dt.timezone.utc) - last_trained, RETRAIN_MIN_INTERVAL_DAYS)
                continue

            log.info("job_nightly_retrain_check: retraining %s (train %s..%s, test %s..%s)", target, train_start, test_start, test_start, today)
            try:
                runner(train_start, test_start, today)
            except Exception:
                log.exception("Retrain failed for %s", target)


def build_scheduler() -> BlockingScheduler:
    tz = settings.timezone
    scheduler = BlockingScheduler(timezone=tz)

    scheduler.add_job(job_morning_schedule, CronTrigger(hour=6, minute=0, timezone=tz), id="morning_schedule")
    scheduler.add_job(job_poll_lineups, IntervalTrigger(minutes=30), id="poll_lineups")
    scheduler.add_job(job_poll_odds, IntervalTrigger(minutes=15), id="poll_odds")
    scheduler.add_job(job_pregame_predictions, IntervalTrigger(minutes=10), id="pregame_predictions")
    scheduler.add_job(job_postgame_results, IntervalTrigger(minutes=20), id="postgame_results")
    scheduler.add_job(job_nightly_retrain_check, CronTrigger(hour=2, minute=0, timezone=tz), id="nightly_retrain_check")

    return scheduler


if __name__ == "__main__":
    log.info("Starting scheduler (timezone=%s)", settings.timezone)
    build_scheduler().start()
