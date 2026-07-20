"""POST /models/retrain (Section 8) - admin-only, triggers a retraining job."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException

from api.schemas import RetrainRequest, RetrainResponse
from config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["models"])

_TRAIN_RUNNERS = {}  # populated lazily below to avoid importing heavy training deps at API startup


def _get_runner(target: str):
    global _TRAIN_RUNNERS
    if not _TRAIN_RUNNERS:
        from models.train_moneyline import run as run_moneyline
        from models.train_nrfi import run as run_nrfi
        from models.train_totals import run as run_totals

        _TRAIN_RUNNERS = {"moneyline": run_moneyline, "total": run_totals, "nrfi": run_nrfi}
    return _TRAIN_RUNNERS.get(target)


def _check_admin(x_admin_key: str | None) -> None:
    if not settings.admin_api_key:
        log.warning("POST /models/retrain called with no ADMIN_API_KEY configured - allowing (dev mode)")
        return
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(401, "Missing or invalid X-Admin-Key header")


@router.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest, background_tasks: BackgroundTasks, x_admin_key: str | None = Header(default=None)):
    _check_admin(x_admin_key)

    runner = _get_runner(request.target)
    if runner is None:
        raise HTTPException(400, f"Unknown target '{request.target}' - expected moneyline, total, or nrfi")

    # Training takes minutes, not milliseconds (see features/build_feature_matrix.py's
    # per-game cost) - runs in the background so the request returns immediately
    # rather than tying up a worker/timing out a client.
    background_tasks.add_task(runner, request.train_start, request.test_start, request.test_end)

    return RetrainResponse(
        status="started",
        target=request.target,
        detail=f"Retraining {request.target} in the background: train {request.train_start}..{request.test_start}, test {request.test_start}..{request.test_end}. Check the model_registry table or GET /backtest/results once it lands.",
    )
