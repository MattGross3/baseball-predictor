"""FastAPI app entrypoint (Section 8). Run with:

    uvicorn api.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import backtest, games, models, predictions
from config import settings

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Baseball Prediction API",
    description="MLB moneyline / totals / NRFI / player-prop predictions, features, odds and backtest results.",
    version="0.1.0",
)

# Wide open by default since this is a single-tenant personal-project API
# behind no auth for its read endpoints; tighten before exposing publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router)
app.include_router(predictions.router)
app.include_router(backtest.router)
app.include_router(models.router)


@app.get("/health")
def health():
    return {"status": "ok"}
