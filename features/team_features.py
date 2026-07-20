"""
Team-level record/form features (Section 6).
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, Team
from ingestion.statcast import fetch_team_defense_oaa


def _season_games(db: Session, team_id: int, as_of_date: dt.date):
    season_start = dt.date(as_of_date.year, 3, 1)
    return db.execute(
        select(Game)
        .where((Game.home_team_id == team_id) | (Game.away_team_id == team_id))
        .where(Game.date >= season_start, Game.date < as_of_date)
        .where(Game.status == "final")
        .order_by(Game.date.desc())
    ).scalars().all()


def compute_team_features(db: Session, team_id: int, as_of_date: dt.date) -> dict:
    games = _season_games(db, team_id, as_of_date)

    wins = losses = runs_for = runs_against = 0
    home_wins = home_losses = away_wins = away_losses = 0

    for g in games:
        is_home = g.home_team_id == team_id
        team_score = g.home_score if is_home else g.away_score
        opp_score = g.away_score if is_home else g.home_score
        if team_score is None or opp_score is None:
            continue
        runs_for += team_score
        runs_against += opp_score
        won = team_score > opp_score
        wins += int(won)
        losses += int(not won)
        if is_home:
            home_wins += int(won)
            home_losses += int(not won)
        else:
            away_wins += int(won)
            away_losses += int(not won)

    total = wins + losses
    last_10 = games[:10]
    last_10_wins = sum(
        1
        for g in last_10
        if (g.home_score or 0) > (g.away_score or 0) and g.home_team_id == team_id
        or (g.away_score or 0) > (g.home_score or 0) and g.away_team_id == team_id
    )

    win_pct_season = round(wins / total, 3) if total else None
    pythag = None
    if runs_for or runs_against:
        rf2, ra2 = runs_for**1.83, runs_against**1.83
        pythag = round(rf2 / (rf2 + ra2), 3) if (rf2 + ra2) else None

    return {
        "win_pct_season": win_pct_season,
        "win_pct_last_10": round(last_10_wins / len(last_10), 3) if last_10 else None,
        "run_diff_season": runs_for - runs_against if games else None,
        "pythag_win_pct": pythag,
        "home_away_win_pct": {
            "home": round(home_wins / (home_wins + home_losses), 3) if (home_wins + home_losses) else None,
            "away": round(away_wins / (away_wins + away_losses), 3) if (away_wins + away_losses) else None,
        },
        "oaa_defense_rating": _oaa_rating(db, team_id, as_of_date.year),
    }


@lru_cache(maxsize=8)
def _oaa_season_table(season: int):
    return fetch_team_defense_oaa(season)


def _oaa_rating(db: Session, team_id: int, season: int) -> float | None:
    """Team OAA for the season from Baseball Savant. Savant's leaderboard
    uses short club names ("Angels") while our `teams.name` stores the full
    name ("Los Angeles Angels") from the MLB Stats API, so we match on
    suffix rather than requiring a separate id-mapping table."""
    table = _oaa_season_table(season)
    if table.empty:
        return None
    team = db.get(Team, team_id)
    if team is None:
        return None
    match = table[table["team_name"].apply(lambda short: team.name.endswith(short))]
    if match.empty:
        return None
    return float(match.iloc[0]["outs_above_average"])
