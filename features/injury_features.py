"""
Injury features - a real gap identified during review: nothing in the
feature layer read injury status at all.

Reconstructed from real MLB transaction history (placed-on-IL /
activated-from-IL moves), not a live "current roster" snapshot. That
matters for two reasons:

1. It's genuinely historical - unlike Statcast trend or umpire zone
   history (this file's nearest analogues, both live-only), a
   transaction carries a real date, so `_players_on_il` can answer "who
   was hurt as of this specific past date" and is safe to use in bulk
   training, not just live single-game prediction. No include_statcast_
   trend-style gate needed here.
2. The obvious-looking approach - `rosterType=injuredList` on the roster
   endpoint - doesn't actually work: MLB's own `/rosterTypes` meta-endpoint
   confirms that parameter value doesn't exist, and passing it silently
   fell back to the full active roster (every "injured" player came back
   with status "Active"). See ingestion/mlb_stats_api.fetch_transactions
   for the full story.

Team attribution caveat: transactions are fetched per our *current*
`teams.mlb_team_id`, so a team's injury history only reflects moves made
while a player transactions show them on that team - correct for
attribution, but if the team itself doesn't change this isn't an issue in
practice (unlike the player.team_id caveat elsewhere in features/, which
is about *players* changing teams).
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from functools import lru_cache

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import Game, Lineup, PitcherGameLog, Player, Team
from ingestion.mlb_stats_api import fetch_transactions

log = logging.getLogger(__name__)

SEASON_START_MONTH_DAY = (3, 1)
MIN_GAMES_TO_COUNT_AS_REGULAR = 10

_PLACED_RE = re.compile(r"placed .* on the .*injured list", re.IGNORECASE)
_ACTIVATED_RE = re.compile(r"activated .* from the .*injured list", re.IGNORECASE)


@lru_cache(maxsize=128)
def _season_transactions(team_mlb_id: int, season: int) -> tuple[dict, ...]:
    """One real HTTP call per (team, season), cached in-process - a
    training-matrix build calls this once per game, and refetching the
    same ~150-row season transaction list every time would be wasteful
    (a single team/season pull is well under a second; the DataFrame
    build for a full season is hundreds of games). Returns a tuple, not a
    list, so it's hashable for lru_cache to key on.
    """
    start = dt.date(season, *SEASON_START_MONTH_DAY)
    end = dt.date(season, 12, 31)
    try:
        return tuple(fetch_transactions(team_mlb_id, start, end))
    except Exception as exc:
        log.warning("Transaction history fetch failed for team %s season %s (%s)", team_mlb_id, season, exc)
        return ()


def replay_il_transactions(transactions, as_of_date: dt.date) -> set[int]:
    """Pure replay logic, split out from `_players_on_il` so it's testable
    against a fabricated transaction list without hitting the network -
    given a season's transactions (any order) and a cutoff date, returns
    the MLB player ids on the IL as of (not including) that date."""
    ordered = sorted(transactions, key=lambda t: t.get("date", ""))

    on_il: set[int] = set()
    for t in ordered:
        t_date_str = t.get("date")
        if not t_date_str or dt.date.fromisoformat(t_date_str) >= as_of_date:
            break
        player_id = (t.get("person") or {}).get("id")
        if player_id is None:
            continue
        description = t.get("description", "")
        if _PLACED_RE.search(description):
            on_il.add(player_id)
        elif _ACTIVATED_RE.search(description):
            on_il.discard(player_id)
    return on_il


def _players_on_il(team_mlb_id: int, as_of_date: dt.date) -> set[int]:
    return replay_il_transactions(_season_transactions(team_mlb_id, as_of_date.year), as_of_date)


def _season_regulars(db: Session, team_id: int, as_of_date: dt.date) -> set[int]:
    """MLB player ids (not our internal players.id) for anyone who's
    started at the plate or on the mound at least MIN_GAMES_TO_COUNT_AS_
    REGULAR times *this season* through as_of_date - distinguishes "a
    regular is hurt" from "some September-callup reliever nobody would
    notice is hurt," which a flat injured-count can't tell apart on its
    own.

    Deliberately season-to-date, not a trailing N-day window: a player
    hurt for the whole trailing window would never show up as "recently
    played" *because* they've been hurt that whole time - a 30-day
    lookback made a real, established regular like a team's starting
    third baseman invisible to this feature the moment he'd been out for
    just over a month. Season-to-date sidesteps that by counting
    appearances however long ago they were, not how recently.
    """
    season_start = dt.date(as_of_date.year, *SEASON_START_MONTH_DAY)

    lineup_counts = db.execute(
        select(Player.mlb_player_id, func.count())
        .join(Lineup, Lineup.player_id == Player.id)
        .join(Game, Game.id == Lineup.game_id)
        .where(Lineup.team_id == team_id, Game.date >= season_start, Game.date < as_of_date)
        .group_by(Player.mlb_player_id)
    ).all()

    starter_counts = db.execute(
        select(Player.mlb_player_id, func.count())
        .join(PitcherGameLog, PitcherGameLog.player_id == Player.id)
        .join(Game, Game.id == PitcherGameLog.game_id)
        .where(Player.team_id == team_id, PitcherGameLog.is_starter.is_(True), Game.date >= season_start, Game.date < as_of_date)
        .group_by(Player.mlb_player_id)
    ).all()

    tallies: dict[int, int] = {}
    for pid, count in (*lineup_counts, *starter_counts):
        if pid is not None:
            tallies[pid] = tallies.get(pid, 0) + count

    return {pid for pid, count in tallies.items() if count >= MIN_GAMES_TO_COUNT_AS_REGULAR}


def compute_injury_features(db: Session, team_id: int, as_of_date: dt.date) -> dict:
    empty = {"injured_count": None, "key_regulars_injured": None}
    team = db.get(Team, team_id)
    if team is None:
        return empty

    injured_ids = _players_on_il(team.mlb_team_id, as_of_date)
    regulars = _season_regulars(db, team_id, as_of_date)

    return {
        "injured_count": len(injured_ids),
        "key_regulars_injured": len(injured_ids & regulars),
    }
