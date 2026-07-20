"""
MLB Stats API ingestion (Section 4.1).

The MLB Stats API (statsapi.mlb.com) is free and requires no API key.
This module has two layers:

- `fetch_*` functions: thin wrappers that hit the API and return parsed
  JSON/dicts. No DB access. Useful standalone (tests, notebooks, the
  scheduler can call these to decide whether there's anything new).
- `ingest_*` / `sync_*` functions: take a SQLAlchemy `Session` and upsert
  the fetched data into our schema. These are idempotent - re-running them
  for the same date/game updates existing rows instead of duplicating them,
  which matters because the scheduler polls the same game repeatedly
  (lineups, live boxscores) throughout the day.
"""
from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import Game, GameStatus, PitcherGameLog, BatterGameLog, Lineup, Player, Team, Venue

log = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
SPORT_ID_MLB = 1
REQUEST_TIMEOUT = 20

_session = requests.Session()
_session.headers.update({"User-Agent": "baseball-predictor/1.0"})


def _get(path: str, params: dict | None = None) -> dict:
    resp = _session.get(f"{BASE_URL}{path}", params=params or {}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# fetch_* : raw API wrappers, no DB access
# --------------------------------------------------------------------------

def fetch_teams() -> list[dict]:
    """All active MLB teams."""
    data = _get("/teams", {"sportId": SPORT_ID_MLB})
    return data.get("teams", [])


def fetch_venue(venue_id: int) -> dict | None:
    data = _get(f"/venues/{venue_id}", {"hydrate": "location,fieldInfo"})
    venues = data.get("venues", [])
    return venues[0] if venues else None


def fetch_person(player_id: int) -> dict | None:
    data = _get(f"/people/{player_id}")
    people = data.get("people", [])
    return people[0] if people else None


def fetch_daily_schedule(date: dt.date) -> list[dict]:
    """Games, start times, venues, probable pitchers for a given date.

    Restricted to `gameType=R` (regular season) - without this, dates
    before Opening Day return spring training/exhibition games (gameType
    'S'/'E') instead of an empty list. Those use split-squad rosters full
    of minor leaguers and would corrupt team win%/rolling-form features if
    ingested as if they were real season games.
    """
    data = _get(
        "/schedule",
        {
            "sportId": SPORT_ID_MLB,
            "date": date.isoformat(),
            "gameType": "R",
            "hydrate": "probablePitcher,linescore,team,venue",
        },
    )
    games: list[dict] = []
    for date_block in data.get("dates", []):
        games.extend(date_block.get("games", []))
    return games


def fetch_probable_pitchers(game_pk: int) -> dict:
    """{'home': {...person...} | None, 'away': {...} | None} for a game."""
    data = _get(f"/schedule", {"sportId": SPORT_ID_MLB, "gamePk": game_pk, "hydrate": "probablePitcher"})
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            if g["gamePk"] == game_pk:
                teams = g.get("teams", {})
                return {
                    "home": teams.get("home", {}).get("probablePitcher"),
                    "away": teams.get("away", {}).get("probablePitcher"),
                }
    return {"home": None, "away": None}


def fetch_boxscore(game_pk: int) -> dict:
    """Post-game (or live) boxscore - player-level batting/pitching lines."""
    return _get(f"/game/{game_pk}/boxscore")


def fetch_linescore(game_pk: int) -> dict | None:
    """Inning-by-inning runs/hits/errors - used to label NRFI/YRFI."""
    data = _get("/schedule", {"sportId": SPORT_ID_MLB, "gamePk": game_pk, "hydrate": "linescore"})
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            if g["gamePk"] == game_pk:
                return g.get("linescore")
    return None


def fetch_confirmed_lineup(game_pk: int) -> dict:
    """Batting order for each team, from the boxscore's `battingOrder` list.

    Not final until close to first pitch - callers should treat an empty
    list as "not confirmed yet" rather than an error.
    """
    box = fetch_boxscore(game_pk)
    result: dict[str, list[dict]] = {"home": [], "away": []}
    for side in ("home", "away"):
        team_box = box.get("teams", {}).get(side, {})
        order = team_box.get("battingOrder", [])
        players = team_box.get("players", {})
        for slot_index, pid in enumerate(order):
            key = f"ID{pid}"
            person = players.get(key, {}).get("person", {})
            result[side].append(
                {
                    "player_id": pid,
                    "name": person.get("fullName"),
                    # MLB's battingOrder encodes lineup slot in the first digit
                    # (e.g. 100 = leadoff, 200 = 2-hole); starters' first 9
                    # entries map 1:1 to batting order position 1-9.
                    "batting_order_position": slot_index + 1,
                }
            )
    return result


def fetch_team_roster(team_id: int, roster_type: str = "active") -> list[dict]:
    data = _get(f"/teams/{team_id}/roster", {"rosterType": roster_type})
    return data.get("roster", [])


def fetch_transactions(team_id: int, start_date: dt.date, end_date: dt.date) -> list[dict]:
    """Roster transactions (trades, call-ups, injured-list moves, etc.)
    for a team in a date range - each has a `person`, a `date`, and a free-
    text `description` (e.g. "placed 3B José Ramírez on the 10-day
    injured list", "activated ... from the 60-day injured list").

    This replaced an earlier, broken attempt at injuries: `rosterType=
    injuredList` looks like a real parameter but isn't one - MLB's own
    `/rosterTypes` meta-endpoint lists only 40Man/fullSeason/fullRoster/
    nonRosterInvitees/active/allTime/depthChart/gameday/coach, so passing
    "injuredList" silently fell back to the full active roster instead of
    raising, and every "injured" player came back with status "Active".
    Transactions are the real source, and - unlike a live roster snapshot -
    they carry a genuine date, so features/injury_features.py can replay
    them to ask "who was on the IL as of this past date," which works for
    historical training data too, not just live predictions.
    """
    data = _get(
        "/transactions",
        {"teamId": team_id, "startDate": start_date.isoformat(), "endDate": end_date.isoformat()},
    )
    return data.get("transactions", [])


# --------------------------------------------------------------------------
# ingest_* / sync_* : write to DB (idempotent upserts)
# --------------------------------------------------------------------------

def _get_or_create_team(db: Session, mlb_team: dict) -> Team:
    team = db.execute(select(Team).where(Team.mlb_team_id == mlb_team["id"])).scalar_one_or_none()
    if team is None:
        team = Team(mlb_team_id=mlb_team["id"])
        db.add(team)
    team.name = mlb_team.get("name", team.name if team.name else "")
    team.abbreviation = mlb_team.get("abbreviation", "")
    team.league = mlb_team.get("league", {}).get("name", "")
    team.division = mlb_team.get("division", {}).get("name", "")
    db.flush()
    return team


def sync_teams(db: Session) -> list[Team]:
    return [_get_or_create_team(db, t) for t in fetch_teams()]


def _get_or_create_venue(db: Session, mlb_venue: dict) -> Venue:
    venue = db.execute(select(Venue).where(Venue.mlb_venue_id == mlb_venue["id"])).scalar_one_or_none()
    if venue is None:
        venue = Venue(mlb_venue_id=mlb_venue["id"])
        db.add(venue)
    venue.name = mlb_venue.get("name", venue.name)
    # Schedule/team payloads only embed {id, name}; hydrate lat/lon/surface
    # lazily on first sight since that's a second API call per venue.
    if venue.lat is None:
        full = fetch_venue(mlb_venue["id"])
        if full:
            location = full.get("location", {}) or {}
            venue.city = location.get("city")
            venue.lat = float(location["defaultCoordinates"]["latitude"]) if location.get("defaultCoordinates") else None
            venue.lon = float(location["defaultCoordinates"]["longitude"]) if location.get("defaultCoordinates") else None
            field_info = full.get("fieldInfo", {}) or {}
            venue.surface_type = field_info.get("turfType")
            venue.roof_type = field_info.get("roofType")
    db.flush()
    return venue


def _get_or_create_player(db: Session, person: dict, team: Team | None = None, hydrate_hand: bool = False) -> Player:
    player = db.execute(select(Player).where(Player.mlb_player_id == person["id"])).scalar_one_or_none()
    if player is None:
        player = Player(mlb_player_id=person["id"])
        db.add(player)
    player.name = person.get("fullName", player.name)
    if team is not None:
        player.team_id = team.id
    if hydrate_hand and player.bats is None:
        full = fetch_person(person["id"])
        if full:
            player.bats = (full.get("batSide") or {}).get("code")
            player.throws = (full.get("pitchHand") or {}).get("code")
            player.position = (full.get("primaryPosition") or {}).get("abbreviation")
    db.flush()
    return player


def ingest_schedule_for_date(db: Session, date: dt.date) -> list[Game]:
    """Upsert games (+ their teams/venues/probable-pitcher players) for a date."""
    games_out: list[Game] = []
    for g in fetch_daily_schedule(date):
        home_raw = g["teams"]["home"]["team"]
        away_raw = g["teams"]["away"]["team"]
        home_team = _get_or_create_team(db, home_raw)
        away_team = _get_or_create_team(db, away_raw)
        venue = _get_or_create_venue(db, g["venue"]) if g.get("venue") else None

        game = db.execute(select(Game).where(Game.mlb_game_id == g["gamePk"])).scalar_one_or_none()
        if game is None:
            game = Game(mlb_game_id=g["gamePk"])
            db.add(game)

        game.date = dt.date.fromisoformat(g["officialDate"])
        game.start_time = dt.datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00"))
        game.home_team_id = home_team.id
        game.away_team_id = away_team.id
        game.venue_id = venue.id if venue else None
        game.status = _map_status(g["status"]["abstractGameState"])
        game.home_score = g["teams"]["home"].get("score")
        game.away_score = g["teams"]["away"].get("score")
        game.is_doubleheader = g.get("doubleHeader", "N") != "N"
        game.game_number_in_series = g.get("gameNumber", 1)
        db.flush()

        for side, team in (("home", home_team), ("away", away_team)):
            probable = g["teams"][side].get("probablePitcher")
            if probable:
                pitcher = _get_or_create_player(db, probable, team=team, hydrate_hand=True)
                setattr(game, f"{side}_starter_id", pitcher.id)
        db.flush()

        games_out.append(game)
    return games_out


def _map_status(abstract_state: str) -> str:
    return {
        "Preview": GameStatus.scheduled.value,
        "Live": GameStatus.live.value,
        "Final": GameStatus.final.value,
    }.get(abstract_state, GameStatus.scheduled.value)


def ingest_confirmed_lineup(db: Session, mlb_game_id: int) -> int:
    """Upsert `lineups` rows for a game. Returns rows written."""
    game = db.execute(select(Game).where(Game.mlb_game_id == mlb_game_id)).scalar_one_or_none()
    if game is None:
        raise ValueError(f"Game {mlb_game_id} not ingested yet - run ingest_schedule_for_date first")

    lineup = fetch_confirmed_lineup(mlb_game_id)
    written = 0
    now = dt.datetime.now(dt.timezone.utc)
    for side, team_id in (("home", game.home_team_id), ("away", game.away_team_id)):
        for slot in lineup[side]:
            player = _get_or_create_player(db, {"id": slot["player_id"], "fullName": slot["name"]})
            row = db.execute(
                select(Lineup).where(Lineup.game_id == game.id, Lineup.player_id == player.id)
            ).scalar_one_or_none()
            if row is None:
                row = Lineup(game_id=game.id, team_id=team_id, player_id=player.id)
                db.add(row)
            row.batting_order_position = slot["batting_order_position"]
            row.confirmed_at = now
            written += 1
    return written


def ingest_boxscore(db: Session, mlb_game_id: int) -> dict:
    """Upsert pitcher_game_logs / batter_game_logs and final score from a boxscore.

    Safe to call for live (in-progress) games too - lines are simply partial
    and get overwritten on the next poll until final.
    """
    game = db.execute(select(Game).where(Game.mlb_game_id == mlb_game_id)).scalar_one_or_none()
    if game is None:
        raise ValueError(f"Game {mlb_game_id} not ingested yet - run ingest_schedule_for_date first")

    box = fetch_boxscore(mlb_game_id)
    counts = {"pitchers": 0, "batters": 0}

    for side, team_id in (("home", game.home_team_id), ("away", game.away_team_id)):
        team_box = box.get("teams", {}).get(side, {})
        players = team_box.get("players", {})
        starter_id = (team_box.get("pitchers") or [None])[0]

        for key, pdata in players.items():
            person = pdata.get("person", {})
            player = _get_or_create_player(db, person, team=None)
            stats = pdata.get("stats", {})

            pitching = stats.get("pitching") or {}
            if pitching:
                row = db.execute(
                    select(PitcherGameLog).where(
                        PitcherGameLog.game_id == game.id, PitcherGameLog.player_id == player.id
                    )
                ).scalar_one_or_none()
                if row is None:
                    row = PitcherGameLog(game_id=game.id, player_id=player.id)
                    db.add(row)
                row.is_starter = person.get("id") == starter_id
                if row.is_starter:
                    setattr(game, f"{side}_starter_id", player.id)
                row.ip = _parse_innings_pitched(pitching.get("inningsPitched"))
                row.er = pitching.get("earnedRuns")
                row.k = pitching.get("strikeOuts")
                row.bb = pitching.get("baseOnBalls")
                row.h = pitching.get("hits")
                row.hr = pitching.get("homeRuns")
                row.pitch_count = pitching.get("numberOfPitches") or pitching.get("pitchesThrown")
                counts["pitchers"] += 1

            batting = stats.get("batting") or {}
            if batting:
                row = db.execute(
                    select(BatterGameLog).where(
                        BatterGameLog.game_id == game.id, BatterGameLog.player_id == player.id
                    )
                ).scalar_one_or_none()
                if row is None:
                    row = BatterGameLog(game_id=game.id, player_id=player.id)
                    db.add(row)
                row.at_bats = batting.get("atBats")
                row.hits = batting.get("hits")
                row.hr = batting.get("homeRuns")
                row.bb = batting.get("baseOnBalls")
                row.k = batting.get("strikeOuts")
                counts["batters"] += 1

    if box.get("teams", {}).get("home", {}).get("teamStats"):
        # Boxscore doesn't carry final linescore runs directly on team block
        # reliably pre-final; leave score updates to ingest_schedule_for_date
        # (schedule endpoint's `score` field), which is authoritative.
        pass

    return counts


def ingest_linescore(db: Session, mlb_game_id: int) -> bool:
    """Write first_inning_home_runs/away_runs onto the game, for NRFI labeling."""
    game = db.execute(select(Game).where(Game.mlb_game_id == mlb_game_id)).scalar_one_or_none()
    if game is None:
        raise ValueError(f"Game {mlb_game_id} not ingested yet - run ingest_schedule_for_date first")

    linescore = fetch_linescore(mlb_game_id)
    if not linescore:
        return False
    innings = linescore.get("innings", [])
    first = next((inn for inn in innings if inn["num"] == 1), None)
    if first is None:
        return False
    game.first_inning_home_runs = first["home"]["runs"]
    game.first_inning_away_runs = first["away"]["runs"]
    return True


def _parse_innings_pitched(ip_str: str | None) -> float | None:
    """MLB reports innings pitched as e.g. "6.1" meaning 6 and 1/3 innings,
    not 6.1 decimal - convert to true decimal (6.333...)."""
    if not ip_str:
        return None
    whole, _, frac = ip_str.partition(".")
    outs = {"0": 0, "1": 1, "2": 2}.get(frac, 0)
    return float(whole) + outs / 3.0
