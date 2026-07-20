# Baseball Predictor

An MLB prediction platform: it ingests daily game/player data, engineers
features, trains calibrated models for moneyline / run totals / NRFI /
player props, serves predictions over an API, and displays them on a
Streamlit dashboard with a backtesting framework to check the models
against actual results. Built from [`baseball_prediction_app_spec.md`](../baseball_prediction_app_spec.md)
in the parent directory, following that document's own suggested build
order (its Section 13).

**Status:** every layer in the spec is implemented and has been run
end-to-end against real MLB data (see [Verified against real data](#verified-against-real-data)
below) - not just scaffolded. The one honest caveat: models are only as
good as the data behind them, and this repo ships with a small ~4-week
demo backfill for verification purposes, not a full season. See
[On predictive power](#on-predictive-power-read-this-before-trusting-a-number).

---

## Contents

- [How it fits together](#how-it-fits-together)
- [Data sources - what's live, what needs a key, what's blocked](#data-sources---whats-live-what-needs-a-key-whats-blocked)
- [Quickstart](#quickstart)
- [Configuration](#configuration-env)
- [Day-to-day commands](#day-to-day-commands)
- [Database schema](#database-schema)
- [Feature engineering](#feature-engineering)
- [Models](#models)
- [API reference](#api-reference)
- [Dashboard](#dashboard)
- [Scheduler](#scheduler)
- [Backtesting](#backtesting)
- [Docker](#docker)
- [Testing](#testing)
- [Known limitations & where the schema was extended](#known-limitations--where-the-schema-was-extended)
- [On predictive power](#on-predictive-power-read-this-before-trusting-a-number)
- [Verified against real data](#verified-against-real-data)

---

## How it fits together

```
MLB Stats API ─┐
Baseball Savant ├─► ingestion/ ─► Postgres (database/) ─► features/ ─► models/ ─► predictions table
The Odds API   ─┤                                              │            │
OpenWeatherMap ─┘                                        backtest/ ◄────────┘
                                                                 │
                                              api/ (FastAPI) ◄───┘
                                                 │
                                        frontend/ (Streamlit)

scheduler/daily_jobs.py ties ingestion → features → models → predictions
together on a schedule; scripts/backfill_data.py does the same thing for
a historical date range in one shot (what you run first).
```

| Directory | What's in it |
|---|---|
| `ingestion/` | Pulls from each external data source, upserts into Postgres. One module per source (Section 4). |
| `database/` | SQLAlchemy models + Alembic migrations (Section 5). |
| `features/` | Turns raw ingested data into per-game feature rows, "as of" a date, never leaking future data (Section 6). |
| `models/` | Trains/calibrates the moneyline, totals, NRFI, and player-prop models; the registry of trained artifacts (Section 7). |
| `backtest/` | Scores a trained model against a date range: accuracy, log-loss, Brier score, ROI, CLV (Section 11). |
| `api/` | FastAPI app exposing games, features, predictions, odds, and backtest results (Section 8). |
| `web/` | React + TypeScript dashboard (Section 9 v2) - the primary UI. |
| `frontend/` | Streamlit dashboard (Section 9 v1) - still works, superseded by `web/`. |
| `scheduler/` | The daily job loop - what actually keeps the data current in production (Section 10). |
| `scripts/` | One-off operational scripts (currently: historical backfill). |
| `tests/` | Unit tests for the pure-logic pieces (FIP calc, wOBA proxy, date-split, odds math, etc). |

---

## Data sources - what's live, what needs a key, what's blocked

| Source | Section | Status | Notes |
|---|---|---|---|
| MLB Stats API | 4.1 | ✅ Live, free, no key | Schedule, boxscores, lineups, rosters, injuries, umpire assignments (via boxscore `officials`), linescores. |
| Baseball Savant / Statcast | 4.2 | ✅ Live, free, no key | Via `pybaseball`. Pitch-level velo/spin/whiff, batted-ball exit velo/barrel%. Also team-level Outs Above Average (Savant's leaderboard CSV export isn't blocked, unlike FanGraphs'). |
| FanGraphs | 4.3 | ⚠️ Blocked upstream | FanGraphs currently returns HTTP 403 to `pybaseball`'s scraper (their own anti-bot measure, not a bug here). `ingestion/fangraphs.py` fails soft and logs a warning; `features/pitcher_features.py` falls back to computing FIP itself from boxscore components (`ingestion/fangraphs.estimate_fip`). SIERA/xFIP/wOBA/wRC+/projections have no fallback and come back `None`. |
| The Odds API | 4.4 | 🔑 Needs `ODDS_API_KEY` | Free tier: 500 req/month. Without a key, odds ingestion no-ops; edge-vs-market and ROI/CLV fields come back `null`/`N/A` everywhere instead of erroring. |
| OpenWeatherMap | 4.5 | 🔑 Needs `WEATHER_API_KEY` | Free tier only covers current conditions + 5-day forecast, so historical weather (for backtesting older games) isn't available even with a key - those rows stay `null`. |
| Umpire assignments | 4.6 | ✅ Live, free, no key | Turns out to be in the MLB Stats API boxscore (`officials`), no separate source needed. |
| Umpire zone history | 4.6 | ✅ Live, computed | Built from our own DB (which umpire worked which game) joined to Statcast pitch calls - see `ingestion/umpire_scorecards.py`. |
| Park factors | 4.7 | ✅ Static, seeded | `ingestion/reference_data/park_factors.csv`, checked into the repo. FanGraphs' and Savant's park-factor *pages* both block scraping, so this is hand-seeded from public multi-year figures - refresh it once a year by hand. |

Nothing in the app hard-fails because a key is missing. Every ingestion
module degrades to `None`/empty/logged-warning, and every feature/model
built on top of it treats that as "missing", not "crash".

---

## Quickstart (Windows / PowerShell)

Prerequisites: Python 3.11+ and PostgreSQL. These steps match exactly what
was run to build and verify this repo, using a local PostgreSQL install
(you have PostgreSQL 18 at `C:\Program Files\PostgreSQL\18\bin`) rather
than Docker - use the [Docker](#docker) section instead if you'd rather
run Postgres in a container.

```powershell
# 1. Virtual environment + dependencies (run from the baseball-predictor/ folder)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Postgres role + database
# The `psql` client isn't necessarily on PATH on Windows - call it by full
# path. This creates a login role `baseball`/`baseball` and a database it
# owns, matching the defaults already in .env.example.
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "DO `$`$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'baseball') THEN CREATE ROLE baseball LOGIN PASSWORD 'baseball'; END IF; END `$`$;"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "CREATE DATABASE baseball OWNER baseball;"
# ("database ... already exists" on the second command just means you (or
# a previous run) already did this step - safe to ignore.)
# If that first command prompts for a password you don't know, PostgreSQL's
# Windows installer sets one for the `postgres` superuser at install time -
# or open pgAdmin (installed alongside PostgreSQL) and run the same two
# statements from its Query Tool instead.

# 3. Config
Copy-Item .env.example .env
# .env works as-is against the database from step 2. ODDS_API_KEY /
# WEATHER_API_KEY are optional - see "API keys" below for what they add
# and where to get them.

# 4. Schema
alembic upgrade head

# 5. Backfill some real history (this is what makes every rolling
#    feature - bullpen ERA, season win%, etc - non-null; without it the
#    app runs but every feature is None for lack of history). This
#    example pulls one month of the 2025 season and takes a couple of
#    minutes; widen the range for a more useful model - see
#    "On predictive power" below.
python -m scripts.backfill_data 2025-04-01 2025-04-29

# 6. Train the models on that history
python -m models.train_moneyline 2025-04-01 2025-04-24 2025-04-29
python -m models.train_totals    2025-04-01 2025-04-24 2025-04-29
python -m models.train_nrfi      2025-04-01 2025-04-24 2025-04-29

# 7. Run the API (leave this terminal open)
uvicorn api.main:app --reload

# 8. Run the dashboard - open a second PowerShell window, activate the
#    same venv (.\venv\Scripts\Activate.ps1), then:
streamlit run frontend/app.py
```

Open http://localhost:8501 for the dashboard, http://localhost:8000/docs
for the interactive API docs.

**macOS/Linux/Git Bash equivalents**, for the two steps that differ:
step 1's activation is `source venv/bin/activate` (macOS/Linux) or
`source venv/Scripts/activate` (Git Bash on Windows); step 3 is
`cp .env.example .env`. Everything else is identical - psql's `DO $$ ...
$$;` block doesn't need the backtick-escaping PowerShell requires for `$`.

### API keys - what you're missing and whether it matters

You don't have either of these yet, and **the app runs correctly without
them** - every feature/endpoint that would use them degrades to `null`/
`N/A` instead of failing. Add them to `.env` any time; no restart-order
dependency, no other config changes needed.

| Key | Get it at | Free tier | Unlocks |
|---|---|---|---|
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com/) - sign up, key's on your dashboard immediately | 500 requests/month | Live moneyline/run-line/total odds, the edge-vs-market number on every game card and in Game Detail, ROI and CLV numbers in Backtest. |
| `WEATHER_API_KEY` | [openweathermap.org/api](https://openweathermap.org/api) - sign up, key's on your account page (can take up to ~1hr to activate) | Yes, generous | Real temp/wind for upcoming games' park/weather features. (Historical weather for old games isn't available even with a key - free tier only covers current + 5-day forecast.) |

Neither is required to reach any part of the app - dashboard, API,
training, backtesting all work today. They just make the odds/weather
columns real instead of blank.

---

## Configuration (`.env`)

All settings load through `config.py` (`pydantic-settings`), reading
`.env`. See `.env.example` for the full list with inline comments. The
short version:

| Variable | Required? | Purpose |
|---|---|---|
| `DATABASE_URL` | Yes | Postgres connection string (native run). |
| `POSTGRES_USER` / `PASSWORD` / `DB` | Docker only | Feeds the `postgres` container + `DATABASE_URL` inside Compose. |
| `ODDS_API_KEY` | No | Enables live odds polling, edge-vs-market, ROI/CLV. |
| `WEATHER_API_KEY` | No | Enables live weather features for upcoming games. |
| `ADMIN_API_KEY` | No | Shared secret for `POST /models/retrain`. Blank = open in dev. |
| `API_BASE_URL` | No | Where `frontend/` looks for the API. Defaults to `localhost:8000`; Compose overrides it to `http://api:8000`. |
| `TIMEZONE` | No | Used by the scheduler's cron jobs (default `America/New_York`, i.e. ET - matches the spec's job schedule). |

---

## Day-to-day commands

These all assume the venv is activated (`.\venv\Scripts\Activate.ps1` on
Windows) and you're in the `baseball-predictor/` folder.

```powershell
# Backfill / catch-up ingestion for a date range (idempotent - safe to re-run)
python -m scripts.backfill_data 2025-05-01 2025-06-01

# Retrain a model over a new date range
python -m models.train_moneyline 2025-04-01 2025-05-15 2025-06-01

# New Alembic migration after changing database/models.py
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Generating a single prediction (writes to the `predictions` table) is a
few lines of Python rather than a one-liner - save this as e.g.
`scratch_predict.py` in `baseball-predictor/` and run `python
scratch_predict.py` (or just paste it into a `python` REPL):

```python
from database.db import session_scope
from models.predict import generate_prediction

with session_scope() as db:
    print(generate_prediction(db, game_id=1, target="moneyline"))
```

`game_id` is the *internal* id (the `id` column, e.g. from
`GET /games/today`), not MLB's own `gamePk`.

---

## Database schema

Implements Section 5's schema (`database/models.py`) with two deliberate
extensions the spec's table definitions didn't cover but the model layer
needs - see [Known limitations](#known-limitations--where-the-schema-was-extended).
Every table upserts on the source system's own id (`mlb_game_id`,
`mlb_team_id`, `mlb_player_id`, `mlb_venue_id`), so re-running ingestion
for a date you've already loaded updates rows instead of duplicating them.

---

## Feature engineering

`features/build_feature_matrix.py` is the entry point:

- `build_game_feature_row(db, game_id)` - full nested feature dict for one
  game (what the API's `/games/{id}/features` and the dashboard's feature
  breakdown tab show).
- `build_training_matrix(db, start_date, end_date, target)` - one flat row
  per completed game plus home-minus-away differential features, for
  `target` in `moneyline` / `total` / `nrfi`.

**Leakage discipline**: every feature is computed strictly "as of" the
game's date - rolling stats, season stats, everything - using
`Game.date < as_of_date` in every query. Train/test splits are always by
date (`models/model_utils.date_split`), never a random shuffle, per the
spec's explicit warning in Section 11.

**Performance note**: two features (`velo_trend_last_3`, and umpire
`strike_zone_size_percentile`/`over_under_lean`/`k_rate_boost`) require
live Statcast network pulls - cheap for one live prediction, far too slow
to repeat across hundreds of historical games. `build_training_matrix`
passes `include_statcast_trend=False` to skip them during bulk training
(those columns are `None` in the training matrix); live single-game
prediction (`models/predict.py`) leaves them on. This alone took a
training run for one month of data from "didn't finish in 10 minutes" to
"finishes in about 20 seconds" - see the umpire zone-history docstring in
`ingestion/umpire_scorecards.py` for why (it was pulling a full-league
Statcast date range per game before the fix).

---

## Models

| Target | Baseline | Production | File |
|---|---|---|---|
| Moneyline | `LogisticRegression` (calibrated, isotonic) | `XGBClassifier` w/ early stopping (calibrated) | `models/train_moneyline.py` |
| Run total | Per-side Poisson GLM, convolved into a full distribution | `XGBRegressor` + Negative Binomial variance estimate | `models/train_totals.py` |
| NRFI/YRFI | `LogisticRegression` (calibrated) | `XGBClassifier`, kept only if it beats logistic by real margin | `models/train_nrfi.py` |
| Player props (stretch) | - | `XGBClassifier` (HR), `XGBRegressor` (hits, pitcher Ks) | `models/train_props.py` |

Every classifier is wrapped in `CalibratedClassifierCV` (isotonic) -
required for comparing predicted probability to the market's implied
probability, which is the whole point of the edge-vs-market feature.
`models/calibration.py` has reliability-diagram and rolling-Brier-score
helpers the dashboard's Backtest page uses.

Trained artifacts are pickled to `models/registry/*.pkl` (gitignored -
they're binary and reproducible from `train_*.py`) with a matching row in
the `model_registry` table (name, version, metrics, file path).
`models/predict.py` picks the latest XGBoost model per target if one's
trained, else falls back to the baseline (NRFI defaults to logistic
either way, per the spec's own rule).

---

## API reference

FastAPI app (`api/main.py`), routers under `api/routers/`:

```
GET  /games/today?date=            Today's (or any date's) slate
GET  /games/{id}                   Game detail
GET  /games/{id}/features          Full nested feature breakdown
GET  /games/{id}/predictions       Predictions + edge vs. market
GET  /games/{id}/odds              Full odds-snapshot history (line movement)
GET  /predictions/history?date_range=&target_type=
GET  /backtest/results?model=&date_range=
POST /models/retrain               Admin-only (X-Admin-Key header), backgrounded
GET  /health
```

Interactive docs at `/docs` once the server's running.

---

## Dashboard

Two dashboards, both talking to the FastAPI backend only (never touching
the DB directly) - Section 9's spec calls for exactly this v1-then-v2
progression:

### `web/` - React + TypeScript (primary)

Vite + React 19 + TypeScript + Tailwind v4 + Recharts. This is the one to
actually use day to day.

```bash
cd web
npm install
npm run dev
```

Opens on http://localhost:5173. In dev, Vite proxies `/api/*` to the
FastAPI backend on `:8000` (`web/vite.config.ts`) - the browser never talks
to the backend directly, so there's no CORS to configure and no backend
URL baked into the JS bundle. Production (`web/Dockerfile` +
`web/nginx.conf`) does the same proxying trick with nginx instead.

Pages:

- **Today's Slate** (`/`) - every game for a date: expected win % for both
  teams, predicted total, NRFI %.
- **Game Detail** (`/games/:id`) - the same metrics plus every stored
  prediction (one row per trained model, not just the one Today's Slate
  displays) and the full per-side feature breakdown.
- **Previous Games** (`/previous-games`) - final games from the last
  3/7/14 days next to what the model predicted going in, with a
  Correct/Missed call on the winner and the total-runs miss distance.
- **Backtest** (`/backtest`) - accuracy/log-loss/Brier/MAE/ROI/CLV for a
  model over a date range; the weekly trend is a separate opt-in "Load
  weekly trend" action, not automatic (see the performance note below).
- **Model Comparison** (`/compare`) - baseline vs. XGBoost side by side,
  plus a real 50/50 blend of both models' stored predictions scored
  against actual outcomes.

**No odds/weather in the UI, by design.** Early versions showed an
"edge vs. market" column and a park/weather block - both always empty
without `ODDS_API_KEY`/`WEATHER_API_KEY`, which is dead weight for anyone
running this without those. The UI leads with the model's own expected win
probability instead (always available, since it only needs a trained
model); the backend odds/weather ingestion and the `/games/{id}/odds` API
endpoint are untouched and start working the moment you add a key - the
frontend change is purely presentational.

**Every prediction is stored once per model family, not recomputed on
view.** `GET /games/{id}/predictions` is a pure DB read - nothing in the
dashboard triggers a live model run. `models/predict.py` upserts on
`(game_id, target_type, model_name)`: re-running prediction generation for
a game updates that model's existing row instead of appending a duplicate,
but *different* model families (`moneyline_logistic` vs
`moneyline_xgboost`) intentionally get separate rows - Model Comparison's
blend needs both at once. The frontend picks the "headline" model per
target the same way the backend does (`web/src/lib/predictions.ts` mirrors
`models/predict.py`'s `PREFERRED_MODEL_BY_TARGET`), so Today's Slate always
shows the production (XGBoost) model's number even though Game Detail's
predictions table lists every model.

**Performance note**: `/backtest/results` rebuilds the full per-game
feature set from scratch on every call (no caching - see
[Feature engineering](#feature-engineering)'s leakage-discipline note), so
it's a genuinely multi-second call, not an instant lookup. The Backtest and
Model Comparison pages default to a 7-day range and load the (much slower)
weekly trend chart as a separate, explicit action rather than blocking the
initial page render - widen the range deliberately, not as something you
wait on by default.

Verified with a headless-Chromium smoke test (`web/verify.mjs`, via
Playwright) against the live API with real data - all five pages/routes
load with zero console errors; screenshots are gitignored (dev artifacts,
not part of the app).

### `frontend/` - Streamlit (secondary)

Still working, still useful for quick ad-hoc inspection, but superseded by
`web/` as the dashboard end users should open:

```bash
streamlit run frontend/app.py
```

- **Today's Slate** (`app.py`), **Game Detail**, **Backtest**, **Model
  Comparison** (`pages/`) - same four views `web/` has, built first as the
  Section 9 v1 proof of concept per the spec's suggested build order.

---

## Scheduler

`scheduler/daily_jobs.py`, run as `python -m scheduler.daily_jobs`
(long-lived process; the `scheduler` Docker service does this). Jobs, per
Section 10:

| Job | Cadence | What it does |
|---|---|---|
| `job_morning_schedule` | 06:00 (configured timezone) | Ingest today's schedule + probable pitchers |
| `job_poll_lineups` | every 30 min | Confirmed lineups for games within 3h of first pitch |
| `job_poll_odds` | every 15 min | Line movement (no-ops without `ODDS_API_KEY`) |
| `job_pregame_predictions` | every 10 min | Generates predictions for games 50-70 min from first pitch |
| `job_postgame_results` | every 20 min | Boxscores/linescores/umpires for newly-finished games |
| `job_nightly_retrain_check` | 02:00 | Retrains only if it's been ≥7 days since the last run per target - the spec's explicit "weekly not daily, to avoid overfitting to noise" rule |

The two "N hours pre-game" jobs are implemented as a short fixed-interval
scan of today's games rather than one dynamically-scheduled job per game -
simpler, self-healing if a run is missed, and idempotent. See the module
docstring for the reasoning.

---

## Backtesting

`backtest/backtest_engine.run_backtest(db, model_name, start_date, end_date)`
loads the latest registered version of `model_name`, scores it against
every completed game in the range, and returns accuracy/log-loss/Brier
score (classification targets) or MAE/RMSE (totals), plus - for moneyline
- simulated flat-bet and Kelly-bet ROI and average CLV, when odds data
exists for those games.

**Always backtest a model against a date range it wasn't trained on.**
The function doesn't enforce this for you (it doesn't know what a model
was trained on beyond what's in the registry's `trained_at`) - that
discipline is on the caller, same as the spec's Section 11 warns.

`backtest/clv_tracker.py` computes closing-line value per bet: how much
better (or worse) your price was than the closing line. Needs ≥2 odds
snapshots per game to mean anything, which needs `ODDS_API_KEY` configured
while those games were still upcoming (CLV can't be computed
retroactively for games that already happened without a key at the time).

---

## Docker

```bash
docker compose up --build
```

Services: `postgres`, a one-shot `migrate` (runs `alembic upgrade head`,
everything else waits on it via `service_completed_successfully`), `api`
(:8000), `scheduler`, `frontend` - Streamlit (:8501), `web` - React, built
via `web/Dockerfile`'s multi-stage node-build-then-nginx (:3000). Secrets
come from `.env` (gitignored) via `env_file:` - never baked into the
image; `web` doesn't take an `env_file:` at all since it has no secrets to
receive (see the "no odds/weather" note in [Dashboard](#dashboard)).

> **Not verified in this build session** - Docker Desktop wasn't running
> in the sandbox this was built in, so `docker compose up` itself hasn't
> been exercised end-to-end here, only reviewed. Everything it runs
> (`alembic upgrade head`, `uvicorn`, `streamlit run`, `python -m
> scheduler.daily_jobs`, `npm run build` + nginx) has been run and
> verified natively/via `npm run dev`. Run it yourself and open an issue
> in your own tracking if something in the Compose config doesn't match
> the native behavior.

---

## Testing

```bash
pytest tests/ -v
```

Covers the pure-logic pieces that don't need a live DB/network: FIP
calculation, innings-pitched parsing (MLB's `"6.1"` → 6⅓, not 6.1),
wOBA-proxy math, umpire-lean z-scores, wind-direction sign, date-based
train/test splitting, American-odds-to-implied-probability conversion,
classification/regression metrics. Everything that *does* need a live DB
or network (ingestion, feature building, training, the API, the
dashboard) was exercised manually against the real backfilled dataset
during development - see [Verified against real data](#verified-against-real-data).

---

## Known limitations & where the schema was extended

The spec's Section 5 schema is thorough but two gaps became apparent
building the model layer on top of it, and got fixed rather than worked
around:

1. **`games` had no pointer to the starting pitcher.** Section 6/7's
   feature and model contracts need to know who's starting to pull their
   stats. Added `home_starter_id` / `away_starter_id`, populated from the
   probable pitcher at schedule-ingest time and overwritten with the
   confirmed starter once the boxscore is available (probables change
   ~10-15% of the time before first pitch - this really happened during
   the demo backfill, see `ingestion/mlb_stats_api.py`'s docstring).
2. **No inning-level data for NRFI labeling.** Added
   `first_inning_home_runs` / `first_inning_away_runs`, populated from the
   schedule endpoint's linescore hydrate.

Other honest simplifications, all documented inline where they live:

- **wOBA is a proxy, not FanGraphs' real wOBA.** The spec's own
  `batter_game_logs` schema only stores AB/H/HR/BB/K - no 2B/3B/HBP
  breakdown - so true linear-weights wOBA isn't computable from it.
  `features/batter_features._woba_proxy` uses a simplified weighting.
  Good for ranking hitters against each other within this app; won't
  match FanGraphs' number.
- **FIP is self-computed**, not FanGraphs', when FanGraphs is unreachable
  (currently always - see the data-sources table). SIERA has no
  no-FanGraphs fallback and is `None`.
- **NRFI features reuse the general season-level features** (era_season,
  lineup wOBA, park factors) rather than true first-inning-specific
  splits (leadoff OBP, starter's first-inning ERA specifically), since
  that needs play-by-play parsing not built in this pass.
- **Wind "blowing out" direction is a heuristic**, not per-park azimuth
  (Savant's venue data has an `azimuthAngle` that would make this exact -
  noted in `features/park_weather_features.py` as a follow-up).
- **Player props (Section 7.4)** are the spec's own stated stretch goal
  and are correspondingly lighter: three targets, simpler features, no
  calibration/backtest wiring like the other three targets have.

---

## On predictive power (read this before trusting a number)

This app's pipeline is real and was verified against real 2025 MLB data
end to end. Its **models are not**, yet, meaningfully predictive - and
that's expected, not a bug: they were trained on ~4 weeks of backfilled
data (299 training games) purely to prove the pipeline works. Held-out
test accuracy came back close to a coin flip (moneyline XGBoost: 58% on
62 held-out games; NRFI: ~48-50%), which is exactly what you'd expect from
one month of data with mostly-thin rolling-window history at the start of
that window.

To get a model actually worth trusting: backfill a full season or more
(`python -m scripts.backfill_data`, just with a much wider date range),
retrain, and re-check the backtest - accuracy/log-loss/Brier score on a
genuinely held-out date range, never the training range. The
[Backtesting](#backtesting) section above is how you'd check that
honestly rather than take a training-set number at face value.

---

## Verified against real data

Concretely, during this build:

- Ingested real 2025-season MLB games (April 1-29, 361 games, 0 errors)
  via the live MLB Stats API - teams, venues (with real lat/lon from MLB's
  own geocoding), players, box scores, lineups, linescores, umpire
  assignments.
- Pulled real Statcast pitch-level data (velocity, spin rate, exit velo,
  barrel%) and real Baseball Savant team defensive OAA for actual
  players/teams.
- Seeded and matched real park factors to 15+ real venues.
- Built the full nested feature row for real games end to end.
- Trained all six moneyline/totals/NRFI baseline+production models on the
  real backfilled data and got real (if unimpressive, see above) held-out
  metrics.
- Generated a real prediction for a real game and served it through every
  API endpoint (`/games/today`, `/games/{id}`, `/features`,
  `/predictions`, `/odds`, `/predictions/history`, `/backtest/results`,
  `/models/retrain`).
- Loaded all four Streamlit pages against the live API with Streamlit's
  `AppTest` harness and confirmed zero exceptions on real data.
- Ran the full `pytest` suite (28 tests, all passing).
