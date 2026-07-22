import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { Game, GameSlateSummary, HealthConfig } from '../api/types'
import { GameRow } from '../components/GameRow'
import { localIsoDate } from '../lib/date'
import { EmptyState, ErrorState, LoadingState } from '../components/States'

const ODDS_BANNER_DISMISSED_KEY = 'oddsKeyBannerDismissed'

export function TodaySlate() {
  const [date, setDate] = useState(() => localIsoDate(new Date()))
  const [games, setGames] = useState<Game[] | null>(null)
  const [summaries, setSummaries] = useState<Record<number, GameSlateSummary>>({})
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const [healthConfig, setHealthConfig] = useState<HealthConfig | null>(null)
  const [bannerDismissed, setBannerDismissed] = useState(() => localStorage.getItem(ODDS_BANNER_DISMISSED_KEY) === 'true')

  const [refreshing, setRefreshing] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null)

  const [syncing, setSyncing] = useState(false)
  const [syncMessage, setSyncMessage] = useState<string | null>(null)

  const [predicting, setPredicting] = useState(false)
  const [predictMessage, setPredictMessage] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    Promise.all([
      api.gamesToday(date),
      api.getGameSlateSummary(date),
    ])
      .then(([fetchedGames, fetchedSummaries]) => {
        if (cancelled) return
        setGames(fetchedGames)
        setSummaries(Object.fromEntries(fetchedSummaries.map((summary) => [summary.game_id, summary])))
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }, [date])

  useEffect(() => {
    // Only fetches once - which optional API keys are configured doesn't
    // change while the page is open, unlike the date-scoped game data above.
    api.healthConfig().then(setHealthConfig).catch(() => setHealthConfig(null))
  }, [])

  function dismissOddsBanner() {
    localStorage.setItem(ODDS_BANNER_DISMISSED_KEY, 'true')
    setBannerDismissed(true)
  }

  function refreshOdds() {
    setRefreshing(true)
    setRefreshMessage(null)

    api
      .refreshOdds()
      .then((result) => {
        setRefreshMessage(`${result.message} (${result.calls_remaining} odds API calls left this month)`)
        // Odds moved - re-pull the slate summary so Spread/Total/ML reflect
        // the fresh snapshot instead of showing stale prices until the
        // next date/page navigation.
        return api.getGameSlateSummary(date)
      })
      .then((fetchedSummaries) => {
        setSummaries(Object.fromEntries(fetchedSummaries.map((summary) => [summary.game_id, summary])))
      })
      .catch((err: unknown) => {
        setRefreshMessage(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => setRefreshing(false))
  }

  function syncGames() {
    setSyncing(true)
    setSyncMessage(null)

    api
      .syncGames()
      .then((result) => {
        setSyncMessage(result.message)
        // New games may now exist for the currently-viewed date (e.g. a
        // future day whose schedule just got pulled in for the first
        // time) - re-fetch rather than requiring a manual date-picker
        // nudge to see them appear.
        return Promise.all([api.gamesToday(date), api.getGameSlateSummary(date)])
      })
      .then(([fetchedGames, fetchedSummaries]) => {
        setGames(fetchedGames)
        setSummaries(Object.fromEntries(fetchedSummaries.map((summary) => [summary.game_id, summary])))
      })
      .catch((err: unknown) => {
        setSyncMessage(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => setSyncing(false))
  }

  function generatePredictions() {
    setPredicting(true)
    setPredictMessage(null)

    api
      .generatePredictions(date)
      .then((result) => {
        setPredictMessage(result.message)
        return api.getGameSlateSummary(date)
      })
      .then((fetchedSummaries) => {
        setSummaries(Object.fromEntries(fetchedSummaries.map((summary) => [summary.game_id, summary])))
      })
      .catch((err: unknown) => {
        setPredictMessage(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => setPredicting(false))
  }

  return (
    <div>
      <div className="flex items-start justify-between mb-6 flex-wrap gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-home)] mb-1">Today</div>
          <h1 className="text-3xl font-bold tracking-tight">Today's Slate</h1>
          <p className="text-sm text-[color:var(--color-ink-muted)] mt-1">
            Expected win probability and predicted run total for each game.
          </p>
          <p className="text-xs text-[color:var(--color-ink-faint)] mt-1">
            NRFI calls only show at 60%+ confidence either way - the model validates at roughly a coin flip below that.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={generatePredictions}
            disabled={predicting}
            className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-3 py-2 text-sm font-medium hover:border-[color:var(--color-ink-muted)] transition-colors disabled:opacity-50"
          >
            {predicting ? 'Predicting…' : 'Make predictions'}
          </button>
          <button
            onClick={syncGames}
            disabled={syncing}
            className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-3 py-2 text-sm font-medium hover:border-[color:var(--color-ink-muted)] transition-colors disabled:opacity-50"
          >
            {syncing ? 'Syncing games…' : 'Sync games'}
          </button>
          <button
            onClick={refreshOdds}
            disabled={refreshing}
            className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-3 py-2 text-sm font-medium hover:border-[color:var(--color-ink-muted)] transition-colors disabled:opacity-50"
          >
            {refreshing ? 'Refreshing odds…' : 'Refresh odds'}
          </button>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
          />
        </div>
      </div>

      {healthConfig && !healthConfig.odds_api_key_configured && !bannerDismissed && (
        <div className="flex items-start justify-between gap-3 rounded-lg border border-[color:var(--color-warning)] bg-[color:var(--color-warning-soft)] px-4 py-3 mb-6 text-sm">
          <p className="text-[color:var(--color-ink)]">
            No <code className="text-xs">ODDS_API_KEY</code> is configured, so odds columns (moneyline price, run line, total
            line, O/U prices, edge vs. market) are blank on every game below by design, not because something's broken - see
            the README's "API keys" section to add one.
          </p>
          <button
            onClick={dismissOddsBanner}
            className="text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)] text-xs font-medium shrink-0"
          >
            Dismiss
          </button>
        </div>
      )}

      {(predictMessage || syncMessage || refreshMessage) && (
        <div className="-mt-4 mb-6 space-y-1">
          {predictMessage && <p className="text-xs text-[color:var(--color-ink-muted)]">{predictMessage}</p>}
          {syncMessage && <p className="text-xs text-[color:var(--color-ink-muted)]">{syncMessage}</p>}
          {refreshMessage && <p className="text-xs text-[color:var(--color-ink-muted)]">{refreshMessage}</p>}
        </div>
      )}

      {loading && <LoadingState label="Loading today's slate…" />}
      {!loading && error && <ErrorState message={error} />}
      {!loading && !error && games?.length === 0 && (
        <EmptyState message={`No games found for ${date}. Try a date with ingested data.`} />
      )}
      {!loading && !error && games && games.length > 0 && (
        <div className="flex flex-col gap-3">
          {games.map((game) => (
            <GameRow
              key={game.id}
              game={game}
              summary={summaries[game.id]}
              oddsKeyConfigured={healthConfig?.odds_api_key_configured ?? true}
            />
          ))}
        </div>
      )}
    </div>
  )
}
