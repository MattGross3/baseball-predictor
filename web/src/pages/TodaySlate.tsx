import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { Game, GameSlateSummary } from '../api/types'
import { GameRow } from '../components/GameRow'
import { localIsoDate } from '../lib/date'
import { EmptyState, ErrorState, LoadingState } from '../components/States'

export function TodaySlate() {
  const [date, setDate] = useState(() => localIsoDate(new Date()))
  const [games, setGames] = useState<Game[] | null>(null)
  const [summaries, setSummaries] = useState<Record<number, GameSlateSummary>>({})
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const [refreshing, setRefreshing] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null)

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

  return (
    <div>
      <div className="flex items-start justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Today's Slate</h1>
          <p className="text-sm text-[color:var(--color-ink-muted)] mt-1">
            Expected win probability and predicted run total for each game.
          </p>
        </div>
        <div className="flex items-center gap-3">
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

      {refreshMessage && (
        <p className="text-xs text-[color:var(--color-ink-muted)] -mt-4 mb-6">{refreshMessage}</p>
      )}

      {loading && <LoadingState label="Loading today's slate…" />}
      {!loading && error && <ErrorState message={error} />}
      {!loading && !error && games?.length === 0 && (
        <EmptyState message={`No games found for ${date}. Try a date with ingested data.`} />
      )}
      {!loading && !error && games && games.length > 0 && (
        <div className="flex flex-col gap-3">
          {games.map((game) => (
            <GameRow key={game.id} game={game} summary={summaries[game.id]} />
          ))}
        </div>
      )}
    </div>
  )
}
