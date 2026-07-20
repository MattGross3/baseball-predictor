import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { Game, Prediction } from '../api/types'
import { GameRow } from '../components/GameRow'
import { localIsoDate } from '../lib/date'
import { EmptyState, ErrorState, LoadingState } from '../components/States'

export function TodaySlate() {
  const [date, setDate] = useState(() => localIsoDate(new Date()))
  const [games, setGames] = useState<Game[] | null>(null)
  const [predictionsByGame, setPredictionsByGame] = useState<Record<number, Prediction[]>>({})
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    api
      .gamesToday(date)
      .then(async (fetchedGames) => {
        if (cancelled) return
        setGames(fetchedGames)

        const entries = await Promise.all(
          fetchedGames.map(async (g) => {
            try {
              const res = await api.getGamePredictions(g.id)
              return [g.id, res.predictions] as const
            } catch {
              return [g.id, []] as const
            }
          }),
        )
        if (!cancelled) setPredictionsByGame(Object.fromEntries(entries))
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

  return (
    <div>
      <div className="flex items-start justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Today's Slate</h1>
          <p className="text-sm text-[color:var(--color-ink-muted)] mt-1">
            Expected win probability and predicted run total for each game.
          </p>
        </div>
        <input
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
        />
      </div>

      {loading && <LoadingState label="Loading today's slate…" />}
      {!loading && error && <ErrorState message={error} />}
      {!loading && !error && games?.length === 0 && (
        <EmptyState message={`No games found for ${date}. Try a date with ingested data.`} />
      )}
      {!loading && !error && games && games.length > 0 && (
        <div className="flex flex-col gap-3">
          {games.map((game) => (
            <GameRow key={game.id} game={game} predictions={predictionsByGame[game.id] ?? []} />
          ))}
        </div>
      )}
    </div>
  )
}
