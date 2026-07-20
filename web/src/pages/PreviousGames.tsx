import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Game, Prediction } from '../api/types'
import { TeamBadge } from '../components/TeamBadge'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { localIsoDate } from '../lib/date'
import { preferredPrediction } from '../lib/predictions'

interface Row {
  game: Game
  moneyline: Prediction | undefined
  total: Prediction | undefined
  winnerCorrect: boolean | null
  totalDiff: number | null
}

export function PreviousGames() {
  const [days, setDays] = useState(7)
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    async function run() {
      try {
        const dates: string[] = []
        for (let i = 1; i <= days; i++) {
          const d = new Date()
          d.setDate(d.getDate() - i)
          dates.push(localIsoDate(d))
        }

        const gamesByDate = await Promise.all(dates.map((d) => api.gamesToday(d)))
        const finalGames = gamesByDate.flat().filter((g) => g.status === 'final' && g.home_score != null && g.away_score != null)

        const withPredictions = await Promise.all(
          finalGames.map(async (game) => {
            const { predictions } = await api.getGamePredictions(game.id).catch(() => ({ predictions: [] as Prediction[] }))
            const moneyline = preferredPrediction(predictions, 'moneyline')
            const total = preferredPrediction(predictions, 'total')

            let winnerCorrect: boolean | null = null
            if (moneyline?.predicted_probability != null && game.home_score != null && game.away_score != null) {
              const predictedHomeWin = moneyline.predicted_probability >= 0.5
              const actualHomeWin = game.home_score > game.away_score
              winnerCorrect = predictedHomeWin === actualHomeWin
            }

            let totalDiff: number | null = null
            if (total?.predicted_value != null && game.home_score != null && game.away_score != null) {
              totalDiff = total.predicted_value - (game.home_score + game.away_score)
            }

            return { game, moneyline, total, winnerCorrect, totalDiff }
          }),
        )

        if (!cancelled) {
          withPredictions.sort((a, b) => b.game.date.localeCompare(a.game.date))
          setRows(withPredictions.filter((r) => r.moneyline || r.total))
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load previous games.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    run()
    return () => {
      cancelled = true
    }
  }, [days])

  return (
    <div>
      <div className="flex items-start justify-between mb-6 flex-wrap gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Previous Games</h1>
          <p className="text-sm text-[color:var(--color-ink-muted)] mt-1">Final results vs. what the model predicted going in.</p>
        </div>
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
        >
          <option value={3}>Last 3 days</option>
          <option value={7}>Last 7 days</option>
          <option value={14}>Last 14 days</option>
        </select>
      </div>

      {loading && <LoadingState label="Loading recent results…" />}
      {!loading && error && <ErrorState message={error} />}
      {!loading && !error && rows.length === 0 && (
        <EmptyState message="No predicted-and-final games in this range yet." />
      )}

      {!loading && !error && rows.length > 0 && (
        <div className="flex flex-col gap-3">
          {rows.map(({ game, moneyline, total, winnerCorrect, totalDiff }) => (
            <Link
              key={game.id}
              to={`/games/${game.id}`}
              className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-6 py-4 flex flex-wrap items-center gap-6 hover:shadow-sm transition-shadow"
            >
              <div className="min-w-[220px]">
                <div className="flex items-center gap-2 font-semibold">
                  <TeamBadge abbr={game.away_team.abbreviation} />
                  <span>{game.away_team.abbreviation}</span>
                  <span className="text-[color:var(--color-ink-faint)] font-normal">@</span>
                  <TeamBadge abbr={game.home_team.abbreviation} />
                  <span>{game.home_team.abbreviation}</span>
                </div>
                <div className="text-xs text-[color:var(--color-ink-faint)] mt-1">
                  {game.date.slice(5)} · Final: {game.away_team.abbreviation} {game.away_score} - {game.home_team.abbreviation} {game.home_score}
                </div>
              </div>

              <div className="flex-1 flex flex-wrap items-center gap-x-8 gap-y-2">
                <div>
                  <div className="text-[10px] font-medium text-[color:var(--color-ink-faint)] tracking-wide">PREDICTED WINNER</div>
                  <div className="text-base font-semibold">
                    {moneyline?.predicted_probability != null
                      ? moneyline.predicted_probability >= 0.5
                        ? `${game.home_team.abbreviation} (${Math.round(moneyline.predicted_probability * 100)}%)`
                        : `${game.away_team.abbreviation} (${Math.round((1 - moneyline.predicted_probability) * 100)}%)`
                      : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] font-medium text-[color:var(--color-ink-faint)] tracking-wide">PREDICTED TOTAL</div>
                  <div className="text-base font-semibold">
                    {total?.predicted_value != null ? (
                      <>
                        {total.predicted_value.toFixed(1)}{' '}
                        <span className="text-[color:var(--color-ink-faint)] font-normal">vs actual {game.home_score! + game.away_score!}</span>
                      </>
                    ) : (
                      '—'
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] font-medium text-[color:var(--color-ink-faint)] tracking-wide">RESULT</div>
                  <div
                    className="text-base font-semibold"
                    style={{ color: winnerCorrect == null ? 'var(--color-ink-faint)' : winnerCorrect ? 'var(--color-good)' : 'var(--color-critical)' }}
                  >
                    {winnerCorrect == null ? '—' : winnerCorrect ? 'Correct' : 'Missed'}
                  </div>
                </div>
                {totalDiff != null && (
                  <div>
                    <div className="text-[10px] font-medium text-[color:var(--color-ink-faint)] tracking-wide">TOTAL DIFF</div>
                    <div className="text-base font-semibold">{totalDiff > 0 ? `+${totalDiff.toFixed(1)}` : totalDiff.toFixed(1)}</div>
                  </div>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
