import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { BacktestResult } from '../api/types'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { localIsoDate } from '../lib/date'

const MODELS = ['moneyline_xgboost', 'moneyline_logistic', 'nrfi_logistic', 'nrfi_xgboost']

interface SeasonRow {
  season: number
  result: BacktestResult | null
  error: string | null
}

// ROI uses the moneyline bet simulation when odds are available, and falls
// back to the plain classification record for NRFI models when there is no
// odds-backed stake path to simulate.
function seasonDateRange(season: number): [string, string] {
  const start = `${season}-03-01`
  const today = localIsoDate(new Date())
  const seasonEnd = `${season}-12-31`
  const end = today < seasonEnd ? today : seasonEnd
  return [start, end]
}

export function ROI() {
  const [model, setModel] = useState(MODELS[0])
  const [seasons, setSeasons] = useState<number[]>([])
  const [rows, setRows] = useState<SeasonRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .backtestSeasons()
      .then((s) => !cancelled && setSeasons(s))
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (seasons.length === 0) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setRows(seasons.map((season) => ({ season, result: null, error: null })))

    Promise.all(
      seasons.map(async (season) => {
        const [start, end] = seasonDateRange(season)
        try {
          const result = await api.backtestResults(model, `${start},${end}`)
          return { season, result, error: null } as SeasonRow
        } catch (err) {
          return {
            season,
            result: null,
            error: err instanceof Error ? err.message : `No data for ${season} yet.`,
          } as SeasonRow
        }
      }),
    ).then((results) => {
      if (!cancelled) setRows(results)
    }).finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }, [model, seasons])

  const pct = (v: number | null | undefined) => (v == null ? null : `${(v * 100).toFixed(1)}%`)

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight mb-1">ROI</h1>
      <p className="text-sm text-[color:var(--color-ink-muted)] mb-6">
        Season-by-season rate of return betting the model's favored side, flat $100/bet, only when it beats the
        market by at least 2 points. Falls back to a plain win/loss record for seasons with no odds history.
      </p>

      <div className="flex flex-wrap gap-3 mb-6">
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>

      {error && <ErrorState message={error} />}
      {!error && seasons.length === 0 && !loading && <EmptyState message="No completed games found yet." />}
      {!error && loading && <LoadingState label="Scoring every season - each one's cached after the first run, so this only takes a while the first time…" />}

      {!error && !loading && rows.length > 0 && (
        <div className="space-y-4">
          {rows
            .slice()
            .sort((a, b) => b.season - a.season)
            .map((row) => {
              const r = row.result
              const hasOdds = r != null && r.roi_flat_bet != null && (r.n_bets ?? 0) > 0
              const record = r && r.wins != null && r.losses != null ? `${r.wins}-${r.losses}` : null

              return (
                <div key={row.season} className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-semibold">{row.season} season</h2>
                    {r && <span className="text-xs text-[color:var(--color-ink-faint)]">{r.date_range}</span>}
                  </div>

                  {row.error && <ErrorState message={row.error} />}

                  {r && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
                        <div className="text-[11px] text-[color:var(--color-ink-faint)]">Record</div>
                        <div className="text-lg font-bold mt-0.5">{record ?? '—'}</div>
                      </div>
                      <div className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
                        <div className="text-[11px] text-[color:var(--color-ink-faint)]">Win %</div>
                        <div className="text-lg font-bold mt-0.5">{pct(r.wins != null && r.n_games ? r.wins / r.n_games : null) ?? '—'}</div>
                      </div>
                      <div className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
                        <div className="text-[11px] text-[color:var(--color-ink-faint)]">Avg return (flat bet)</div>
                        <div className="text-lg font-bold mt-0.5">{hasOdds ? pct(r.roi_flat_bet) : 'N/A'}</div>
                      </div>
                      <div className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
                        <div className="text-[11px] text-[color:var(--color-ink-faint)]">Bets placed</div>
                        <div className="text-lg font-bold mt-0.5">{r.n_bets ?? 0}</div>
                      </div>
                    </div>
                  )}

                  {r && !hasOdds && (
                    <p className="text-xs text-[color:var(--color-ink-faint)] mt-3">
                      No odds history for this season (or too few games met the betting edge threshold) - showing
                      the win/loss record instead of a real dollar return.
                    </p>
                  )}
                </div>
              )
            })}
        </div>
      )}
    </div>
  )
}
