import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { BacktestResult, SpreadResult } from '../api/types'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { localIsoDate } from '../lib/date'

// Season boundary: March 1 through today (or Dec 31, whichever is
// earlier) - matches how every other season-scoped view in this app
// (build_training_matrix's SEASON_START_MONTH_DAY, etc.) defines "season".
function seasonDateRange(season: number): [string, string] {
  const start = `${season}-03-01`
  const today = localIsoDate(new Date())
  const seasonEnd = `${season}-12-31`
  const end = today < seasonEnd ? today : seasonEnd
  return [start, end]
}

interface BetTypeCard {
  key: string
  label: string
  color: string
  softColor: string
  roi: number | null
  hasOdds: boolean
  wins: number | null
  losses: number | null
  winRate: number | null
}

function pct(v: number | null | undefined, digits = 0) {
  return v == null ? null : `${(v * 100).toFixed(digits)}%`
}

export function ROI() {
  const [seasons, setSeasons] = useState<number[]>([])
  const [season, setSeason] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [cards, setCards] = useState<BetTypeCard[]>([])

  useEffect(() => {
    let cancelled = false
    api
      .backtestSeasons()
      .then((s) => {
        if (cancelled) return
        setSeasons(s)
        setSeason(s.length ? s[s.length - 1] : null)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (season == null) return
    let cancelled = false
    setLoading(true)
    setError(null)

    const [start, end] = seasonDateRange(season)
    const dateRange = `${start},${end}`

    Promise.all([
      api.backtestResults('moneyline_xgboost', dateRange).catch(() => null as BacktestResult | null),
      api.backtestResults('totals_poisson', dateRange).catch(() => null as BacktestResult | null),
      api.spreadResults(dateRange).catch(() => null as SpreadResult | null),
      api.backtestResults('nrfi_logistic', dateRange).catch(() => null as BacktestResult | null),
    ])
      .then(([moneyline, , spread, nrfi]) => {
        if (cancelled) return
        setCards([
          {
            key: 'moneyline',
            label: 'Moneyline',
            color: 'var(--color-home)',
            softColor: 'var(--color-home-soft)',
            roi: moneyline?.roi_flat_bet ?? null,
            hasOdds: !!(moneyline?.roi_flat_bet != null && (moneyline?.n_bets ?? 0) > 0),
            wins: moneyline?.wins ?? null,
            losses: moneyline?.losses ?? null,
            winRate: moneyline?.wins != null && moneyline?.n_games ? moneyline.wins / moneyline.n_games : null,
          },
          {
            key: 'total',
            label: 'Run Total',
            color: 'var(--color-good)',
            softColor: 'var(--color-good-soft)',
            roi: null, // regression target - "win/loss" isn't a total-runs concept; MAE/RMSE live on the Backtest page instead
            hasOdds: false,
            wins: null,
            losses: null,
            winRate: null,
          },
          {
            key: 'spread',
            label: 'Spread',
            color: 'var(--color-warning)',
            softColor: 'var(--color-warning-soft)',
            roi: spread?.roi_flat_bet ?? null,
            hasOdds: !!(spread?.roi_flat_bet != null && (spread?.n_bets ?? 0) > 0),
            wins: spread?.wins ?? null,
            losses: spread?.losses ?? null,
            winRate: spread?.wins != null && spread?.n_games ? spread.wins / spread.n_games : null,
          },
          {
            key: 'nrfi',
            label: 'NRFI',
            color: 'var(--color-nrfi)',
            softColor: 'var(--color-nrfi-soft)',
            // NRFI has no real market odds stored (no nrfi_odds column) -
            // there's genuinely no price to compute a dollar return
            // against, so this always falls back to win/loss, not "N/A
            // because we haven't polled enough" like moneyline/spread.
            roi: null,
            hasOdds: false,
            wins: nrfi?.wins ?? null,
            losses: nrfi?.losses ?? null,
            winRate: nrfi?.wins != null && nrfi?.n_games ? nrfi.wins / nrfi.n_games : null,
          },
        ])
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }, [season])

  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-home)] mb-1">Performance</div>
      <h1 className="text-2xl font-semibold tracking-tight mb-1">Model ROI</h1>
      <p className="text-sm text-[color:var(--color-ink-muted)] mb-6">
        Return on a flat 1-unit bet, by bet type, across all graded picks.
      </p>

      {seasons.length > 0 && (
        <div className="flex gap-1 mb-6 border-b border-[color:var(--color-border)]">
          {seasons.map((s) => (
            <button
              key={s}
              onClick={() => setSeason(s)}
              className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                season === s
                  ? 'border-[color:var(--color-home)] text-[color:var(--color-home)]'
                  : 'border-transparent text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)]'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {error && <ErrorState message={error} />}
      {!error && seasons.length === 0 && !loading && <EmptyState message="No completed games found yet." />}
      {!error && loading && <LoadingState label="Scoring every bet type for this season - cached after the first run, so this only takes a while once…" />}

      {!error && !loading && cards.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {cards.map((c) => (
            <div
              key={c.key}
              className="rounded-xl bg-[color:var(--color-surface-card)] p-4 border-l-4"
              style={{ borderLeftColor: c.color }}
            >
              <div className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: c.color }}>
                {c.label}
              </div>
              <div className="text-2xl font-bold mb-2" style={{ color: c.roi == null ? 'var(--color-ink-faint)' : c.roi >= 0 ? 'var(--color-good)' : 'var(--color-critical)' }}>
                {c.hasOdds ? `${c.roi! >= 0 ? '+' : ''}${pct(c.roi)}` : 'N/A'}
              </div>
              <div className="text-xs text-[color:var(--color-ink-muted)]">
                {c.wins != null && c.losses != null ? (
                  <>
                    {c.wins}-{c.losses} record
                    <br />
                    {pct(c.winRate, 0) ?? '—'} win rate
                  </>
                ) : (
                  'No graded picks yet'
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
