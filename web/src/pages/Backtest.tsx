import { useEffect, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api/client'
import type { BacktestResult } from '../api/types'
import { MetricCard } from '../components/MetricCard'
import { localIsoDate, localIsoDaysAgo } from '../lib/date'
import { EmptyState, ErrorState, LoadingState } from '../components/States'

const MODELS = ['moneyline_xgboost', 'moneyline_logistic', 'totals_xgboost', 'totals_poisson', 'nrfi_logistic', 'nrfi_xgboost']

interface WeekPoint {
  week: string
  accuracy: number | null
  brierScore: number | null
}

export function Backtest() {
  const [model, setModel] = useState(MODELS[0])
  // Kept short by default - a range never scored before still rebuilds
  // the full per-game feature set (tens of seconds), even though the
  // backend now caches that result per (model, range) so every repeat
  // visit after the first is instant. Widen it deliberately, not as the
  // default someone waits on before seeing anything.
  const [start, setStart] = useState(localIsoDaysAgo(7))
  const [end, setEnd] = useState(localIsoDaysAgo(0))
  const [overall, setOverall] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const [trend, setTrend] = useState<WeekPoint[]>([])
  const [trendLoading, setTrendLoading] = useState(false)
  const [trendRequested, setTrendRequested] = useState(false)

  function runBacktest(refresh = false) {
    if (start >= end) return () => {}
    let cancelled = false
    setLoading(true)
    setError(null)
    setTrend([])
    setTrendRequested(false)

    api
      .backtestResults(model, `${start},${end}`, refresh)
      .then((result) => !cancelled && setOverall(result))
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : `Backtest failed - is '${model}' trained yet?`)
      })
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }

  useEffect(() => {
    return runBacktest(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, start, end])

  function loadTrend() {
    setTrendRequested(true)
    setTrendLoading(true)

    const weeks: Date[] = []
    const cursor = new Date(start)
    while (cursor < new Date(end)) {
      weeks.push(new Date(cursor))
      cursor.setDate(cursor.getDate() + 7)
    }

    Promise.all(
      weeks.map(async (weekStart) => {
        const weekEnd = new Date(weekStart)
        weekEnd.setDate(weekEnd.getDate() + 7)
        const wsISO = localIsoDate(weekStart)
        const weISO = localIsoDate(weekEnd)
        try {
          const r = await api.backtestResults(model, `${wsISO},${weISO}`)
          if (!r.n_bets) return null
          return { week: wsISO.slice(5), accuracy: r.accuracy, brierScore: r.brier_score } as WeekPoint
        } catch {
          return null
        }
      }),
    )
      .then((points) => setTrend(points.filter((p): p is WeekPoint => p !== null)))
      .finally(() => setTrendLoading(false))
  }

  const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
  const num = (v: number | null | undefined, digits = 4) => (v == null ? 'N/A' : v.toFixed(digits))

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight mb-1">Backtest Results</h1>
      <p className="text-sm text-[color:var(--color-ink-muted)] mb-6">
        Accuracy, calibration, and ROI/CLV for a trained model over a date range.
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
        <input
          type="date"
          value={start}
          onChange={(e) => setStart(e.target.value)}
          className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
        />
        <input
          type="date"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
          className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
        />
      </div>

      {loading && <LoadingState label="Scoring the model against this range - cached after the first run for this exact range, so repeat visits are instant…" />}
      {!loading && error && <ErrorState message={error} />}

      {!loading && !error && overall && (
        <>
          <div className="flex items-center justify-between mb-3 text-xs text-[color:var(--color-ink-faint)]">
            <span>{overall.computed_at ? `Computed ${new Date(overall.computed_at).toLocaleString()}` : ''}</span>
            <button
              onClick={() => runBacktest(true)}
              className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-3 py-1.5 font-medium hover:border-[color:var(--color-home)]/50 transition-colors"
            >
              Refresh
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
            <MetricCard label="Accuracy" value={pct(overall.accuracy)} />
            <MetricCard label="Log loss" value={num(overall.log_loss)} />
            <MetricCard label="Brier score" value={num(overall.brier_score)} />
            <MetricCard label="MAE" value={num(overall.mae)} />
            <MetricCard label="ROI (flat bet)" value={overall.roi_flat_bet != null ? pct(overall.roi_flat_bet) : 'N/A'} />
            <MetricCard label="ROI (Kelly)" value={overall.roi_kelly != null ? pct(overall.roi_kelly) : 'N/A'} />
            <MetricCard label="Avg CLV" value={overall.clv_avg != null ? `${overall.clv_avg.toFixed(2)}%` : 'N/A'} />
            <MetricCard label="Games scored" value={String(overall.n_bets)} hint={overall.date_range} />
          </div>

          {!trendRequested && (
            <button
              onClick={loadTrend}
              className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-4 py-2 text-sm font-medium hover:border-[color:var(--color-home)]/50 transition-colors"
            >
              Load weekly trend
            </button>
          )}

          {trendLoading && <LoadingState label="Building weekly trend - one more backtest per week, so this is the slow part…" />}

          {trendRequested && !trendLoading && (
            trend.length > 0 ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-4">
                  <h3 className="text-sm font-semibold text-[color:var(--color-ink-muted)] mb-3">Accuracy by week</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={trend}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="week" stroke="var(--color-ink-faint)" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis
                        stroke="var(--color-ink-faint)"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                      />
                      <Tooltip
                        contentStyle={{ background: 'var(--color-surface-raised)', border: '1px solid var(--color-border)', borderRadius: 8, fontSize: 12 }}
                        formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`}
                      />
                      <Line type="monotone" dataKey="accuracy" stroke="var(--color-home)" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-4">
                  <h3 className="text-sm font-semibold text-[color:var(--color-ink-muted)] mb-3">
                    Brier score by week (lower = better calibrated)
                  </h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={trend}>
                      <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="week" stroke="var(--color-ink-faint)" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="var(--color-ink-faint)" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip
                        contentStyle={{ background: 'var(--color-surface-raised)', border: '1px solid var(--color-border)', borderRadius: 8, fontSize: 12 }}
                      />
                      <Line type="monotone" dataKey="brierScore" stroke="var(--color-away)" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <EmptyState message="Not enough weekly data to plot a trend over this range." />
            )
          )}
        </>
      )}
    </div>
  )
}
