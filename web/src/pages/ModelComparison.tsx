import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { BacktestResult } from '../api/types'
import { MetricCard } from '../components/MetricCard'
import { localIsoDaysAgo } from '../lib/date'
import { ErrorState, LoadingState } from '../components/States'

const PAIRS: Record<string, [string, string]> = {
  moneyline: ['moneyline_logistic', 'moneyline_xgboost'],
  nrfi: ['nrfi_logistic', 'nrfi_xgboost'],
}

function brierScore(labels: number[], probs: number[]) {
  const n = labels.length
  return labels.reduce((sum, y, i) => sum + (probs[i] - y) ** 2, 0) / n
}

function logLoss(labels: number[], probs: number[]) {
  const eps = 1e-15
  const n = labels.length
  return (
    -labels.reduce((sum, y, i) => {
      const p = Math.min(Math.max(probs[i], eps), 1 - eps)
      return sum + (y * Math.log(p) + (1 - y) * Math.log(1 - p))
    }, 0) / n
  )
}

export function ModelComparison() {
  const [target, setTarget] = useState<keyof typeof PAIRS>('moneyline')
  // Short by default - each side's backtest rebuilds features for every
  // game in range from scratch (no caching), so a wide window is slow.
  // See Backtest.tsx's identical comment for the underlying reason.
  const [start, setStart] = useState(localIsoDaysAgo(7))
  const [end, setEnd] = useState(localIsoDaysAgo(0))
  const [results, setResults] = useState<Record<string, BacktestResult | { error: string }>>({})
  const [blend, setBlend] = useState<{ accuracy: number; logLoss: number; brier: number; n: number } | null>(null)
  const [loading, setLoading] = useState(false)

  const [baselineName, xgbName] = PAIRS[target]

  useEffect(() => {
    if (start >= end) return
    let cancelled = false
    setLoading(true)
    setBlend(null)

    async function run() {
      const dateRange = `${start},${end}`
      const entries = await Promise.all(
        [baselineName, xgbName].map(async (name) => {
          try {
            return [name, await api.backtestResults(name, dateRange)] as const
          } catch {
            return [name, { error: 'Not trained yet or no data for this range.' }] as const
          }
        }),
      )
      if (cancelled) return
      setResults(Object.fromEntries(entries))

      try {
        const [basePreds, xgbPreds] = await Promise.all([
          api.predictionHistory(dateRange, target),
          api.predictionHistory(dateRange, target),
        ])
        const byGameBase = new Map(
          basePreds.filter((p) => p.model_name === baselineName && p.predicted_probability != null).map((p) => [p.game_id, p.predicted_probability as number]),
        )
        const byGameXgb = new Map(
          xgbPreds.filter((p) => p.model_name === xgbName && p.predicted_probability != null).map((p) => [p.game_id, p.predicted_probability as number]),
        )
        const commonIds = [...byGameBase.keys()].filter((id) => byGameXgb.has(id))

        const games = await Promise.all(commonIds.map((id) => api.getGame(id).catch(() => null)))
        const labels: number[] = []
        const blended: number[] = []
        games.forEach((g, i) => {
          if (!g || g.home_score == null || g.away_score == null) return
          labels.push(g.home_score > g.away_score ? 1 : 0)
          const id = commonIds[i]
          blended.push(((byGameBase.get(id) ?? 0) + (byGameXgb.get(id) ?? 0)) / 2)
        })

        if (!cancelled && labels.length > 0) {
          const acc = labels.reduce((s, y, i) => s + (Number(blended[i] >= 0.5) === y ? 1 : 0), 0) / labels.length
          setBlend({ accuracy: acc, logLoss: logLoss(labels, blended), brier: brierScore(labels, blended), n: labels.length })
        }
      } catch {
        // blend is optional - leave it null if prediction history isn't available
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    run()
    return () => {
      cancelled = true
    }
  }, [target, start, end, baselineName, xgbName])

  const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
  const num = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(4))

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight mb-1">Model Comparison</h1>
      <p className="text-sm text-[color:var(--color-ink-muted)] mb-6">
        Baseline vs. XGBoost side by side, plus a simple 50/50 blend scored against actual outcomes.
      </p>

      <div className="flex flex-wrap gap-3 mb-6">
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value as keyof typeof PAIRS)}
          className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm"
        >
          {Object.keys(PAIRS).map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <input type="date" value={start} onChange={(e) => setStart(e.target.value)} className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm" />
        <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} className="bg-[color:var(--color-surface-card)] border border-[color:var(--color-border)] rounded-lg px-3 py-2 text-sm" />
      </div>

      {loading && <LoadingState label="Backtesting both models over this range - can take a few seconds…" />}

      {!loading && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            {[baselineName, xgbName].map((name) => {
              const r = results[name]
              return (
                <div key={name} className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-4">
                  <h3 className="text-sm font-semibold mb-3">{name}</h3>
                  {r && 'error' in r ? (
                    <ErrorState message={r.error} />
                  ) : r ? (
                    <div className="grid grid-cols-3 gap-2">
                      <MetricCard label="Accuracy" value={pct(r.accuracy)} />
                      <MetricCard label="Log loss" value={num(r.log_loss)} />
                      <MetricCard label="Brier" value={num(r.brier_score)} />
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>

          <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-4">
            <h3 className="text-sm font-semibold mb-1">Blended (mean of both models)</h3>
            <p className="text-xs text-[color:var(--color-ink-faint)] mb-3">
              Averages each model's per-game probability and scores it against actual outcomes.
            </p>
            {blend ? (
              <div className="grid grid-cols-4 gap-2">
                <MetricCard label="Accuracy" value={pct(blend.accuracy)} />
                <MetricCard label="Log loss" value={num(blend.logLoss)} />
                <MetricCard label="Brier" value={num(blend.brier)} />
                <MetricCard label="Games" value={String(blend.n)} />
              </div>
            ) : (
              <p className="text-sm text-[color:var(--color-ink-faint)]">
                No overlapping predictions from both models yet for this range.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
