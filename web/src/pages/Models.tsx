import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { ModelInfo } from '../api/types'
import { Panel, StatGrid } from '../components/StatGrid'
import { ErrorState, LoadingState } from '../components/States'

interface TargetInfo {
  target: string
  title: string
  question: string
  description: string
  baseline: string
  production: string
  keyFeatures: string[]
}

const TARGETS: TargetInfo[] = [
  {
    target: 'moneyline',
    title: 'Moneyline',
    question: 'Who wins the game?',
    description:
      "Predicts the home team's win probability. Both models are wrapped in isotonic calibration so a " +
      '"65%" prediction actually wins about 65% of the time - required for comparing the model\'s probability ' +
      "to the market's own implied probability (see market_implied_probability in the feature set) and for the " +
      "ROI/Kelly-stake backtest simulation.",
    baseline: 'LogisticRegression (class-balanced, isotonic-calibrated)',
    production: 'XGBClassifier with early stopping, calibrated on a held-out slice',
    keyFeatures: [
      'Starter ERA/FIP, last-3-starts form, velocity trend',
      'Team win% (season, last 10, home/away splits), run differential, Pythagorean win%',
      'Bullpen rolling ERA and availability',
      'Lineup wOBA, platoon advantage, injuries',
      'Park factors, umpire strike-zone lean, market implied probability',
    ],
  },
  {
    target: 'total',
    title: 'Run Total (Over/Under)',
    question: 'How many total runs will be scored?',
    description:
      'Predicts combined home + away runs. The baseline models each side (home runs, away runs) as its own ' +
      'Poisson process and convolves the two distributions into a full run-total distribution - not just a ' +
      'point estimate, so it can also answer "what\'s the probability of exactly 7 runs?" The production model ' +
      'is a plain regressor with a Negative Binomial variance estimate layered on for the same distributional view.',
    baseline: 'Per-side Poisson GLM, convolved into a full run distribution',
    production: 'XGBRegressor + Negative Binomial variance estimate',
    keyFeatures: [
      'Both starters’ ERA/FIP and recent form',
      'Bullpen fatigue (innings thrown last 3 games)',
      'Team offensive form and lineup wOBA for both sides',
      'Park run/HR factors, weather (temp, wind)',
      'Market implied probability (from the total/over-under line, when available)',
    ],
  },
  {
    target: 'nrfi',
    title: 'NRFI / YRFI',
    question: 'Will either team score in the 1st inning?',
    description:
      'Predicts "No Run First Inning" - a single yes/no event, not a full-game outcome, so it leans on '
      + 'first-inning-relevant signals specifically: the confirmed leadoff hitter’s OBP and each starter’s '
      + 'early-game form, on top of the general team/park context. XGBoost is only kept as the production model '
      + 'if it beats the logistic baseline by a real margin on held-out data - otherwise the simpler, more '
      + 'stable logistic model wins by default.',
    baseline: 'LogisticRegression (isotonic-calibrated)',
    production: 'XGBClassifier - kept only if it meaningfully beats the logistic baseline',
    keyFeatures: [
      'Leadoff hitter OBP for both lineups',
      'Starter ERA/FIP and recent form (first-inning-specific splits are a known gap - see README)',
      'Park run factor, weather',
      'Team season-level offensive form',
    ],
  },
  {
    target: 'prop',
    title: 'Player Props',
    question: 'Will this player hit a HR / get a hit / how many strikeouts?',
    description:
      "The spec's own stated stretch goal, and correspondingly lighter than the three targets above: three " +
      'separate targets (home run, hits, pitcher strikeouts) with simpler features and no calibration or ' +
      'backtest wiring yet.',
    baseline: '-',
    production: 'XGBClassifier (HR), XGBRegressor (hits, pitcher strikeouts)',
    keyFeatures: ['Player rolling Statcast form', 'Matchup handedness', 'Park factors'],
  },
]

function targetTypeFor(target: string) {
  return target === 'total' ? 'total' : target
}

export function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    api
      .listModels()
      .then((data) => !cancelled && setModels(data))
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-semibold tracking-tight mb-1">Models</h1>
      <p className="text-sm text-[color:var(--color-ink-muted)] mb-6">
        What each model predicts, how it's built, and its current held-out test metrics.
      </p>

      {loading && <LoadingState label="Loading model registry…" />}
      {!loading && error && <ErrorState message={error} />}

      {!loading && !error && (
        <div className="space-y-6">
          {TARGETS.map((t) => {
            const trained = models.filter((m) => m.target_type === targetTypeFor(t.target))
            return (
              <Panel key={t.target} title={t.title}>
                <p className="text-sm font-medium mb-1">{t.question}</p>
                <p className="text-sm text-[color:var(--color-ink-muted)] mb-4">{t.description}</p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
                  <div className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
                    <div className="text-[11px] text-[color:var(--color-ink-faint)]">Baseline</div>
                    <div className="text-sm font-semibold mt-0.5">{t.baseline}</div>
                  </div>
                  <div className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
                    <div className="text-[11px] text-[color:var(--color-ink-faint)]">Production</div>
                    <div className="text-sm font-semibold mt-0.5">{t.production}</div>
                  </div>
                </div>

                <div className="mb-4">
                  <div className="text-[11px] text-[color:var(--color-ink-faint)] uppercase tracking-wide mb-1.5">
                    Key features
                  </div>
                  <ul className="text-sm text-[color:var(--color-ink-muted)] list-disc list-inside space-y-0.5">
                    {t.keyFeatures.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                </div>

                {trained.length > 0 ? (
                  <div>
                    <div className="text-[11px] text-[color:var(--color-ink-faint)] uppercase tracking-wide mb-1.5">
                      Currently trained (held-out test metrics)
                    </div>
                    <div className="space-y-3">
                      {trained.map((m) => (
                        <div key={m.model_name}>
                          <div className="text-xs text-[color:var(--color-ink-muted)] mb-1">
                            {m.model_name} <span className="text-[color:var(--color-ink-faint)]">{m.version} · trained {new Date(m.trained_at).toLocaleString()}</span>
                          </div>
                          <StatGrid
                            items={Object.entries(m.metrics)
                              .filter(([k]) => k !== 'n')
                              .map(([k, v]) => ({ label: k, value: typeof v === 'number' ? v : String(v) }))}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-[color:var(--color-ink-faint)]">Not trained yet.</p>
                )}
              </Panel>
            )
          })}
        </div>
      )}
    </div>
  )
}
