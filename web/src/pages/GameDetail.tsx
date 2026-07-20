import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import type { Game, GameFeatures, Prediction, StarterFeatures } from '../api/types'
import { Panel, StatGrid } from '../components/StatGrid'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { TeamBadge } from '../components/TeamBadge'
import { MetricCard } from '../components/MetricCard'
import { preferredPrediction } from '../lib/predictions'

function starterStats(s: StarterFeatures) {
  return [
    { label: 'ERA (season)', value: s.era_season },
    { label: 'FIP (season)', value: s.fip_season },
    { label: 'ERA (L3 starts)', value: s.era_last_3_starts },
    { label: 'K% (rolling)', value: s.k_pct_rolling, suffix: '%' },
    { label: 'BB% (rolling)', value: s.bb_pct_rolling, suffix: '%' },
    { label: 'Days rest', value: s.days_rest },
    { label: 'Last start pitches', value: s.pitch_count_last_start },
    { label: 'Throws', value: s.handedness },
  ]
}

const TABS = ['Predictions', 'Feature breakdown'] as const

export function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>()
  const id = Number(gameId)

  const [game, setGame] = useState<Game | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [features, setFeatures] = useState<GameFeatures | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<(typeof TABS)[number]>('Predictions')

  useEffect(() => {
    if (!Number.isFinite(id)) return
    let cancelled = false
    setLoading(true)
    setError(null)

    Promise.all([api.getGame(id), api.getGamePredictions(id), api.getGameFeatures(id)])
      .then(([g, p, f]) => {
        if (cancelled) return
        setGame(g)
        setPredictions(p.predictions)
        setFeatures(f.features)
      })
      .catch((err: unknown) => {
        if (cancelled) return
        setError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => !cancelled && setLoading(false))

    return () => {
      cancelled = true
    }
  }, [id])

  if (loading) return <LoadingState label="Loading game - the feature breakdown pulls live Statcast data, so this can take a bit…" />
  if (error) return <ErrorState message={error} />
  if (!game) return <EmptyState message="Game not found." />

  const moneyline = preferredPrediction(predictions, 'moneyline')
  const total = preferredPrediction(predictions, 'total')
  const nrfi = preferredPrediction(predictions, 'nrfi')
  const homeProb = moneyline?.predicted_probability ?? null

  return (
    <div>
      <Link to="/" className="text-sm text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)]">
        ← Today's Slate
      </Link>

      <div className="mt-3 mb-6 flex items-center gap-3">
        <TeamBadge abbr={game.away_team.abbreviation} size={36} />
        <h1 className="text-2xl font-bold tracking-tight">
          {game.away_team.abbreviation} @ {game.home_team.abbreviation}
        </h1>
        <TeamBadge abbr={game.home_team.abbreviation} size={36} />
      </div>
      <p className="text-sm text-[color:var(--color-ink-muted)] -mt-4 mb-6">
        {game.date} · {game.venue?.name ?? 'TBD'} · {game.status}
        {game.status === 'final' && game.away_score != null && game.home_score != null &&
          ` · Final ${game.away_team.abbreviation} ${game.away_score} - ${game.home_team.abbreviation} ${game.home_score}`}
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <MetricCard label={`${game.home_team.abbreviation} expected win %`} value={homeProb != null ? `${Math.round(homeProb * 100)}%` : '—'} />
        <MetricCard label={`${game.away_team.abbreviation} expected win %`} value={homeProb != null ? `${Math.round((1 - homeProb) * 100)}%` : '—'} />
        <MetricCard label="Predicted total runs" value={total?.predicted_value != null ? total.predicted_value.toFixed(1) : '—'} />
        <MetricCard label="NRFI probability" value={nrfi?.predicted_probability != null ? `${Math.round(nrfi.predicted_probability * 100)}%` : '—'} />
      </div>

      <div className="border-b border-[color:var(--color-border)] mb-6 flex gap-6">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`pb-3 text-sm font-medium border-b-2 -mb-px transition-colors ${
              tab === t
                ? 'border-[color:var(--color-home)] text-[color:var(--color-home)]'
                : 'border-transparent text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)]'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Predictions' && (
        predictions.length === 0 ? (
          <EmptyState message="No predictions generated yet for this game." />
        ) : (
          <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] font-medium text-[color:var(--color-ink-faint)] tracking-wide border-b border-[color:var(--color-border)]">
                  <th className="px-4 py-3">Target</th>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-4 py-3">Probability</th>
                  <th className="px-4 py-3">Value</th>
                  <th className="px-4 py-3">Generated</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.id} className="border-b border-[color:var(--color-border)] last:border-0">
                    <td className="px-4 py-3 font-semibold capitalize">{p.target_type}</td>
                    <td className="px-4 py-3 text-[color:var(--color-ink-muted)]">{p.model_version}</td>
                    <td className="px-4 py-3">{p.predicted_probability != null ? `${Math.round(p.predicted_probability * 100)}%` : '—'}</td>
                    <td className="px-4 py-3">{p.predicted_value != null ? p.predicted_value.toFixed(1) : '—'}</td>
                    <td className="px-4 py-3 text-[color:var(--color-ink-faint)]">{new Date(p.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}

      {tab === 'Feature breakdown' && features && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-4">
            <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-away)]">
              {game.away_team.abbreviation} (away)
            </div>
            <Panel title="Starter">
              <StatGrid items={starterStats(features.away_starter)} />
            </Panel>
            <Panel title="Bullpen">
              <StatGrid
                items={[
                  { label: 'ERA (7d)', value: features.away_bullpen.bullpen_era_rolling_7d },
                  { label: 'ERA (14d)', value: features.away_bullpen.bullpen_era_rolling_14d },
                  { label: 'IP last 3g', value: features.away_bullpen.innings_thrown_last_3_games },
                  { label: 'Closer available', value: features.away_bullpen.closer_available == null ? null : features.away_bullpen.closer_available ? 'Yes' : 'No' },
                ]}
              />
            </Panel>
            <Panel title="Team form">
              <StatGrid
                items={[
                  { label: 'Win% (season)', value: features.away_team.win_pct_season },
                  { label: 'Win% (L10)', value: features.away_team.win_pct_last_10 },
                  { label: 'Run diff', value: features.away_team.run_diff_season },
                  { label: 'Pythag win%', value: features.away_team.pythag_win_pct },
                  { label: 'OAA (defense)', value: features.away_team.oaa_defense_rating },
                ]}
              />
            </Panel>
            <Panel title="Lineup">
              <StatGrid
                items={[
                  { label: 'wOBA (weighted)', value: features.away_lineup.lineup_wOBA_weighted_by_order },
                  { label: 'Platoon advantage', value: features.away_lineup.platoon_advantage_count },
                  { label: 'Confirmed', value: features.away_lineup.lineup_confirmed ? 'Yes' : 'No' },
                ]}
              />
            </Panel>
          </div>

          <div className="space-y-4">
            <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-home)]">
              {game.home_team.abbreviation} (home)
            </div>
            <Panel title="Starter">
              <StatGrid items={starterStats(features.home_starter)} />
            </Panel>
            <Panel title="Bullpen">
              <StatGrid
                items={[
                  { label: 'ERA (7d)', value: features.home_bullpen.bullpen_era_rolling_7d },
                  { label: 'ERA (14d)', value: features.home_bullpen.bullpen_era_rolling_14d },
                  { label: 'IP last 3g', value: features.home_bullpen.innings_thrown_last_3_games },
                  { label: 'Closer available', value: features.home_bullpen.closer_available == null ? null : features.home_bullpen.closer_available ? 'Yes' : 'No' },
                ]}
              />
            </Panel>
            <Panel title="Team form">
              <StatGrid
                items={[
                  { label: 'Win% (season)', value: features.home_team.win_pct_season },
                  { label: 'Win% (L10)', value: features.home_team.win_pct_last_10 },
                  { label: 'Run diff', value: features.home_team.run_diff_season },
                  { label: 'Pythag win%', value: features.home_team.pythag_win_pct },
                  { label: 'OAA (defense)', value: features.home_team.oaa_defense_rating },
                ]}
              />
            </Panel>
            <Panel title="Lineup">
              <StatGrid
                items={[
                  { label: 'wOBA (weighted)', value: features.home_lineup.lineup_wOBA_weighted_by_order },
                  { label: 'Platoon advantage', value: features.home_lineup.platoon_advantage_count },
                  { label: 'Confirmed', value: features.home_lineup.lineup_confirmed ? 'Yes' : 'No' },
                ]}
              />
            </Panel>
          </div>

          <div className="md:col-span-2">
            <Panel title="Park & umpire">
              <StatGrid
                items={[
                  { label: 'Park factor (runs)', value: features.park_weather.park_factor_runs },
                  { label: 'Park factor (HR)', value: features.park_weather.park_factor_hr },
                  { label: 'Roof closed', value: features.park_weather.roof_closed == null ? null : features.park_weather.roof_closed ? 'Yes' : 'No' },
                  { label: 'Umpire zone size %ile', value: features.umpire.strike_zone_size_percentile },
                  { label: 'Umpire O/U lean (avg runs)', value: features.umpire.over_under_lean },
                  { label: 'Umpire K/game', value: features.umpire.k_rate_boost },
                ]}
              />
            </Panel>
          </div>
        </div>
      )}
    </div>
  )
}
