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

const TABS = ['Predictions', 'Feature breakdown', "How it's calculated"] as const

export function GameDetail() {
  const { gameId } = useParams<{ gameId: string }>()
  const id = Number(gameId)

  const [game, setGame] = useState<Game | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [edgeVsMarket, setEdgeVsMarket] = useState<{ model_probability_home: number; market_implied_probability_home: number; edge: number; expected_roi: number } | null>(null)
  const [features, setFeatures] = useState<GameFeatures | null>(null)
  const [featuresComputedAt, setFeaturesComputedAt] = useState<string | null>(null)
  const [featuresLoading, setFeaturesLoading] = useState(false)
  const [featuresError, setFeaturesError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<(typeof TABS)[number]>('Predictions')

  // Game + predictions are already-computed DB reads - fast. Features used
  // to be fetched in the same Promise.all, which meant the whole page (even
  // the Predictions tab, the default view) sat on a loading spinner behind
  // a live Statcast feature build no one had asked to see yet. Loaded
  // separately below, only once the Feature breakdown tab is opened.
  useEffect(() => {
    if (!Number.isFinite(id)) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setFeatures(null)
    setFeaturesComputedAt(null)
    setFeaturesError(null)

    Promise.all([api.getGame(id), api.getGamePredictions(id)])
      .then(([g, p]) => {
        if (cancelled) return
        setGame(g)
        setPredictions(p.predictions)
        setEdgeVsMarket(p.edge_vs_market)
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

  function loadFeatures(refresh = false) {
    if (!Number.isFinite(id)) return
    setFeaturesLoading(true)
    setFeaturesError(null)
    api
      .getGameFeatures(id, refresh)
      .then((f) => {
        setFeatures(f.features)
        setFeaturesComputedAt(f.computed_at)
      })
      .catch((err: unknown) => {
        setFeaturesError(err instanceof ApiError ? `API error (${err.status}): ${err.message}` : 'Could not reach the API.')
      })
      .finally(() => setFeaturesLoading(false))
  }

  useEffect(() => {
    if (tab === 'Feature breakdown' && features === null && !featuresLoading && !featuresError) {
      loadFeatures()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab, id])

  if (loading) return <LoadingState label="Loading game…" />
  if (error) return <ErrorState message={error} />
  if (!game) return <EmptyState message="Game not found." />

  const moneyline = preferredPrediction(predictions, 'moneyline')
  const total = preferredPrediction(predictions, 'total')
  const nrfi = preferredPrediction(predictions, 'nrfi')
  const homeProb = moneyline?.predicted_probability ?? null
  const totalSplit = total?.predicted_home_value != null && total?.predicted_away_value != null
    ? `${game.home_team.abbreviation} ${total.predicted_home_value.toFixed(1)} · ${game.away_team.abbreviation} ${total.predicted_away_value.toFixed(1)}`
    : total?.predicted_value != null
      ? total.predicted_value.toFixed(1)
      : '—'

  return (
    <div>
      <Link to="/" className="text-sm text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)]">
        ← Today's Slate
      </Link>

      <div className="text-xs font-semibold uppercase tracking-wide text-[color:var(--color-home)] mt-3 mb-1">Matchup</div>
      <div className="mb-6 flex items-center gap-3">
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

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
        <MetricCard label={`${game.home_team.abbreviation} win %`} value={homeProb != null ? `${Math.round(homeProb * 100)}%` : '—'} />
        <MetricCard label={`${game.away_team.abbreviation} win %`} value={homeProb != null ? `${Math.round((1 - homeProb) * 100)}%` : '—'} />
        <MetricCard label="Predicted total" value={totalSplit} />
        <MetricCard
          label="Expected ROI (flat bet)"
          value={edgeVsMarket?.expected_roi != null ? `${edgeVsMarket.expected_roi >= 0 ? '+' : ''}${Math.round(edgeVsMarket.expected_roi * 100)}%` : 'N/A'}
          tone={edgeVsMarket?.expected_roi != null ? (edgeVsMarket.expected_roi >= 0 ? 'good' : 'critical') : undefined}
        />
      </div>

      <p className="text-sm text-[color:var(--color-ink-muted)] mb-6">
        NRFI (no runs first inning) probability:{' '}
        <span className="font-semibold text-[color:var(--color-ink)]">
          {nrfi?.predicted_probability != null ? `${Math.round(nrfi.predicted_probability * 100)}%` : '—'}
        </span>
      </p>

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
                  <th className="px-4 py-3">ROI</th>
                  <th className="px-4 py-3">Generated</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => {
                  // Expected ROI is only a real, computable number for
                  // moneyline (needs a market price to bet against - see
                  // api/routers/games._compute_edge_vs_market) - total/NRFI
                  // rows show N/A rather than a fabricated figure.
                  const roi = p.target_type === 'moneyline' ? edgeVsMarket?.expected_roi : null
                  return (
                    <tr key={p.id} className="border-b border-[color:var(--color-border)] last:border-0">
                      <td className="px-4 py-3 font-semibold capitalize">{p.target_type}</td>
                      <td className="px-4 py-3 text-[color:var(--color-ink-muted)]">{p.model_version}</td>
                      <td className="px-4 py-3">{p.predicted_probability != null ? `${Math.round(p.predicted_probability * 100)}%` : '—'}</td>
                      <td className="px-4 py-3">{p.predicted_value != null ? p.predicted_value.toFixed(1) : '—'}</td>
                      <td className={`px-4 py-3 font-medium ${roi != null ? (roi >= 0 ? 'text-[color:var(--color-good)]' : 'text-[color:var(--color-critical)]') : 'text-[color:var(--color-warning)]'}`}>
                        {roi != null ? `${roi >= 0 ? '+' : ''}${Math.round(roi * 100)}%` : 'N/A'}
                      </td>
                      <td className="px-4 py-3 text-[color:var(--color-ink-faint)]">{new Date(p.created_at).toLocaleString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )
      )}

      {tab === 'Feature breakdown' && featuresLoading && (
        <LoadingState label="Building the feature breakdown - pulls live Statcast data the first time, so this can take a bit…" />
      )}

      {tab === 'Feature breakdown' && !featuresLoading && featuresError && <ErrorState message={featuresError} />}

      {tab === 'Feature breakdown' && features && (
        <div>
          <div className="flex items-center justify-between mb-4 text-xs text-[color:var(--color-ink-faint)]">
            <span>{featuresComputedAt ? `Computed ${new Date(featuresComputedAt).toLocaleString()}` : ''}</span>
            <button
              onClick={() => loadFeatures(true)}
              disabled={featuresLoading}
              className="rounded-lg border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-3 py-1.5 font-medium hover:border-[color:var(--color-home)]/50 transition-colors disabled:opacity-50"
            >
              Refresh
            </button>
          </div>
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
        </div>
      )}

      {tab === "How it's calculated" && (
        <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-6 space-y-4 text-sm text-[color:var(--color-ink-muted)] max-w-3xl">
          <div>
            <h3 className="font-semibold text-[color:var(--color-ink)] mb-1">Moneyline &amp; NRFI</h3>
            <p>
              An <code className="text-xs">XGBClassifier</code> (falling back to a calibrated logistic regression
              when XGBoost hasn't shown a real improvement) trained on starter and bullpen form, team win rates,
              lineup quality, park factors, and market pricing where available. Both are wrapped in isotonic
              calibration so a "65%" prediction actually wins about 65% of the time - required for comparing
              against the market's own probability.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-[color:var(--color-ink)] mb-1">Run total</h3>
            <p>
              Two independent Poisson distributions (one per team) convolved into a full distribution over
              possible totals, or an <code className="text-xs">XGBRegressor</code> predicting the combined total
              directly - whichever backtests better wins. See the <Link to="/models" className="text-[color:var(--color-home)] hover:underline">Models</Link> page for current metrics.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-[color:var(--color-ink)] mb-1">Expected ROI</h3>
            <p>
              Bet $1 on whichever side the model favors, at that side's real market price: <code className="text-xs">model_win_prob × profit_per_dollar − (1 − model_win_prob)</code>.
              This is a real expected-value calculation, not a backtested result - it assumes the model's own
              probability is correct, which the <Link to="/roi" className="text-[color:var(--color-home)] hover:underline">Model ROI</Link> page's held-out track record is what actually
              validates (or doesn't).
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-[color:var(--color-ink)] mb-1">Market edge</h3>
            <p>
              Sportsbook prices always imply more than 100% combined probability (the "vig") - the model's
              probability is compared against the de-vigged, normalized fair probability, not the raw market
              number, so the edge shown isn't inflated by the book's own margin.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
