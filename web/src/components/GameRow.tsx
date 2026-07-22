import { Link } from 'react-router-dom'
import type { Game, GameSlateSummary } from '../api/types'

// NRFI validates at ~50% (see README "On predictive power") - showing a
// call on every game as if it were a peer of moneyline/total would
// overstate weak signal as a real edge. Only surface it when the model's
// probability clears this far from a coin flip, either direction. Single
// named constant so the bar is easy to move later, not scattered inline.
const NRFI_YRFI_THRESHOLD = 0.6

// Confidence bands for the win-probability display - describes how far
// the favored side's probability sits from a coin flip, not a claimed
// win rate (see the tooltip text below, which says this explicitly).
const CONFIDENCE_BANDS = [
  { min: 0.65, label: 'Confident' },
  { min: 0.55, label: 'Leaning' },
  { min: 0, label: 'Near toss-up' },
] as const

const statusStyle: Record<string, string> = {
  final: 'text-[color:var(--color-ink-faint)]',
  live: 'text-[color:var(--color-good)] font-medium',
  scheduled: 'text-[color:var(--color-ink-muted)]',
  postponed: 'text-[color:var(--color-warning)]',
  cancelled: 'text-[color:var(--color-critical)]',
}

function formatTime(iso: string | null) {
  if (!iso) return 'TBD'
  return new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
}

function scoreLabel(game: Game): string {
  if (game.status !== 'final') return game.status === 'live' ? 'Live' : formatTime(game.start_time)
  if (game.away_score == null || game.home_score == null) return 'Final'
  return `Final: ${game.away_team.abbreviation} ${game.away_score} - ${game.home_team.abbreviation} ${game.home_score}`
}

function americanOdds(n: number | null | undefined): string {
  if (n == null) return '—'
  return n > 0 ? `+${n}` : `${n}`
}

function signedNumber(n: number | null | undefined): string {
  if (n == null) return '—'
  return n > 0 ? `+${n}` : `${n}`
}

const COLUMN_HEADERS = ['Score / ML', 'Spread / Run Line', 'Total / O-U'] as const

interface Props {
  game: Game
  summary?: GameSlateSummary
  oddsKeyConfigured?: boolean
}

export function GameRow({ game, summary, oddsKeyConfigured = true }: Props) {
  const odds = summary?.latest_odds ?? null

  const awayScore = summary?.total_away_prediction
  const homeScore = summary?.total_home_prediction
  const modelTotal = summary?.total_prediction ?? null

  // run_line is always quoted from the home team's perspective (see
  // ingestion/odds_api._extract_best_lines) - the away side's line is
  // just its negation, but we only ever ingest the home side's *price*,
  // so the away pill's odds are honestly "—" rather than a guessed number.
  const homeRunLine = odds?.run_line ?? null
  const awayRunLine = homeRunLine != null ? -homeRunLine : null

  const total = odds?.total ?? null
  const nrfiProb = summary?.nrfi_probability ?? null

  // Only a game whose key is configured but simply has no snapshot yet
  // gets this tooltip - when the key isn't configured at all, every game
  // is blank for the same page-wide reason already covered by the banner
  // (see TodaySlate.tsx), and repeating that per row would be noisy.
  const oddsMissingTitle = oddsKeyConfigured && odds == null ? 'No odds have been pulled for this game yet' : undefined

  const runLinePick = summary?.run_line_pick_side ?? null
  const totalPick = summary?.pick_type === 'over' || summary?.pick_type === 'under' ? summary.pick_type : null

  const homeWinProb = summary?.moneyline_probability ?? null
  const awayPct = homeWinProb != null ? Math.round((1 - homeWinProb) * 100) : null
  const homePct = homeWinProb != null ? Math.round(homeWinProb * 100) : null
  const favoredIsHome = homePct != null && awayPct != null ? homePct >= awayPct : null

  // NRFI call: only surfaced when the model clears NRFI_YRFI_THRESHOLD in
  // either direction - see the constant's own comment for why. `pct` is
  // always the winning direction's own confidence, never the raw
  // (potentially < 50%) nrfiProb.
  const nrfiCall =
    nrfiProb != null && nrfiProb >= NRFI_YRFI_THRESHOLD
      ? { label: 'NRFI', pct: Math.round(nrfiProb * 100) }
      : nrfiProb != null && nrfiProb <= 1 - NRFI_YRFI_THRESHOLD
        ? { label: 'YRFI', pct: Math.round((1 - nrfiProb) * 100) }
        : null

  const eraWhip = (era: number | null | undefined, whip: number | null | undefined) =>
    era != null || whip != null ? `ERA ${era != null ? era.toFixed(2) : '—'} · WHIP ${whip != null ? whip.toFixed(2) : '—'}` : null

  return (
    <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-start gap-4">
        <div className="w-36 shrink-0 pt-1 space-y-1.5">
          <div>
            <div className="font-bold text-sm leading-tight">{game.away_team.abbreviation}</div>
            {summary?.away_starter_name && (
              <div className="text-[10px] text-[color:var(--color-ink-muted)] leading-tight truncate" title={summary.away_starter_name}>
                {summary.away_starter_name}
              </div>
            )}
            {eraWhip(summary?.away_starter_era, summary?.away_starter_whip) && (
              <div className="text-[10px] text-[color:var(--color-ink-faint)] leading-tight">
                {eraWhip(summary?.away_starter_era, summary?.away_starter_whip)}
              </div>
            )}
          </div>
          <div className="text-[10px] text-[color:var(--color-ink-faint)] leading-tight uppercase">at</div>
          <div>
            <div className="font-bold text-sm leading-tight">{game.home_team.abbreviation}</div>
            {summary?.home_starter_name && (
              <div className="text-[10px] text-[color:var(--color-ink-muted)] leading-tight truncate" title={summary.home_starter_name}>
                {summary.home_starter_name}
              </div>
            )}
            {eraWhip(summary?.home_starter_era, summary?.home_starter_whip) && (
              <div className="text-[10px] text-[color:var(--color-ink-faint)] leading-tight">
                {eraWhip(summary?.home_starter_era, summary?.home_starter_whip)}
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                {COLUMN_HEADERS.map((h) => (
                  <th
                    key={h}
                    className="text-center text-[11px] font-semibold uppercase tracking-wide pb-2 px-1 text-[color:var(--color-ink-faint)]"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <Cell highlighted={false} title={oddsMissingTitle}>
                  {game.away_team.abbreviation} {awayScore != null ? awayScore.toFixed(1) : '—'} {americanOdds(odds?.moneyline_away)}
                </Cell>
                <Cell highlighted={runLinePick === 'away'} title={oddsMissingTitle}>
                  {/* Only the home side's run-line price is actually ingested (see
                      ingestion/odds_api._extract_best_lines) - showing a fabricated
                      away-side price would be dishonest, so this just omits it. */}
                  {awayRunLine != null ? signedNumber(awayRunLine) : '—'}
                </Cell>
                <Cell highlighted={totalPick === 'under'} title={oddsMissingTitle}>
                  {total != null ? `U ${total} (${americanOdds(odds?.under_odds)})` : '—'}
                </Cell>
              </tr>
              <tr>
                <Cell highlighted={false} title={oddsMissingTitle}>
                  {game.home_team.abbreviation} {homeScore != null ? homeScore.toFixed(1) : '—'} {americanOdds(odds?.moneyline_home)}
                </Cell>
                <Cell highlighted={runLinePick === 'home'} title={oddsMissingTitle}>
                  {homeRunLine != null ? `${signedNumber(homeRunLine)} (${americanOdds(odds?.run_line_odds)})` : '—'}
                </Cell>
                <Cell highlighted={totalPick === 'over'} title={oddsMissingTitle}>
                  {total != null ? `O ${total} (${americanOdds(odds?.over_odds)})` : '—'}
                </Cell>
              </tr>
            </tbody>
          </table>

          {modelTotal != null && (
            <div className="text-xs text-[color:var(--color-ink-muted)] mt-2 px-1">
              {total != null ? (
                <>
                  Model <span className="font-semibold text-[color:var(--color-ink)]">{modelTotal.toFixed(1)}</span> vs line {total} →{' '}
                  <span className="font-semibold text-[color:var(--color-ink)]">
                    {modelTotal > total ? 'OVER' : 'UNDER'} by {Math.abs(modelTotal - total).toFixed(1)}
                  </span>
                </>
              ) : (
                <>
                  Model total: <span className="font-semibold text-[color:var(--color-ink)]">{modelTotal.toFixed(1)}</span>
                </>
              )}
            </div>
          )}

          {homePct != null && awayPct != null && favoredIsHome != null && (() => {
            const favoredAbbr = favoredIsHome ? game.home_team.abbreviation : game.away_team.abbreviation
            const favoredPct = favoredIsHome ? homePct : awayPct
            const underdogPct = favoredIsHome ? awayPct : homePct
            const confidence = CONFIDENCE_BANDS.find((b) => favoredPct / 100 >= b.min)?.label ?? 'Near toss-up'
            const awaySegmentClass = favoredIsHome ? 'bg-[color:var(--color-ink-faint)]' : 'bg-[color:var(--color-home)]'
            const homeSegmentClass = favoredIsHome ? 'bg-[color:var(--color-home)]' : 'bg-[color:var(--color-ink-faint)]'
            return (
              <div className="mt-3 px-1">
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="text-sm font-bold text-[color:var(--color-home)]">
                    {favoredAbbr} {favoredPct}%
                  </span>
                  <span className="text-xs text-[color:var(--color-ink-faint)]">vs {underdogPct}%</span>
                  <span
                    className="text-[10px] uppercase tracking-wide text-[color:var(--color-ink-faint)] ml-auto"
                    title="How confident the model is in this pick - not an expected win rate"
                  >
                    {confidence}
                  </span>
                  {nrfiCall && (
                    <span
                      className="text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded bg-[color:var(--color-nrfi-soft)] text-[color:var(--color-nrfi)]"
                      title={`${nrfiCall.label} calls are only shown at ${Math.round(NRFI_YRFI_THRESHOLD * 100)}%+ confidence - the model validates at roughly a coin flip otherwise, so lower-confidence games show nothing here rather than a misleading pick`}
                    >
                      {nrfiCall.label} {nrfiCall.pct}%
                    </span>
                  )}
                </div>
                <div className="h-1.5 rounded-full overflow-hidden bg-[color:var(--color-border)] flex">
                  <div className={`h-full ${awaySegmentClass}`} style={{ width: `${awayPct}%` }} />
                  <div className={`h-full ${homeSegmentClass}`} style={{ width: `${homePct}%` }} />
                </div>
              </div>
            )
          })()}
        </div>

        <Link
          to={`/games/${game.id}`}
          className="text-xs font-medium text-[color:var(--color-ink-muted)] hover:text-[color:var(--color-ink)] hover:underline whitespace-nowrap shrink-0 pt-1"
        >
          View detail →
        </Link>
      </div>

      <div className="text-xs text-[color:var(--color-ink-faint)] mt-3">
        {game.venue?.name ?? 'TBD'} · <span className={statusStyle[game.status]}>{scoreLabel(game)}</span>
      </div>
    </div>
  )
}

function Cell({ highlighted, title, children }: { highlighted: boolean; title?: string; children: React.ReactNode }) {
  return (
    <td className="px-1 py-1.5">
      <div
        title={title}
        className={`rounded-lg text-center font-semibold text-xs py-1.5 px-2 whitespace-nowrap border ${
          highlighted
            ? 'bg-[color:var(--color-home-soft)] text-[color:var(--color-home)] border-[color:var(--color-home)]'
            : 'bg-[color:var(--color-surface-raised)] text-[color:var(--color-ink)] border-transparent'
        }`}
      >
        {children}
      </div>
    </td>
  )
}
