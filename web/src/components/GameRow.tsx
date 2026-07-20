import { Link } from 'react-router-dom'
import type { Game, GameSlateSummary } from '../api/types'

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

const COLUMN_HEADERS = ['Predicted Score / ML', 'Spread', 'Total', 'NRFI'] as const

const headerColor: Record<(typeof COLUMN_HEADERS)[number], string> = {
  'Predicted Score / ML': 'var(--color-home)',
  Spread: 'var(--color-warning)',
  Total: 'var(--color-good)',
  NRFI: 'var(--color-nrfi)',
}

interface Props {
  game: Game
  summary?: GameSlateSummary
}

export function GameRow({ game, summary }: Props) {
  const odds = summary?.latest_odds ?? null

  const awayScore = summary?.total_away_prediction
  const homeScore = summary?.total_home_prediction

  // run_line is always quoted from the home team's perspective (see
  // ingestion/odds_api._extract_best_lines) - the away side's line is
  // just its negation, but we only ever ingest the home side's *price*,
  // so the away pill's odds are honestly "—" rather than a guessed number.
  const homeRunLine = odds?.run_line ?? null
  const awayRunLine = homeRunLine != null ? -homeRunLine : null

  const total = odds?.total ?? null
  const nrfiProb = summary?.nrfi_probability ?? null

  const runLinePick = summary?.run_line_pick_side ?? null
  const totalPick = summary?.pick_type === 'over' || summary?.pick_type === 'under' ? summary.pick_type : null

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
          <div className="text-[10px] text-[color:var(--color-ink-faint)] leading-tight">vs</div>
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
                    className="text-center text-[11px] font-semibold uppercase tracking-wide pb-2 px-1"
                    style={{ color: headerColor[h] }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <Cell color="var(--color-home)" highlighted={false}>
                  {game.away_team.abbreviation} {awayScore != null ? awayScore.toFixed(1) : '—'} {americanOdds(odds?.moneyline_away)}
                </Cell>
                <Cell color="var(--color-warning)" highlighted={runLinePick === 'away'}>
                  {/* Only the home side's run-line price is actually ingested (see
                      ingestion/odds_api._extract_best_lines) - showing a fabricated
                      away-side price would be dishonest, so this just omits it. */}
                  {awayRunLine != null ? signedNumber(awayRunLine) : '—'}
                </Cell>
                <Cell color="var(--color-good)" highlighted={totalPick === 'under'}>
                  {total != null ? `U ${total} ${americanOdds(odds?.under_odds)}` : '—'}
                </Cell>
                <Cell color="var(--color-nrfi)" highlighted={nrfiProb != null && nrfiProb < 0.5}>
                  No
                </Cell>
              </tr>
              <tr>
                <Cell color="var(--color-home)" highlighted={false}>
                  {game.home_team.abbreviation} {homeScore != null ? homeScore.toFixed(1) : '—'} {americanOdds(odds?.moneyline_home)}
                </Cell>
                <Cell color="var(--color-warning)" highlighted={runLinePick === 'home'}>
                  {homeRunLine != null ? `${signedNumber(homeRunLine)} (${americanOdds(odds?.run_line_odds)})` : '—'}
                </Cell>
                <Cell color="var(--color-good)" highlighted={totalPick === 'over'}>
                  {total != null ? `O ${total} ${americanOdds(odds?.over_odds)}` : '—'}
                </Cell>
                <Cell color="var(--color-nrfi)" highlighted={nrfiProb != null && nrfiProb >= 0.5}>
                  Yes
                </Cell>
              </tr>
            </tbody>
          </table>
        </div>

        <Link
          to={`/games/${game.id}`}
          className="text-xs font-medium text-[color:var(--color-home)] hover:underline whitespace-nowrap shrink-0 pt-1"
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

function Cell({ color, highlighted, children }: { color: string; highlighted: boolean; children: React.ReactNode }) {
  return (
    <td className="px-1 py-1.5">
      <div
        className="rounded-lg text-center font-semibold text-xs py-1.5 px-2 whitespace-nowrap"
        style={{
          color,
          background: `color-mix(in srgb, ${color} 12%, transparent)`,
          border: `1.5px solid ${highlighted ? color : 'transparent'}`,
        }}
      >
        {children}
      </div>
    </td>
  )
}
