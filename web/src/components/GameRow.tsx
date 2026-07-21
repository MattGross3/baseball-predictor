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

const COLUMN_HEADERS = ['Score / ML', 'Spread / Run Line', 'Total / O-U', 'NRFI'] as const

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

  const homeWinProb = summary?.moneyline_probability ?? null
  const awayPct = homeWinProb != null ? Math.round((1 - homeWinProb) * 100) : null
  const homePct = homeWinProb != null ? Math.round(homeWinProb * 100) : null

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
                <Cell highlighted={false}>
                  {game.away_team.abbreviation} {awayScore != null ? awayScore.toFixed(1) : '—'} {americanOdds(odds?.moneyline_away)}
                </Cell>
                <Cell highlighted={runLinePick === 'away'}>
                  {/* Only the home side's run-line price is actually ingested (see
                      ingestion/odds_api._extract_best_lines) - showing a fabricated
                      away-side price would be dishonest, so this just omits it. */}
                  {awayRunLine != null ? signedNumber(awayRunLine) : '—'}
                </Cell>
                <Cell highlighted={totalPick === 'under'}>
                  {total != null ? `U ${total} (${americanOdds(odds?.under_odds)})` : '—'}
                </Cell>
                <Cell highlighted={nrfiProb != null && nrfiProb < 0.5}>No</Cell>
              </tr>
              <tr>
                <Cell highlighted={false}>
                  {game.home_team.abbreviation} {homeScore != null ? homeScore.toFixed(1) : '—'} {americanOdds(odds?.moneyline_home)}
                </Cell>
                <Cell highlighted={runLinePick === 'home'}>
                  {homeRunLine != null ? `${signedNumber(homeRunLine)} (${americanOdds(odds?.run_line_odds)})` : '—'}
                </Cell>
                <Cell highlighted={totalPick === 'over'}>
                  {total != null ? `O ${total} (${americanOdds(odds?.over_odds)})` : '—'}
                </Cell>
                <Cell highlighted={nrfiProb != null && nrfiProb >= 0.5}>Yes</Cell>
              </tr>
            </tbody>
          </table>

          {homePct != null && awayPct != null && (
            <div className="flex items-center gap-3 mt-3 px-1">
              <div className="flex-1 h-1.5 rounded-full overflow-hidden bg-[color:var(--color-border)] flex">
                <div className="h-full bg-[color:var(--color-ink-faint)]" style={{ width: `${awayPct}%` }} />
                <div className="h-full bg-[color:var(--color-home)]" style={{ width: `${homePct}%` }} />
              </div>
              <span className="text-xs font-medium text-[color:var(--color-ink-muted)] shrink-0 tabular-nums">
                {awayPct}% / {homePct}%
              </span>
            </div>
          )}
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

function Cell({ highlighted, children }: { highlighted: boolean; children: React.ReactNode }) {
  return (
    <td className="px-1 py-1.5">
      <div
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
