import { Link } from 'react-router-dom'
import type { Game, GameSlateSummary } from '../api/types'
import { TeamBadge } from './TeamBadge'

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

interface Props {
  game: Game
  summary?: GameSlateSummary
}

export function GameRow({ game, summary }: Props) {
  const homeProb = summary?.moneyline_probability ?? null
  const totalSplit = summary?.total_home_prediction != null && summary?.total_away_prediction != null
    ? `${game.home_team.abbreviation} ${summary.total_home_prediction.toFixed(1)} · ${game.away_team.abbreviation} ${summary.total_away_prediction.toFixed(1)}`
    : summary?.total_prediction != null
      ? summary.total_prediction.toFixed(1)
      : '—'
  const recommendation = summary?.pick_type && summary.pick_side ? `${summary.pick_type.toUpperCase()} ${summary.pick_side.toUpperCase()}` : '—'
  const confidence = summary?.confidence != null ? `${Math.round(summary.confidence * 100)}%` : '—'

  const odds = summary?.latest_odds ?? null
  const moneylineValue = odds ? `${americanOdds(odds.moneyline_home)} / ${americanOdds(odds.moneyline_away)}` : '—'
  const spreadValue = odds?.run_line != null
    ? `${game.home_team.abbreviation} ${odds.run_line > 0 ? `+${odds.run_line}` : odds.run_line} (${americanOdds(odds.run_line_odds)})`
    : '—'
  const totalOddsValue = odds?.total != null
    ? `${odds.total} (O ${americanOdds(odds.over_odds)} / U ${americanOdds(odds.under_odds)})`
    : '—'

  return (
    <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-6 py-4 flex flex-wrap items-center gap-6 hover:shadow-sm transition-shadow">
      <div className="min-w-[220px]">
        <div className="flex items-center gap-2 font-semibold">
          <TeamBadge abbr={game.away_team.abbreviation} />
          <span>{game.away_team.abbreviation}</span>
          <span className="text-[color:var(--color-ink-faint)] font-normal">@</span>
          <TeamBadge abbr={game.home_team.abbreviation} />
          <span>{game.home_team.abbreviation}</span>
        </div>
        <div className="text-xs text-[color:var(--color-ink-faint)] mt-1">
          {game.venue?.name ?? 'TBD'} ·{' '}
          <span className={statusStyle[game.status]}>{scoreLabel(game)}</span>
        </div>
      </div>

      <div className="flex-1 flex flex-wrap items-center gap-x-8 gap-y-2">
        <Stat label={`${game.home_team.abbreviation} WIN %`} value={homeProb != null ? `${Math.round(homeProb * 100)}%` : '—'} />
        <Stat label={`${game.away_team.abbreviation} WIN %`} value={homeProb != null ? `${Math.round((1 - homeProb) * 100)}%` : '—'} />
        <Stat label="TOTAL SPLIT" value={totalSplit} />
        <Stat label="RECOMMENDATION" value={recommendation} />
        <Stat label="EDGE CONFIDENCE" value={confidence} />
        <Stat label="MONEYLINE (H/A)" value={moneylineValue} />
        <Stat label={`SPREAD (${game.home_team.abbreviation})`} value={spreadValue} />
        <Stat label="TOTAL ODDS" value={totalOddsValue} />
      </div>

      <Link to={`/games/${game.id}`} className="text-sm font-medium text-[color:var(--color-home)] hover:underline whitespace-nowrap">
        View detail →
      </Link>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] font-medium text-[color:var(--color-ink-faint)] tracking-wide">{label}</div>
      <div className="text-base font-semibold">{value}</div>
    </div>
  )
}
