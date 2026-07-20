import { Link } from 'react-router-dom'
import type { Game, Prediction } from '../api/types'
import { preferredPrediction } from '../lib/predictions'
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

interface Props {
  game: Game
  predictions: Prediction[]
}

export function GameRow({ game, predictions }: Props) {
  const moneyline = preferredPrediction(predictions, 'moneyline')
  const total = preferredPrediction(predictions, 'total')
  const nrfi = preferredPrediction(predictions, 'nrfi')
  const homeProb = moneyline?.predicted_probability ?? null

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
        <Stat label="PREDICTED TOTAL" value={total?.predicted_value != null ? total.predicted_value.toFixed(1) : '—'} />
        <Stat label="NRFI %" value={nrfi?.predicted_probability != null ? `${Math.round(nrfi.predicted_probability * 100)}%` : '—'} />
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
