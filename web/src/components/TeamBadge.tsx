import { teamColor } from '../lib/teamColors'

export function TeamBadge({ abbr, size = 28 }: { abbr: string; size?: number }) {
  return (
    <span
      className="inline-flex items-center justify-center rounded-full font-bold text-white shrink-0"
      style={{ background: teamColor(abbr), width: size, height: size, fontSize: size * 0.36 }}
    >
      {abbr}
    </span>
  )
}
