interface Props {
  label: string
  value: string
  hint?: string
  tone?: 'good' | 'critical'
}

const toneColor: Record<string, string> = {
  good: 'text-[color:var(--color-good)]',
  critical: 'text-[color:var(--color-critical)]',
}

export function MetricCard({ label, value, hint, tone }: Props) {
  return (
    <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] px-4 py-3">
      <div className="text-xs text-[color:var(--color-ink-faint)]">{label}</div>
      <div className={`text-xl font-semibold mt-0.5 ${tone ? toneColor[tone] : ''}`}>{value}</div>
      {hint && <div className="text-[11px] text-[color:var(--color-ink-faint)] mt-0.5">{hint}</div>}
    </div>
  )
}
