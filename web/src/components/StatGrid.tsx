import type { ReactNode } from 'react'

export interface StatItem {
  label: string
  value: string | number | null | undefined
  suffix?: string
}

function formatValue(value: string | number | null | undefined, suffix?: string) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return `${Number.isInteger(value) ? value : value.toFixed(2)}${suffix ?? ''}`
  return `${value}${suffix ?? ''}`
}

/** Small labeled key/value tiles - the "nice" replacement for dumping a
 * feature dict as raw JSON. Purely presentational; callers decide which
 * fields are worth surfacing. */
export function StatGrid({ items }: { items: StatItem[] }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map((item) => (
        <div key={item.label} className="rounded-lg bg-[color:var(--color-surface-raised)] px-3 py-2">
          <div className="text-[11px] text-[color:var(--color-ink-faint)] leading-tight">{item.label}</div>
          <div className="text-sm font-semibold mt-0.5">{formatValue(item.value, item.suffix)}</div>
        </div>
      ))}
    </div>
  )
}

export function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-[color:var(--color-border)] bg-[color:var(--color-surface-card)] p-4">
      <h3 className="text-sm font-semibold text-[color:var(--color-ink-muted)] uppercase tracking-wide mb-3">
        {title}
      </h3>
      {children}
    </div>
  )
}
