export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-16 justify-center text-[color:var(--color-ink-muted)]">
      <span className="h-4 w-4 rounded-full border-2 border-[color:var(--color-border)] border-t-[color:var(--color-home)] animate-spin" />
      {label}
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div
      className="rounded-xl border px-4 py-3 text-sm"
      style={{ borderColor: 'var(--color-critical)', background: 'var(--color-critical-soft)', color: 'var(--color-critical)' }}
    >
      {message}
    </div>
  )
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed border-[color:var(--color-border)] px-4 py-10 text-center text-sm text-[color:var(--color-ink-muted)]">
      {message}
    </div>
  )
}
