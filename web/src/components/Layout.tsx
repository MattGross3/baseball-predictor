import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/', label: "Today's Slate", end: true },
  { to: '/previous-games', label: 'Previous Games' },
  { to: '/backtest', label: 'Backtest' },
  { to: '/compare', label: 'Model Comparison' },
]

export function Layout() {
  return (
    <div className="min-h-screen flex">
      <aside className="w-64 shrink-0 border-r border-[color:var(--color-border)] flex flex-col justify-between py-6 px-4">
        <div>
          <div className="px-2 mb-8">
            <div className="font-bold text-lg tracking-tight">⚾ Baseball Predictor</div>
          </div>
          <nav className="flex flex-col gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-[color:var(--color-home-soft)] text-[color:var(--color-home)]'
                      : 'text-[color:var(--color-ink-muted)] hover:bg-[color:var(--color-surface-raised)] hover:text-[color:var(--color-ink)]'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <p className="px-2 text-xs text-[color:var(--color-ink-faint)]">
          Predictions are model estimates, not betting advice.
        </p>
      </aside>
      <main className="flex-1 px-10 py-8 max-w-6xl">
        <Outlet />
      </main>
    </div>
  )
}
