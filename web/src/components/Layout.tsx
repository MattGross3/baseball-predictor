import { NavLink, Outlet } from 'react-router-dom'

const navItems = [
  { to: '/', label: "Today's Slate", end: true },
  { to: '/previous-games', label: 'Previous Games' },
  { to: '/backtest', label: 'Backtest' },
  { to: '/compare', label: 'Model Comparison' },
  { to: '/roi', label: 'Model ROI' },
  { to: '/models', label: 'Models' },
]

export function Layout() {
  return (
    <div className="min-h-screen flex">
      <aside className="w-64 shrink-0 bg-[color:var(--color-sidebar-bg)] flex flex-col justify-between py-6 px-4">
        <div>
          <div className="flex items-center gap-2.5 px-2 mb-8">
            <div className="w-8 h-8 rounded-lg bg-[color:var(--color-sidebar-accent)] flex items-center justify-center text-white font-bold text-sm shrink-0">
              B
            </div>
            <div className="min-w-0">
              <div className="font-bold text-sm tracking-tight text-[color:var(--color-sidebar-ink)] truncate">
                Baseball Predictor
              </div>
              <div className="text-[10px] uppercase tracking-wide text-[color:var(--color-sidebar-ink-muted)]">
                Live Predictions
              </div>
            </div>
          </div>
          <nav className="flex flex-col gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `px-3 py-2 rounded-lg text-sm font-medium transition-colors border-l-2 ${
                    isActive
                      ? 'bg-[color:var(--color-sidebar-active-bg)] text-[color:var(--color-sidebar-accent)] border-[color:var(--color-sidebar-accent)]'
                      : 'text-[color:var(--color-sidebar-ink)] border-transparent hover:bg-[color:var(--color-sidebar-active-bg)]'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <p className="px-2 text-xs text-[color:var(--color-sidebar-ink-muted)] border-t border-[color:var(--color-sidebar-border)] pt-4">
          Predictions are model estimates, not betting advice.
        </p>
      </aside>
      <main className="flex-1 px-10 py-8 max-w-6xl">
        <Outlet />
      </main>
    </div>
  )
}
