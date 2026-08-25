import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  BarChart3,
  Database,
  FileText,
  History,
  LayoutDashboard,
  Ruler,
  LogOut,
  Menu,
  Moon,
  Search,
  Settings,
  Sun,
  X,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { useTheme } from '../context/ThemeContext'

const NAV = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/investigation', label: 'Investigation', icon: Search },
  { to: '/history', label: 'History', icon: History },
  { to: '/kpi', label: 'KPI contract', icon: Ruler },
  { to: '/data', label: 'Business data', icon: Database },
  { to: '/documents', label: 'Documents', icon: FileText },
  { to: '/settings', label: 'Settings', icon: Settings },
]

const ROLE_LABEL = { data_analyst: 'Data Analyst', business_leader: 'Business Leader' }

export default function Layout({ children }) {
  const { profile, logout } = useAuth()
  const { theme, toggle } = useTheme()
  const [open, setOpen] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen">
      {/* top bar */}
      <header
        className="sticky top-0 z-30 border-b backdrop-blur"
        style={{ borderColor: 'var(--border)', background: 'color-mix(in srgb, var(--surface-1) 88%, transparent)' }}
      >
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-3 px-3 sm:px-5">
          <button
            type="button"
            className="btn-ghost !px-2 lg:hidden"
            onClick={() => setOpen((v) => !v)}
            aria-label={open ? 'Close navigation' : 'Open navigation'}
            aria-expanded={open}
          >
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>

          <div className="flex min-w-0 items-center gap-2">
            <span
              className="flex h-7 w-7 items-center justify-center rounded-md"
              style={{ background: 'var(--series-1)' }}
            >
              <BarChart3 className="h-4 w-4 text-white" aria-hidden />
            </span>
            <span className="truncate text-sm font-semibold tracking-tight text-ink">
              BusinessIntelligence<span style={{ color: 'var(--series-1)' }}>.ai</span>
            </span>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={toggle}
              className="btn-ghost !px-2"
              aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            >
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            {profile ? (
              <div className="hidden items-center gap-2 sm:flex">
                <div className="text-right leading-tight">
                  <div className="max-w-[160px] truncate text-xs font-medium text-ink">
                    {profile.display_name || profile.email}
                  </div>
                  <div className="text-[11px] text-ink-muted">{ROLE_LABEL[profile.role]}</div>
                </div>
                <button type="button" onClick={logout} className="btn-ghost !px-2" aria-label="Sign out">
                  <LogOut className="h-4 w-4" />
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1400px] gap-5 px-3 py-4 sm:px-5">
        {/* sidebar */}
        <nav
          className={`${open ? 'block' : 'hidden'} fixed inset-x-0 top-14 z-20 border-b px-3 pb-3 pt-2 lg:static lg:block lg:w-56 lg:shrink-0 lg:border-0 lg:p-0`}
          style={{ borderColor: 'var(--border)', background: open ? 'var(--surface-1)' : 'transparent' }}
          aria-label="Main navigation"
        >
          <ul className="space-y-1 lg:sticky lg:top-20">
            {NAV.map(({ to, label, icon: Icon }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  onClick={() => setOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors ${
                      isActive ? 'font-semibold text-ink' : 'text-ink-secondary hover:text-ink'
                    }`
                  }
                  style={({ isActive }) => (isActive ? { background: 'var(--plane)' } : undefined)}
                >
                  <Icon className="h-4 w-4 shrink-0" aria-hidden />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <main className="min-w-0 flex-1 pb-16" key={location.pathname}>
          {children}
        </main>
      </div>
    </div>
  )
}
