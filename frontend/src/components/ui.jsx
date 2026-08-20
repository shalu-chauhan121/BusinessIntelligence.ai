import { AlertTriangle, Info, Loader2, Lock } from 'lucide-react'

export function Spinner({ label = 'Loading…', className = '' }) {
  return (
    <div className={`flex items-center gap-2 text-sm text-ink-secondary ${className}`}>
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      <span>{label}</span>
    </div>
  )
}

export function LoadingCard({ label = 'Loading…', lines = 3 }) {
  return (
    <div className="card card-pad" role="status" aria-live="polite">
      <Spinner label={label} />
      <div className="mt-4 space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className="h-3 animate-pulse rounded"
            style={{ background: 'var(--gridline)', width: `${100 - i * 12}%` }}
          />
        ))}
      </div>
    </div>
  )
}

export function ErrorState({ title = 'Something went wrong', message, onRetry, action }) {
  return (
    <div className="card card-pad" role="alert">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" style={{ color: 'var(--status-critical)' }} aria-hidden />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-ink">{title}</h3>
          {message ? <p className="mt-1 text-sm text-ink-secondary">{message}</p> : null}
          <div className="mt-3 flex flex-wrap gap-2">
            {onRetry ? (
              <button type="button" className="btn-ghost" onClick={onRetry}>
                Try again
              </button>
            ) : null}
            {action}
          </div>
        </div>
      </div>
    </div>
  )
}

export function EmptyState({ icon: Icon = Info, title, message, action }) {
  return (
    <div className="card card-pad text-center">
      <div
        className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full"
        style={{ background: 'var(--plane)' }}
      >
        <Icon className="h-5 w-5 text-ink-muted" aria-hidden />
      </div>
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {message ? <p className="mx-auto mt-1 max-w-md text-sm text-ink-secondary">{message}</p> : null}
      {action ? <div className="mt-4 flex justify-center gap-2">{action}</div> : null}
    </div>
  )
}

export function SectionTitle({ eyebrow, title, description, right }) {
  return (
    <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        {eyebrow ? (
          <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-muted">{eyebrow}</div>
        ) : null}
        <h2 className="text-base font-semibold text-ink sm:text-lg">{title}</h2>
        {description ? <p className="mt-1 max-w-2xl text-sm text-ink-secondary">{description}</p> : null}
      </div>
      {right}
    </div>
  )
}

/**
 * Analyst-only block.
 *
 * This is a presentation convenience only — the corresponding fields are also
 * removed from the API response for non-analyst roles, so hiding here is not
 * the security boundary.
 */
export function AnalystOnly({ show, title = 'Analyst detail', children, defaultOpen = false }) {
  if (!show) return null
  return (
    <details className="mt-3 rounded-lg border" style={{ borderColor: 'var(--border)' }} open={defaultOpen}>
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
        <Lock className="mr-1.5 inline h-3 w-3 align-[-2px]" aria-hidden />
        {title}
      </summary>
      <div className="border-t px-3 py-3 text-sm" style={{ borderColor: 'var(--border)' }}>
        {children}
      </div>
    </details>
  )
}

export function Callout({ tone = 'info', title, children }) {
  const colors = {
    info: 'var(--series-1)',
    warning: 'var(--status-warning)',
    critical: 'var(--status-critical)',
    good: 'var(--status-good)',
  }
  return (
    <div
      className="rounded-lg border-l-4 px-3 py-2.5 text-sm"
      style={{ borderColor: colors[tone], background: 'var(--plane)' }}
    >
      {title ? <div className="font-semibold text-ink">{title}</div> : null}
      <div className="text-ink-secondary">{children}</div>
    </div>
  )
}

export function Badge({ tone = 'neutral', children, icon: Icon }) {
  const map = {
    neutral: { color: 'var(--text-secondary)', bg: 'var(--plane)' },
    good: { color: 'var(--status-good)', bg: 'color-mix(in srgb, var(--status-good) 12%, transparent)' },
    warning: { color: 'var(--status-warning)', bg: 'color-mix(in srgb, var(--status-warning) 16%, transparent)' },
    serious: { color: 'var(--status-serious)', bg: 'color-mix(in srgb, var(--status-serious) 16%, transparent)' },
    critical: { color: 'var(--status-critical)', bg: 'color-mix(in srgb, var(--status-critical) 14%, transparent)' },
  }
  const s = map[tone] || map.neutral
  return (
    <span className="chip" style={{ color: s.color, background: s.bg, borderColor: 'transparent' }}>
      {Icon ? <Icon className="h-3.5 w-3.5" aria-hidden /> : null}
      {children}
    </span>
  )
}
