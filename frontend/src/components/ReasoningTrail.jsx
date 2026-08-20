import { CheckCircle2 } from 'lucide-react'

/** The audit trail: every check the system ran, in the order it ran them. */
export default function ReasoningTrail({ steps = [] }) {
  if (!steps.length) return null
  return (
    <ol className="space-y-2.5">
      {steps.map((s, i) => (
        <li key={i} className="flex gap-2.5">
          <div className="flex flex-col items-center">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
            {i < steps.length - 1 ? (
              <span className="mt-0.5 w-px flex-1" style={{ background: 'var(--gridline)' }} aria-hidden />
            ) : null}
          </div>
          <div className="min-w-0 pb-1">
            <div className="text-xs font-semibold uppercase tracking-wide text-ink-muted">{s.step}</div>
            <p className="mt-0.5 text-sm text-ink-secondary">{s.detail}</p>
          </div>
        </li>
      ))}
    </ol>
  )
}
