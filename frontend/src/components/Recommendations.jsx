import { AlertCircle, Bell, CircleDot, Target, User } from 'lucide-react'
import { Badge, Callout } from './ui'

const PRIORITY_TONE = { high: 'critical', medium: 'warning', low: 'neutral' }

export default function Recommendations({ recommendations = [], limits = [] }) {
  if (!recommendations.length) {
    return (
      <Callout tone="warning" title="No action recommended on this number alone">
        No explanation reached a level of evidence that would justify acting on it. That is a finding, not a failure —
        it usually means the change is inside normal variation, or that the data available cannot separate the
        competing explanations.
      </Callout>
    )
  }

  return (
    <div className="space-y-4">
      {recommendations.map((rec) => (
        <article key={rec.id} className="card card-pad">
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="flex min-w-0 items-start gap-2.5">
              <Target className="mt-0.5 h-4 w-4 shrink-0" style={{ color: 'var(--series-1)' }} aria-hidden />
              <h3 className="text-sm font-semibold text-ink sm:text-base">{rec.title}</h3>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={PRIORITY_TONE[rec.priority] || 'neutral'}>{rec.priority} priority</Badge>
              <Badge tone="neutral">{rec.horizon}</Badge>
            </div>
          </div>

          <p className="mt-2 text-sm text-ink-secondary">{rec.rationale}</p>

          <ul className="mt-3 space-y-1.5">
            {rec.actions.map((a, i) => (
              <li key={i} className="flex gap-2 text-sm text-ink">
                <CircleDot className="mt-1 h-3 w-3 shrink-0 text-ink-muted" aria-hidden />
                <span>{a}</span>
              </li>
            ))}
          </ul>

          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-muted">
            <span className="inline-flex items-center gap-1">
              <User className="h-3 w-3" aria-hidden />
              {rec.owner}
            </span>
            <span>
              Based on: {rec.based_on.hypothesis}{rec.based_on.confidence == null ? ' — confidence not assessed' : ` (${rec.based_on.confidence}% ${rec.based_on.band})`}
            </span>
          </div>

          {rec.supporting_evidence?.length ? (
            <div className="mt-3 rounded-lg px-3 py-2 text-xs" style={{ background: 'var(--plane)' }}>
              <div className="mb-1 font-semibold uppercase tracking-wide text-ink-muted">Evidence behind this</div>
              <ul className="space-y-0.5 text-ink-secondary">
                {rec.supporting_evidence.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
                {rec.documentary_evidence?.map((e, i) => (
                  <li key={`d${i}`} className="italic">{e}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {rec.monitoring?.length ? (
            <div className="mt-3">
              <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                <Bell className="h-3 w-3" aria-hidden />
                Early-warning thresholds, from your own history
              </div>
              <ul className="space-y-1 text-sm text-ink-secondary">
                {rec.monitoring.map((m) => (
                  <li key={m.metric}>{m.rule}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {rec.what_would_change_this?.length ? (
            <div className="mt-3">
              <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                <AlertCircle className="h-3 w-3" aria-hidden />
                What would change this advice
              </div>
              <ul className="space-y-1 text-sm text-ink-secondary">
                {rec.what_would_change_this.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </article>
      ))}

      {limits.length ? (
        <div className="card card-pad">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">Limits of this analysis</div>
          <ul className="space-y-1 text-sm text-ink-secondary">
            {limits.map((l, i) => (
              <li key={i}>· {l}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}
