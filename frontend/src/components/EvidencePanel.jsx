import { FileText, Minus, Quote, Sigma, ThumbsDown, ThumbsUp } from 'lucide-react'
import { formatDelta, formatValue } from '../lib/format'

const STANCE = {
  supporting: { icon: ThumbsUp, color: 'var(--status-good)', label: 'Supports' },
  contradicting: { icon: ThumbsDown, color: 'var(--status-critical)', label: 'Contradicts' },
  neutral: { icon: Minus, color: 'var(--text-muted)', label: 'Context' },
}

export function StructuredEvidence({ items = [], showWeights = false }) {
  if (!items.length) return <p className="text-sm text-ink-muted">No structured tests were available for this explanation.</p>
  return (
    <ul className="space-y-2">
      {items.map((e, i) => {
        const stance = STANCE[e.stance] || STANCE.neutral
        const Icon = stance.icon
        return (
          <li
            key={`${e.metric || e.label}-${i}`}
            className="rounded-lg border px-3 py-2"
            style={{ borderColor: 'var(--border)' }}
          >
            <div className="flex items-start gap-2">
              <Icon className="mt-0.5 h-4 w-4 shrink-0" style={{ color: stance.color }} aria-hidden />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <span className="text-sm font-medium text-ink">{e.label}</span>
                  <span className="text-[11px] uppercase tracking-wide" style={{ color: stance.color }}>
                    {stance.label}
                  </span>
                  {e.scope && e.scope !== 'whole business' ? (
                    <span className="text-[11px] text-ink-muted">in {e.scope}</span>
                  ) : null}
                </div>
                <p className="mt-0.5 text-sm text-ink-secondary">{e.detail}</p>
                {e.note ? <p className="mt-0.5 text-xs text-ink-muted">{e.note}</p> : null}
                {showWeights && e.strength !== undefined ? (
                  <p className="mt-1 text-[11px] text-ink-muted tnum">
                    strength {e.strength} × weight {e.weight}
                    {e.change_pct !== undefined && e.change_pct !== null
                      ? ` · measured ${formatDelta(e.change_pct)}`
                      : ''}
                  </p>
                ) : null}
              </div>
              <Sigma className="h-3.5 w-3.5 shrink-0 text-ink-muted" aria-label="from structured data" />
            </div>
          </li>
        )
      })}
    </ul>
  )
}

export function DocumentEvidence({ items = [], stance = 'supporting', emptyMessage }) {
  if (!items.length) {
    return <p className="text-sm text-ink-muted">{emptyMessage || 'No document passages were retrieved.'}</p>
  }
  const color = stance === 'contradicting' ? 'var(--status-critical)' : 'var(--status-good)'
  return (
    <ul className="space-y-2">
      {items.map((d, i) => (
        <li
          key={d.chunk_id || i}
          className="rounded-lg border px-3 py-2"
          style={{ borderColor: 'var(--border)', borderLeft: `3px solid ${color}` }}
        >
          <div className="flex items-start gap-2">
            <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-muted" aria-hidden />
            <div className="min-w-0 flex-1">
              <p className="text-sm italic text-ink-secondary">“{d.quote}”</p>
              <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-ink-muted">
                <span className="inline-flex items-center gap-1 font-medium text-ink-secondary">
                  <FileText className="h-3 w-3" aria-hidden />
                  {d.source}
                </span>
                {d.section ? <span>§ {d.section}</span> : null}
                {d.doc_type ? <span>· {d.doc_type.replace(/_/g, ' ')}</span> : null}
                {d.classified_by ? <span>· stance via {d.classified_by.replace(/_/g, ' ')}</span> : null}
              </div>
            </div>
          </div>
        </li>
      ))}
    </ul>
  )
}

export function MissingEvidence({ items = [] }) {
  if (!items.length) return null
  return (
    <ul className="space-y-1.5">
      {items.map((m, i) => (
        <li key={i} className="flex gap-2 text-sm text-ink-secondary">
          <span aria-hidden style={{ color: 'var(--status-warning)' }}>?</span>
          <span>{m}</span>
        </li>
      ))}
    </ul>
  )
}
