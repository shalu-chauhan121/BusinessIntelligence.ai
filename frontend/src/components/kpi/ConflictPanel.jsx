import { useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, ShieldAlert } from 'lucide-react'
import { Badge, Callout, SectionTitle } from '../ui'

const SEVERITY = {
  blocking: { tone: 'critical', icon: ShieldAlert, label: 'Blocking' },
  warning: { tone: 'warning', icon: AlertTriangle, label: 'Warning' },
  info: { tone: 'neutral', icon: Info, label: 'Note' },
}

/**
 * Ambiguities the system refused to settle on its own.
 *
 * Every resolution asks for a rationale, because the audit trail has to explain
 * not just what the definition is but why it is that one and not the alternative.
 */
export default function ConflictPanel({ conflicts = [], onResolve, canManage = false, busy = false }) {
  if (!conflicts.length) {
    return (
      <Callout tone="good" title="No unresolved conflicts">
        Nothing in this contract is contradictory or ambiguous enough to need a human decision.
      </Callout>
    )
  }

  const open = conflicts.filter((c) => !c.resolved)
  const resolved = conflicts.filter((c) => c.resolved)

  return (
    <section>
      <SectionTitle
        title="Conflicts and ambiguities"
        description="The system flags these rather than choosing. A blocking conflict prevents approval until someone decides."
      />
      <div className="space-y-3">
        {open.map((conflict) => (
          <ConflictCard
            key={conflict.conflict_id}
            conflict={conflict}
            onResolve={onResolve}
            canManage={canManage}
            busy={busy}
          />
        ))}
        {resolved.map((conflict) => (
          <div key={conflict.conflict_id} className="card card-pad opacity-70">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone="good" icon={CheckCircle2}>
                Resolved
              </Badge>
              <span className="text-xs text-ink-muted">{conflict.kind.replace(/_/g, ' ')}</span>
            </div>
            <p className="mt-2 text-sm text-ink-secondary">{conflict.detail}</p>
            {conflict.resolution_rationale ? (
              <p className="mt-1 text-xs text-ink-muted">
                {conflict.resolved_by} chose “{conflict.resolution_option_id}”:{' '}
                {conflict.resolution_rationale}
              </p>
            ) : null}
          </div>
        ))}
      </div>
    </section>
  )
}

function ConflictCard({ conflict, onResolve, canManage, busy }) {
  const severity = SEVERITY[conflict.severity] || SEVERITY.info
  const Icon = severity.icon
  const [option, setOption] = useState(conflict.resolution_options?.[0]?.option_id || '')
  const [rationale, setRationale] = useState('')

  return (
    <div className="card card-pad">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={severity.tone} icon={Icon}>
          {severity.label}
        </Badge>
        <span className="text-xs uppercase tracking-wider text-ink-muted">
          {conflict.kind.replace(/_/g, ' ')}
        </span>
      </div>

      <p className="mt-2 text-sm text-ink">{conflict.detail}</p>

      {conflict.affected_kpis?.length ? (
        <p className="mt-1 text-xs text-ink-muted">
          Affects: {conflict.affected_kpis.join(', ')}
        </p>
      ) : null}
      {conflict.affected_fields?.length ? (
        <p className="mt-1 text-xs text-ink-muted">
          Columns: {conflict.affected_fields.join(', ')}
        </p>
      ) : null}

      {canManage && conflict.resolution_options?.length ? (
        <div className="mt-3 space-y-2">
          <label className="label" htmlFor={`opt-${conflict.conflict_id}`}>
            How should this be resolved?
          </label>
          <select
            id={`opt-${conflict.conflict_id}`}
            className="field"
            value={option}
            onChange={(e) => setOption(e.target.value)}
          >
            {conflict.resolution_options.map((o) => (
              <option key={o.option_id} value={o.option_id}>
                {o.label}
                {o.detail ? ` — ${o.detail}` : ''}
              </option>
            ))}
          </select>
          <label className="label" htmlFor={`why-${conflict.conflict_id}`}>
            Why (kept in the provenance of every KPI this touches)
          </label>
          <input
            id={`why-${conflict.conflict_id}`}
            className="field"
            value={rationale}
            placeholder="e.g. Finance owns this definition for statutory reporting."
            onChange={(e) => setRationale(e.target.value)}
          />
          <button
            type="button"
            className="btn-primary"
            disabled={busy || !option || !rationale.trim()}
            onClick={() => onResolve(conflict.conflict_id, option, rationale.trim())}
          >
            Record this decision
          </button>
        </div>
      ) : null}
    </div>
  )
}
