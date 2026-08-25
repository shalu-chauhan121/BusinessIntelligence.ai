import { useEffect, useState } from 'react'
import { Save, Trash2, X } from 'lucide-react'
import { titleCase } from '../../lib/format'
import { AnalystOnly, Badge, Callout } from '../ui'
import GranularityPicker from './GranularityPicker'
import KpiPreview from './KpiPreview'

const UNITS = ['currency', 'count', 'percent', 'ratio', 'duration', 'score']

/**
 * The full definition of one KPI, editable.
 *
 * Every field the contract carries is either editable here or shown as
 * provenance, because the point of the contract is that a KPI is not a mystery:
 * a business analyst can see exactly what it means, change it, and see who
 * changed it last.
 */
export default function KpiDefinitionEditor({
  kpi,
  dimensions = [],
  datasetId,
  isAnalyst = false,
  canManage = false,
  busy = false,
  onSave,
  onApprove,
  onReject,
  onDelete,
  onClose,
}) {
  const [form, setForm] = useState(() => initial(kpi))
  const [dirty, setDirty] = useState(false)

  useEffect(() => {
    setForm(initial(kpi))
    setDirty(false)
  }, [kpi.kpi_id])

  const set = (key) => (value) => {
    setForm((f) => ({ ...f, [key]: value }))
    setDirty(true)
  }
  const setEvent = (key) => (e) => set(key)(e.target.value)

  const isRatio = kpi.formula.kind === 'ratio'

  const save = () => {
    const patch = {
      name: form.name,
      business_definition: form.business_definition,
      relevance: form.relevance,
      unit: form.unit,
      higher_is_better: form.higher_is_better,
      granularity: form.granularity,
    }
    if (form.formulaChanged) {
      patch.formula = isRatio
        ? {
            ...kpi.formula,
            numerator_expression: form.numerator,
            denominator_expression: form.denominator,
            scale: Number(form.scale) || 1,
          }
        : { ...kpi.formula, expression: form.expression }
    }
    onSave(kpi.kpi_id, patch)
  }

  return (
    <div className="card card-pad">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="truncate text-sm font-semibold text-ink">{kpi.name}</h3>
            <StatusBadge status={kpi.status} />
            <span className="chip">{kpi.kpi_type.replace(/_/g, ' ')}</span>
          </div>
          <p className="mt-0.5 font-mono text-xs text-ink-muted">{kpi.kpi_id}</p>
        </div>
        <button type="button" className="btn-ghost !px-2" onClick={onClose} aria-label="Close editor">
          <X className="h-4 w-4" aria-hidden />
        </button>
      </div>

      {kpi.approval?.rejected_reason ? (
        <Callout tone="warning" title="Rejected">
          {kpi.approval.rejected_reason}
        </Callout>
      ) : null}

      <div className="mt-4 space-y-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label className="label" htmlFor="kpi-name">Name</label>
            <input id="kpi-name" className="field" value={form.name} onChange={setEvent('name')}
                   disabled={!canManage} />
          </div>
          <div>
            <label className="label" htmlFor="kpi-unit">Unit</label>
            <select id="kpi-unit" className="field" value={form.unit} onChange={setEvent('unit')}
                    disabled={!canManage}>
              {UNITS.map((u) => <option key={u} value={u}>{titleCase(u)}</option>)}
            </select>
          </div>
        </div>

        <div>
          <label className="label" htmlFor="kpi-def">Business definition</label>
          <textarea id="kpi-def" className="field" rows={2} value={form.business_definition}
                    onChange={setEvent('business_definition')} disabled={!canManage} />
        </div>

        <div>
          <label className="label" htmlFor="kpi-why">Why this KPI is relevant</label>
          <textarea id="kpi-why" className="field" rows={2} value={form.relevance}
                    onChange={setEvent('relevance')} disabled={!canManage} />
        </div>

        {/* formula */}
        <div>
          <span className="label">Computation</span>
          {isRatio ? (
            <div className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
              <input className="field font-mono" value={form.numerator} aria-label="Numerator"
                     onChange={(e) => { set('numerator')(e.target.value); set('formulaChanged')(true) }}
                     disabled={!canManage} />
              <input className="field font-mono" value={form.denominator} aria-label="Denominator"
                     onChange={(e) => { set('denominator')(e.target.value); set('formulaChanged')(true) }}
                     disabled={!canManage} />
              <input className="field tnum w-24" value={form.scale} aria-label="Scale"
                     onChange={(e) => { set('scale')(e.target.value); set('formulaChanged')(true) }}
                     disabled={!canManage} />
            </div>
          ) : (
            <input className="field font-mono" value={form.expression} aria-label="Expression"
                   onChange={(e) => { set('expression')(e.target.value); set('formulaChanged')(true) }}
                   disabled={!canManage} />
          )}
          <p className="mt-1 text-xs text-ink-muted">
            Reference a column as <code>{'{column_name}'}</code>. Only + − × ÷ and parentheses are
            allowed; formulas are parsed, never executed.
          </p>
        </div>

        {kpi.filters?.length ? (
          <div>
            <span className="label">Filters</span>
            <ul className="text-xs text-ink-secondary">
              {kpi.filters.map((f, i) => (
                <li key={i} className="font-mono">{f.field} {f.op} {String(f.value)}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {/* granularity */}
        <div>
          <span className="label">Granularity</span>
          <GranularityPicker
            granularity={form.granularity}
            aggregation={kpi.aggregation}
            dimensions={dimensions}
            onChange={(g) => set('granularity')(g)}
          />
        </div>

        {/* time semantics + sources, read-only for now */}
        <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
          <Row k="Calendar" v={kpi.time_semantics?.calendar_id} />
          <Row k="Date field" v={kpi.time_semantics?.date_field} />
          <Row k="Compared by default" v={(kpi.time_semantics?.comparison_default || '').replace(/_/g, ' ')} />
          <Row k="Source columns" v={(kpi.source_fields || []).join(', ')} />
          <Row k="Source dataset" v={kpi.sources?.[0]?.source_label || kpi.sources?.[0]?.dataset_id} />
          <Row k="Refresh cadence" v={kpi.sources?.[0]?.refresh_cadence} />
          {kpi.depends_on?.length ? <Row k="Depends on" v={kpi.depends_on.join(', ')} /> : null}
          {kpi.semantic_tags?.length ? <Row k="Tags" v={kpi.semantic_tags.join(', ')} /> : null}
        </dl>

        <KpiPreview kpiId={kpi.kpi_id} datasetId={datasetId} reloadKey={kpi.status + kpi.name} />

        <AnalystOnly show={isAnalyst} title="Provenance and validation">
          <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
            <Row k="Origin" v={(kpi.provenance?.origin || '').replace(/_/g, ' ')} />
            <Row k="Derivation rule" v={kpi.provenance?.derivation_rule || '—'} />
            <Row k="Screened by" v={kpi.provenance?.screened_by || '—'} />
            <Row k="Confidence" v={kpi.confidence != null ? kpi.confidence.toFixed(2) : '—'} />
            <Row k="Edited by" v={(kpi.provenance?.edited_by || []).join(', ') || '—'} />
            <Row k="Approved by" v={kpi.approval?.approved_by || '—'} />
          </dl>
          {kpi.validation_rules?.length ? (
            <ul className="mt-2 space-y-1 text-xs text-ink-secondary">
              {kpi.validation_rules.map((r, i) => (
                <li key={i}>· <span className="font-mono">{r.kind}</span> — {r.detail}</li>
              ))}
            </ul>
          ) : null}
          {kpi.provenance?.notes?.length ? (
            <ul className="mt-2 space-y-1 text-xs text-ink-muted">
              {kpi.provenance.notes.map((n, i) => <li key={i}>· {n}</li>)}
            </ul>
          ) : null}
          {Object.keys(kpi.provenance?.computability_evidence || {}).length ? (
            <pre className="mt-2 overflow-x-auto rounded-lg p-2 text-[11px]"
                 style={{ background: 'var(--plane)' }}>
              {JSON.stringify(kpi.provenance.computability_evidence, null, 2)}
            </pre>
          ) : null}
        </AnalystOnly>
      </div>

      {canManage ? (
        <div className="mt-4 flex flex-wrap gap-2 border-t pt-3" style={{ borderColor: 'var(--border)' }}>
          <button type="button" className="btn-primary" onClick={save} disabled={busy || !dirty}>
            <Save className="h-4 w-4" aria-hidden />
            Save changes
          </button>
          <button type="button" className="btn-ghost" onClick={() => onApprove(kpi.kpi_id)}
                  disabled={busy || kpi.status === 'approved'}>
            Approve
          </button>
          <button type="button" className="btn-ghost" onClick={() => onReject(kpi.kpi_id)}
                  disabled={busy || kpi.status === 'rejected'}>
            Reject
          </button>
          <button type="button" className="btn-ghost !px-2 ml-auto" onClick={() => onDelete(kpi.kpi_id)}
                  disabled={busy} aria-label={`Delete ${kpi.name}`}>
            <Trash2 className="h-4 w-4" aria-hidden />
          </button>
        </div>
      ) : null}
    </div>
  )
}

function initial(kpi) {
  return {
    name: kpi.name,
    business_definition: kpi.business_definition || '',
    relevance: kpi.relevance || '',
    unit: kpi.unit,
    higher_is_better: kpi.higher_is_better,
    granularity: { ...kpi.granularity },
    expression: kpi.formula.expression || '',
    numerator: kpi.formula.numerator_expression || '',
    denominator: kpi.formula.denominator_expression || '',
    scale: kpi.formula.scale ?? 1,
    formulaChanged: false,
  }
}

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-muted">{k}</dt>
      <dd className="truncate text-right text-ink-secondary">{v || '—'}</dd>
    </div>
  )
}

export function StatusBadge({ status }) {
  const map = {
    approved: ['good', 'Approved'],
    proposed: ['warning', 'Proposed'],
    needs_confirmation: ['serious', 'Needs confirmation'],
    rejected: ['neutral', 'Rejected'],
  }
  const [tone, label] = map[status] || ['neutral', status]
  return <Badge tone={tone}>{label}</Badge>
}
