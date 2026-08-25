import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Database, Plus } from 'lucide-react'
import ConflictPanel from '../components/kpi/ConflictPanel'
import ContractHeader from '../components/kpi/ContractHeader'
import KpiDefinitionEditor from '../components/kpi/KpiDefinitionEditor'
import KpiProposalList from '../components/kpi/KpiProposalList'
import { AnalystOnly, Callout, EmptyState, ErrorState, LoadingCard, SectionTitle } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { api } from '../lib/api'

/**
 * The KPI Studio — where proposed KPI definitions become authoritative ones.
 *
 * The whole point of this screen is that nothing here is decided for you: the
 * system says what it found, what it declined and what it cannot resolve, and a
 * person makes the call.
 */
export default function KpiStudioPage() {
  const { isAnalyst, can } = useAuth()
  const canManage = typeof can === 'function' ? can('manage_kpi_contract') : isAnalyst

  const [state, setState] = useState({ data: null, error: null, loading: true })
  const [selectedId, setSelectedId] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)
  const [creating, setCreating] = useState(false)

  const load = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }))
    try {
      const data = await api.kpiProposals()
      setState({ data, error: null, loading: false })
    } catch (error) {
      setState({ data: null, error, loading: false })
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const allKpis = useMemo(
    () => Object.values(state.data?.groups || {}).flat(),
    [state.data],
  )
  const selected = allKpis.find((k) => k.kpi_id === selectedId) || null
  const dimensions = selected?.dimensions || []

  async function act(fn, successMessage) {
    setBusy(true)
    setMessage(null)
    try {
      await fn()
      await load()
      setMessage(successMessage)
    } catch (error) {
      setState((s) => ({ ...s, error }))
    } finally {
      setBusy(false)
    }
  }

  if (state.loading && !state.data) return <LoadingCard label="Loading the KPI contract…" lines={5} />

  if (state.error && !state.data) {
    return state.error.status === 409 ? (
      <EmptyState
        icon={Database}
        title="No business data yet"
        message="KPIs are discovered from your data. Upload a CSV, or load the bundled sample company, and the contract will be built from it."
        action={
          <Link className="btn-primary" to="/data">
            Go to Business data
          </Link>
        }
      />
    ) : (
      <ErrorState title="Could not load the KPI contract" message={state.error.message} onRetry={load} />
    )
  }

  const { summary, groups, conflicts, unavailable, rejected_candidates: rejected, discovery_note: note } =
    state.data

  return (
    <div className="space-y-5">
      <ContractHeader
        summary={summary}
        datasetLabel="KPI definitions for your active dataset"
        domains={summary.detected_domains}
        screenedBy={state.data.screened_by || 'deterministic'}
        busy={busy}
        canManage={canManage}
        onDiscover={() =>
          act(() => api.kpiDiscover({ useLlm: true }), 'Discovery finished. Review the proposals below.')
        }
        onApprove={() =>
          act(
            () => api.kpiApproveContract(),
            'The contract is now authoritative. Every stage of the analysis resolves KPIs through it.',
          )
        }
      />

      {state.error ? (
        <Callout tone="warning" title="That did not work">
          {state.error.message}
        </Callout>
      ) : null}
      {message ? (
        <Callout tone="good" title="Done">
          {message}
        </Callout>
      ) : null}
      {note ? (
        <Callout tone="info" title="Discovery notes">
          {note}
        </Callout>
      ) : null}

      <ConflictPanel
        conflicts={conflicts}
        canManage={canManage}
        busy={busy}
        onResolve={(id, option, rationale) =>
          act(() => api.kpiResolveConflict(id, option, rationale), 'Decision recorded.')
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div>
          {canManage ? (
            <div className="mb-3">
              <button type="button" className="btn-ghost" onClick={() => setCreating(true)} disabled={busy}>
                <Plus className="h-4 w-4" aria-hidden />
                Define a KPI yourself
              </button>
            </div>
          ) : null}
          <KpiProposalList groups={groups} selectedId={selectedId} onSelect={setSelectedId} />
        </div>

        <div className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          {creating ? (
            <NewKpiForm
              busy={busy}
              onCancel={() => setCreating(false)}
              onCreate={(payload) =>
                act(async () => {
                  const created = await api.kpiCreate(payload)
                  setCreating(false)
                  setSelectedId(created.kpi.kpi_id)
                }, 'KPI created. Review it, then approve it.')
              }
            />
          ) : null}

          {selected ? (
            <KpiDefinitionEditor
              kpi={selected}
              dimensions={dimensions}
              isAnalyst={isAnalyst}
              canManage={canManage}
              busy={busy}
              onClose={() => setSelectedId(null)}
              onSave={(id, patch) =>
                act(
                  () => api.kpiUpdate(id, patch),
                  'Definition updated — this KPI now needs to be approved again, and the ' +
                    'contract re-approved, before the Dashboard reflects the change.'
                )
              }
              onApprove={(id) => act(() => api.kpiApprove(id), 'KPI approved.')}
              onReject={(id) =>
                act(() => api.kpiReject(id, 'Not used by this business.'), 'KPI rejected.')
              }
              onDelete={(id) =>
                act(async () => {
                  await api.kpiDelete(id)
                  setSelectedId(null)
                }, 'KPI removed.')
              }
            />
          ) : !creating ? (
            <Callout tone="info" title="Select a KPI">
              Choose a KPI on the left to see its full definition — formula, grain, calendar,
              sources, business rules and provenance — and to edit or approve it.
            </Callout>
          ) : null}

          <NotProposed unavailable={unavailable} rejected={rejected} isAnalyst={isAnalyst} />
        </div>
      </div>
    </div>
  )
}

/**
 * What the system did NOT propose, and why.
 *
 * This is the evidence that a KPI is absent because the data cannot support it,
 * rather than because nothing thought to look.
 */
function NotProposed({ unavailable = [], rejected = [], isAnalyst }) {
  if (!unavailable.length && !rejected.length) return null
  return (
    <AnalystOnly show={isAnalyst} title={`Considered and not proposed (${unavailable.length + rejected.length})`}>
      {unavailable.length ? (
        <>
          <p className="mb-1 text-xs font-medium text-ink-secondary">
            Library KPIs this dataset cannot support
          </p>
          <ul className="mb-3 space-y-1 text-xs text-ink-muted">
            {unavailable.slice(0, 12).map((u) => (
              <li key={u.library_id}>
                · <span className="text-ink-secondary">{u.name}</span> — {u.why_unavailable}
              </li>
            ))}
            {unavailable.length > 12 ? <li>· and {unavailable.length - 12} more</li> : null}
          </ul>
        </>
      ) : null}
      {rejected.length ? (
        <>
          <p className="mb-1 text-xs font-medium text-ink-secondary">
            Combinations that are computable but not meaningful
          </p>
          <ul className="space-y-1 text-xs text-ink-muted">
            {rejected.slice(0, 12).map((r, i) => (
              <li key={i}>
                · <span className="font-mono text-ink-secondary">{r.expression}</span> — {r.reason}
              </li>
            ))}
            {rejected.length > 12 ? <li>· and {rejected.length - 12} more</li> : null}
          </ul>
        </>
      ) : null}
    </AnalystOnly>
  )
}

function NewKpiForm({ onCreate, onCancel, busy }) {
  const [form, setForm] = useState({
    name: '',
    business_definition: '',
    kind: 'sum',
    expression: '',
    numerator_expression: '',
    denominator_expression: '',
    scale: 1,
    unit: 'currency',
  })
  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }))
  const isRatio = form.kind === 'ratio'

  const submit = (e) => {
    e.preventDefault()
    onCreate({
      name: form.name,
      business_definition: form.business_definition,
      unit: form.unit,
      formula: isRatio
        ? {
            kind: 'ratio',
            expression: '',
            numerator_expression: form.numerator_expression,
            denominator_expression: form.denominator_expression,
            scale: Number(form.scale) || 1,
          }
        : { kind: form.kind, expression: form.expression },
    })
  }

  return (
    <form className="card card-pad space-y-3" onSubmit={submit}>
      <SectionTitle
        title="Define a KPI"
        description="For metrics your organisation uses that cannot be inferred from the data alone."
      />
      <div>
        <label className="label" htmlFor="new-name">Name</label>
        <input id="new-name" className="field" value={form.name} onChange={set('name')} required />
      </div>
      <div>
        <label className="label" htmlFor="new-def">Business definition</label>
        <textarea id="new-def" className="field" rows={2} value={form.business_definition}
                  onChange={set('business_definition')} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className="label" htmlFor="new-kind">Aggregation</label>
          <select id="new-kind" className="field" value={form.kind} onChange={set('kind')}>
            <option value="sum">Sum over the period</option>
            <option value="mean">Average over the period</option>
            <option value="ratio">Ratio of two expressions</option>
          </select>
        </div>
        <div>
          <label className="label" htmlFor="new-unit">Unit</label>
          <select id="new-unit" className="field" value={form.unit} onChange={set('unit')}>
            {['currency', 'count', 'percent', 'ratio', 'duration', 'score'].map((u) => (
              <option key={u} value={u}>{u}</option>
            ))}
          </select>
        </div>
      </div>
      {isRatio ? (
        <div className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
          <input className="field font-mono" placeholder="{numerator}" aria-label="Numerator"
                 value={form.numerator_expression} onChange={set('numerator_expression')} required />
          <input className="field font-mono" placeholder="{denominator}" aria-label="Denominator"
                 value={form.denominator_expression} onChange={set('denominator_expression')} required />
          <input className="field tnum w-24" aria-label="Scale" value={form.scale} onChange={set('scale')} />
        </div>
      ) : (
        <input className="field font-mono" placeholder="{revenue} - {cost_of_goods}" aria-label="Expression"
               value={form.expression} onChange={set('expression')} required />
      )}
      <p className="text-xs text-ink-muted">
        Reference a column as <code>{'{column_name}'}</code>. Only + − × ÷ and parentheses are allowed.
      </p>
      <div className="flex gap-2">
        <button type="submit" className="btn-primary" disabled={busy}>Create</button>
        <button type="button" className="btn-ghost" onClick={onCancel} disabled={busy}>Cancel</button>
      </div>
    </form>
  )
}
