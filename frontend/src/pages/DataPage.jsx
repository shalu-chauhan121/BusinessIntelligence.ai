import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowRight,
  CheckCircle2,
  CircleSlash,
  Download,
  FileSpreadsheet,
  Layers,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react'
import { Badge, Callout, ErrorState, LoadingCard, SectionTitle, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { titleCase } from '../lib/format'

export default function DataPage() {
  const [datasets, setDatasets] = useState(null)
  const [active, setActive] = useState(null)
  const [library, setLibrary] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)
  const [pendingFiles, setPendingFiles] = useState(null)
  const [sourceMeta, setSourceMeta] = useState({})
  const fileInput = useRef(null)

  const load = useCallback(async () => {
    setError(null)
    try {
      const list = await api.listDatasets()
      setDatasets(list.datasets)
      if (list.datasets.length) {
        try {
          setActive(await api.activeDataset())
        } catch {
          setActive(null)
        }
        try {
          setLibrary(await api.kpiLibrary())
        } catch {
          setLibrary(null)
        }
      } else {
        setActive(null)
        setLibrary(null)
      }
    } catch (e) {
      setError(e)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  async function withBusy(fn, successMessage) {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      await fn()
      setMessage(successMessage)
      await load()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const upload = (files) => {
    if (!files?.length) return
    if (files.length === 1) {
      withBusy(() => api.uploadDataset(files[0]), `${files[0].name} uploaded and made active.`)
      return
    }
    // Several files: stage them so a refresh cadence can be given per source
    // before reconciling. Purely optional metadata — leaving it blank behaves
    // exactly as before (cadence unknown, freshness taken from the data itself).
    const list = Array.from(files)
    setPendingFiles(list)
    setSourceMeta(Object.fromEntries(list.map((f) => [f.name, { cadence: '', last_refresh_at: '' }])))
  }

  const updateSourceMeta = (filename, patch) => {
    setSourceMeta((prev) => ({ ...prev, [filename]: { ...prev[filename], ...patch } }))
  }

  const confirmMultisourceUpload = () => {
    if (!pendingFiles?.length) return
    const meta = Object.fromEntries(
      Object.entries(sourceMeta)
        .map(([name, m]) => [name, {
          ...(m.cadence ? { cadence: m.cadence } : {}),
          ...(m.last_refresh_at ? { last_refresh_at: new Date(m.last_refresh_at).toISOString() } : {}),
        }])
        .filter(([, m]) => Object.keys(m).length),
    )
    withBusy(
      () => api.uploadDatasets(pendingFiles, meta),
      `${pendingFiles.length} sources reconciled into one business view — see where the data came from on the dashboard.`,
    ).then(() => {
      setPendingFiles(null)
      setSourceMeta({})
    })
  }

  const cancelMultisourceUpload = () => {
    setPendingFiles(null)
    setSourceMeta({})
  }

  if (!datasets && !error) return <LoadingCard label="Loading your datasets…" />

  return (
    <div className="space-y-5">
      <SectionTitle
        eyebrow="Your data"
        title="Business metrics"
        description="Everything the product shows is computed from the file you upload here. Different users analyse different data — nothing is shared between accounts."
      />

      {error ? <ErrorState title="Upload failed" message={error.message} onRetry={load} /> : null}
      {pendingFiles?.length ? (
        <div className="card card-pad space-y-3">
          <SectionTitle
            eyebrow="Before reconciling"
            title="Refresh cadence per source"
            description="Optional. Left blank, a source's freshness is judged from its own data rather than asserted — leaving these blank is a legitimate choice, not a shortcut."
          />
          <div className="space-y-2">
            {pendingFiles.map((f) => (
              <div key={f.name} className="flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: 'var(--border)' }}>
                <Layers className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
                <span className="min-w-0 flex-1 truncate font-medium text-ink">{f.name}</span>
                <select
                  className="field"
                  value={sourceMeta[f.name]?.cadence || ''}
                  onChange={(e) => updateSourceMeta(f.name, { cadence: e.target.value })}
                >
                  <option value="">Cadence unknown</option>
                  <option value="hourly">Hourly</option>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
                <input
                  type="datetime-local"
                  className="field"
                  value={sourceMeta[f.name]?.last_refresh_at || ''}
                  onChange={(e) => updateSourceMeta(f.name, { last_refresh_at: e.target.value })}
                  title="Last refreshed (optional) — when this source itself says its data was last updated"
                />
              </div>
            ))}
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn-primary" disabled={busy} onClick={confirmMultisourceUpload}>
              Reconcile {pendingFiles.length} sources
            </button>
            <button type="button" className="btn-ghost" disabled={busy} onClick={cancelMultisourceUpload}>
              Cancel
            </button>
          </div>
          {busy ? <Spinner label="Reconciling…" /> : null}
        </div>
      ) : null}
      {message ? (
        <Callout tone="good" title="Done">
          {message}
        </Callout>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="card card-pad lg:col-span-2">
          <div
            className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center"
            style={{ borderColor: 'var(--baseline)' }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              upload(e.dataTransfer.files)
            }}
          >
            <Upload className="mb-2 h-6 w-6 text-ink-muted" aria-hidden />
            <p className="text-sm font-medium text-ink">Drop your CSVs here, or choose files</p>
            <p className="mt-1 max-w-md text-xs text-ink-secondary">
              One row per period × dimensions. Drop several files at once — if your business data is
              split across systems, they are reconciled into one view. They just need to use the same
              field names; there is nothing to map.
            </p>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              multiple
              className="sr-only"
              onChange={(e) => upload(e.target.files)}
            />
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              <button type="button" className="btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
                Choose CSVs
              </button>
              <a className="btn-ghost" href={api.templateUrl()}>
                <Download className="h-4 w-4" aria-hidden />
                Download template
              </a>
              <button
                type="button"
                className="btn-ghost"
                disabled={busy}
                onClick={() => withBusy(api.loadSampleDataset, 'Sample company dataset loaded (14 quarters).')}
              >
                <Sparkles className="h-4 w-4" aria-hidden />
                Load sample company
              </button>
            </div>
            {busy ? <div className="mt-3"><Spinner label="Working…" /></div> : null}
          </div>
        </div>

        <div className="card card-pad">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Active dataset</div>
          {active ? (
            <div className="mt-2 space-y-2 text-sm">
              <div className="flex items-center gap-2 font-medium text-ink">
                <FileSpreadsheet className="h-4 w-4 text-ink-muted" aria-hidden />
                <span className="truncate">{active.filename}</span>
              </div>
              <dl className="space-y-1 text-xs text-ink-secondary">
                <Row k="Rows" v={active.schema.row_count?.toLocaleString()} />
                <Row k="Grain" v={active.schema.grain} />
                <Row k="Covers" v={`${active.schema.date_min} → ${active.schema.date_max}`} />
                <Row k="Quarters" v={active.schema.quarters?.length} />
                <Row k="Dimensions" v={active.schema.dimensions?.map(titleCase).join(', ') || 'none'} />
                <Row k="KPIs available" v={active.schema.available_kpis?.length} />
              </dl>
              {active.schema.warnings?.length ? (
                <div className="mt-2 space-y-1 text-xs" style={{ color: 'var(--status-warning)' }}>
                  {active.schema.warnings.map((w, i) => (
                    <p key={i}>{w}</p>
                  ))}
                </div>
              ) : null}
            </div>
          ) : (
            <p className="mt-2 text-sm text-ink-secondary">No dataset uploaded yet.</p>
          )}
        </div>
      </div>

      {active ? (
        <Callout tone="info" title="KPIs are defined by the KPI contract">
          <p>
            {active.schema.available_kpis?.length ?? 0} KPIs are currently authoritative for this
            dataset. Review what the system discovered, set each KPI's granularity, override a
            definition or add one of your own before relying on the analysis.
          </p>
          <Link className="btn-ghost mt-2 inline-flex" to="/kpi">
            Open the KPI contract
            <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
        </Callout>
      ) : null}

      {datasets?.length ? (
        <section>
          <SectionTitle title="Uploaded datasets" description="Only the active dataset is analysed." />
          <div className="card divide-y" style={{ borderColor: 'var(--border)' }}>
            {datasets.map((d) => (
              <div key={d.id} className="flex flex-wrap items-center gap-3 p-3">
                <FileSpreadsheet className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium text-ink">{d.filename}</span>
                    {d.is_active ? <Badge tone="good" icon={CheckCircle2}>Active</Badge> : null}
                  </div>
                  <div className="text-xs text-ink-muted">
                    {d.schema?.row_count?.toLocaleString()} rows · {(d.size_bytes / 1024).toFixed(0)} KB ·{' '}
                    {new Date(d.created_at).toLocaleString()}
                  </div>
                </div>
                {!d.is_active ? (
                  <button
                    type="button"
                    className="btn-ghost"
                    disabled={busy}
                    onClick={() => withBusy(() => api.activateDataset(d.id), `${d.filename} is now the active dataset.`)}
                  >
                    Make active
                  </button>
                ) : null}
                <button
                  type="button"
                  className="btn-ghost !px-2"
                  aria-label={`Delete ${d.filename}`}
                  disabled={busy}
                  onClick={() => withBusy(() => api.deleteDataset(d.id), `${d.filename} deleted.`)}
                >
                  <Trash2 className="h-4 w-4" aria-hidden />
                </button>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section>
        <SectionTitle
          title="What the system understood about your data"
          description="Nothing here is hard-coded to one kind of business. Column names are matched to KPI concepts, so net_sales, turnover and revenue all mean the same thing — and a KPI only appears when the data can actually support it."
        />
        {library ? (
          <LibraryReport library={library} />
        ) : (
          <Callout tone="info" title="Upload a dataset to see this">
            Once a dataset is active, this shows exactly which KPIs it supports and which it does
            not — with the reason for each.
          </Callout>
        )}
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Callout tone="info" title="The only column that is genuinely required">
            A date column, formatted <span className="font-mono">YYYY-MM-DD</span>. Everything else
            is discovered: numeric columns become measures, text columns become dimensions you can
            slice by, and derived KPIs are proposed from the combinations that are meaningful.
          </Callout>
          <Callout tone="warning" title="How much history to provide">
            The significance test compares the selected quarter against its own history. Five
            quarters is the minimum; eight or more lets the system model seasonality and tell a real
            signal from a seasonal one.
          </Callout>
        </div>
      </section>
    </div>
  )
}

/**
 * Which library KPIs bound to this dataset, and why the rest did not.
 *
 * The unavailable list matters as much as the available one: it is the evidence
 * that a KPI is missing because the data cannot support it, rather than because
 * nothing looked for it.
 */
function LibraryReport({ library }) {
  const bound = library.filter((e) => e.bound)
  const unavailable = library.filter((e) => !e.bound)
  const byDomain = {}
  bound.forEach((e) => {
    byDomain[e.domain] = byDomain[e.domain] || []
    byDomain[e.domain].push(e)
  })

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="card card-pad">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">
          <CheckCircle2 className="h-3.5 w-3.5" aria-hidden />
          Available in your data ({bound.length})
        </div>
        {Object.entries(byDomain).map(([domain, entries]) => (
          <div key={domain} className="mb-3">
            <div className="mb-1 text-[11px] uppercase tracking-wider text-ink-muted">
              {domain.replace(/_/g, ' ')}
            </div>
            <ul className="space-y-1 text-xs">
              {entries.map((e) => (
                <li key={e.id} className="flex justify-between gap-3">
                  <span className="truncate text-ink">{e.name}</span>
                  <span className="shrink-0 text-ink-muted">{e.unit}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="card card-pad">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">
          <CircleSlash className="h-3.5 w-3.5" aria-hidden />
          Considered, but not available ({unavailable.length})
        </div>
        <ul className="space-y-1 text-xs text-ink-muted">
          {unavailable.slice(0, 14).map((e) => (
            <li key={e.id}>
              · <span className="text-ink-secondary">{e.name}</span> — {e.why_unavailable}
            </li>
          ))}
          {unavailable.length > 14 ? <li>· and {unavailable.length - 14} more</li> : null}
        </ul>
        <p className="mt-2 text-xs text-ink-muted">
          Add the columns these need and they become available. They are never invented.
        </p>
      </div>
    </div>
  )
}

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-muted">{k}</dt>
      <dd className="truncate text-right text-ink-secondary">{v ?? '—'}</dd>
    </div>
  )
}
