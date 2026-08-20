import { useCallback, useEffect, useRef, useState } from 'react'
import { CheckCircle2, Download, FileSpreadsheet, Sparkles, Trash2, Upload } from 'lucide-react'
import { Badge, Callout, ErrorState, LoadingCard, SectionTitle, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { titleCase } from '../lib/format'

const REQUIRED = [
  ['date', 'YYYY-MM-DD', 'Required', 'Period start date. Weekly rows (one per Monday) work best.'],
  ['revenue', 'number', 'Recommended', 'The default headline KPI.'],
]

const DIMENSIONS = [
  ['region', 'North, South, East, West'],
  ['product', 'Product A, Product B'],
  ['channel', 'Online, Retail Partner'],
  ['segment', 'Enterprise, SMB'],
]

const METRICS = [
  ['units_sold', 'sum', 'Volume, for the price-versus-volume split'],
  ['orders', 'sum', 'Demand signal'],
  ['customers', 'sum', 'Demand signal and acquisition cost'],
  ['fulfilled_orders', 'sum', 'Fulfilment rate'],
  ['stockout_events', 'sum', 'Stockout rate — supply hypotheses'],
  ['inventory_units', 'average', 'Inventory position — supply hypotheses'],
  ['cost_of_goods', 'sum', 'Gross margin'],
  ['marketing_spend', 'sum', 'Acquisition cost and marketing hypotheses'],
  ['returns', 'sum', 'Quality signal'],
  ['support_tickets', 'sum', 'Service-load signal'],
]

export default function DataPage() {
  const [datasets, setDatasets] = useState(null)
  const [active, setActive] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState(null)
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
      } else {
        setActive(null)
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
    withBusy(() => api.uploadDataset(files[0]), `${files[0].name} uploaded and made active.`)
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
            <p className="text-sm font-medium text-ink">Drop a CSV here, or choose a file</p>
            <p className="mt-1 max-w-md text-xs text-ink-secondary">
              One row per period × dimensions. The format reference below lists every column the analysis understands.
            </p>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={(e) => upload(e.target.files)}
            />
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              <button type="button" className="btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
                Choose CSV
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
          title="Format reference"
          description="Any additional text column is treated as another dimension; any additional numeric column becomes an extra KPI."
        />
        <div className="grid gap-4 lg:grid-cols-3">
          <FormatTable
            title="Required"
            head={['Column', 'Type', 'Notes']}
            rows={REQUIRED.map(([c, t, , n]) => [c, t, n])}
          />
          <FormatTable title="Dimension columns" head={['Column', 'Example values']} rows={DIMENSIONS} />
          <FormatTable title="Metric columns" head={['Column', 'Aggregated', 'Enables']} rows={METRICS} />
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <Callout tone="info" title="Derived KPIs — do not upload these">
            Gross margin %, average order value, average selling price, fulfilment rate, stockout rate, return rate,
            customer acquisition cost, units per order and support tickets per 1,000 orders are all computed for you
            from the columns above.
          </Callout>
          <Callout tone="warning" title="How much history to provide">
            The significance test compares the selected quarter against its own history. Five quarters is the minimum;
            eight or more lets the system model seasonality and tell a real signal from a seasonal one.
          </Callout>
        </div>
      </section>
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

function FormatTable({ title, head, rows }) {
  return (
    <div className="card card-pad">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">{title}</div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-ink-muted">
            <tr>
              {head.map((h) => (
                <th key={h} className="py-1 pr-3 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="text-ink-secondary">
            {rows.map((r) => (
              <tr key={r[0]}>
                {r.map((cell, i) => (
                  <td key={i} className={`py-1 pr-3 ${i === 0 ? 'font-mono text-ink' : ''}`}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
