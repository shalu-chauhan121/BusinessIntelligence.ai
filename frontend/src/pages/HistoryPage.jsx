import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { History, Trash2 } from 'lucide-react'
import { Badge, EmptyState, ErrorState, LoadingCard, SectionTitle } from '../components/ui'
import { api } from '../lib/api'
import { VERDICT_COPY, formatDelta } from '../lib/format'

export default function HistoryPage() {
  const [items, setItems] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    try {
      const data = await api.listInvestigations()
      setItems(data.investigations)
    } catch (e) {
      setError(e)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  if (error) return <ErrorState title="Could not load your investigations" message={error.message} onRetry={load} />
  if (!items) return <LoadingCard label="Loading investigations…" />

  if (!items.length) {
    return (
      <EmptyState
        icon={History}
        title="No investigations yet"
        message="Run one from the Investigation page and it will be saved here, with its evidence and conclusions."
        action={
          <Link className="btn-primary" to="/investigation">
            Run an investigation
          </Link>
        }
      />
    )
  }

  return (
    <div className="space-y-4">
      <SectionTitle
        eyebrow="Saved work"
        title="Investigation history"
        description="Every run is stored with its full evidence trail, so a conclusion can be revisited and challenged later."
      />
      <div className="card divide-y" style={{ borderColor: 'var(--border)' }}>
        {items.map((inv) => {
          const verdict = VERDICT_COPY[inv.verdict] || VERDICT_COPY.within_normal_variation
          return (
            <div key={inv.id} className="flex flex-wrap items-center gap-3 p-3">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <Link to={`/investigation/${inv.id}`} className="truncate text-sm font-medium text-ink hover:underline">
                    {inv.headline || `${inv.kpi_label} · ${inv.timeframe?.pretty}`}
                  </Link>
                  <Badge tone={verdict.tone}>{verdict.label}</Badge>
                </div>
                <div className="mt-0.5 text-xs text-ink-muted">
                  {inv.kpi_label} · {inv.timeframe?.pretty} vs {inv.baseline_timeframe?.pretty} ·{' '}
                  {new Date(inv.created_at).toLocaleString()}
                </div>
                {inv.leading_hypothesis ? (
                  <div className="mt-0.5 text-xs text-ink-secondary">
                    Leading explanation: {inv.leading_hypothesis} ({inv.leading_confidence}%)
                  </div>
                ) : null}
              </div>
              <span className="tnum text-sm font-semibold" style={{ color: 'var(--delta-bad)' }}>
                {formatDelta(inv.change_pct)}
              </span>
              <button
                type="button"
                className="btn-ghost !px-2"
                aria-label="Delete investigation"
                onClick={async () => {
                  await api.deleteInvestigation(inv.id)
                  load()
                }}
              >
                <Trash2 className="h-4 w-4" aria-hidden />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
