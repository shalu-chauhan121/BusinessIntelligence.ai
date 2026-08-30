import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, History, Trash2 } from 'lucide-react'
import { Badge, EmptyState, ErrorState, LoadingCard, SectionTitle } from '../components/ui'
import { api } from '../lib/api'

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

  if (error) return <ErrorState title="Could not load your saved answers" message={error.message} onRetry={load} />
  if (!items) return <LoadingCard label="Loading saved answers…" />

  if (!items.length) {
    return (
      <EmptyState
        icon={History}
        title="No saved answers yet"
        message="Ask a question from the Investigation page and save it, and it will appear here with its full evidence trail."
        action={
          <Link className="btn-primary" to="/investigation">
            Ask a question
          </Link>
        }
      />
    )
  }

  return (
    <div className="space-y-4">
      <SectionTitle
        eyebrow="Saved work"
        title="Question history"
        description="Every saved answer keeps its full evidence trail, so a conclusion can be revisited and checked later."
      />
      <div className="card divide-y" style={{ borderColor: 'var(--border)' }}>
        {items.map((inv) => (
          <div key={inv.id} className="flex flex-wrap items-center gap-3 p-3">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <Link to={`/investigation/${inv.id}`} className="truncate text-sm font-medium text-ink hover:underline">
                  {inv.question || 'Untitled question'}
                </Link>
                {inv.legacy_format ? <Badge tone="warning">Old format</Badge> : null}
              </div>
              {inv.legacy_format ? (
                <div className="mt-0.5 flex items-center gap-1 text-xs text-ink-muted">
                  <AlertTriangle className="h-3 w-3" aria-hidden />
                  Saved before this page's current format — open it to ask again.
                </div>
              ) : (
                <>
                  <p className="mt-0.5 truncate text-xs text-ink-secondary">{inv.answer_preview}</p>
                  <div className="mt-1 flex flex-wrap items-center gap-2">
                    {(inv.kpis_used || []).map((k) => (
                      <Badge key={k} tone="neutral">
                        {k}
                      </Badge>
                    ))}
                    <span className="text-xs text-ink-muted">
                      {inv.turns != null ? `${inv.turns} tool ${inv.turns === 1 ? 'call' : 'calls'} · ` : ''}
                      {new Date(inv.created_at).toLocaleString()}
                    </span>
                  </div>
                </>
              )}
            </div>
            <button
              type="button"
              className="btn-ghost !px-2"
              aria-label="Delete saved answer"
              onClick={async () => {
                await api.deleteInvestigation(inv.id)
                load()
              }}
            >
              <Trash2 className="h-4 w-4" aria-hidden />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}
