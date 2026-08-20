import { useCallback, useEffect, useRef, useState } from 'react'
import { FileText, Search, Sparkles, Trash2, Upload } from 'lucide-react'
import { Badge, Callout, EmptyState, ErrorState, LoadingCard, SectionTitle, Spinner } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { api } from '../lib/api'
import { titleCase } from '../lib/format'

export default function DocumentsPage() {
  const { isAnalyst } = useAuth()
  const [state, setState] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const fileInput = useRef(null)

  const load = useCallback(async () => {
    try {
      setState(await api.listDocuments())
    } catch (e) {
      setError(e)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  async function withBusy(fn) {
    setBusy(true)
    setError(null)
    try {
      await fn()
      await load()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  async function search(e) {
    e.preventDefault()
    if (!query.trim()) return
    setBusy(true)
    try {
      setResults(await api.searchDocuments(query, 6))
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  if (!state && !error) return <LoadingCard label="Loading your documents…" />

  return (
    <div className="space-y-5">
      <SectionTitle
        eyebrow="Your data"
        title="Business documents"
        description="Operations reports, customer feedback, market intelligence, management commentary. These are chunked and indexed so the investigation can quote them as evidence — with the document and section shown."
      />

      {error ? <ErrorState title="Something went wrong" message={error.message} onRetry={load} /> : null}

      <div className="card card-pad">
        <div
          className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-8 text-center"
          style={{ borderColor: 'var(--baseline)' }}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            withBusy(() => api.uploadDocuments(e.dataTransfer.files))
          }}
        >
          <Upload className="mb-2 h-6 w-6 text-ink-muted" aria-hidden />
          <p className="text-sm font-medium text-ink">Drop documents here</p>
          <p className="mt-1 text-xs text-ink-secondary">.md, .txt, .pdf, .csv — several at once is fine.</p>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".md,.txt,.pdf,.csv,.log,.json"
            className="sr-only"
            onChange={(e) => withBusy(() => api.uploadDocuments(e.target.files))}
          />
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            <button type="button" className="btn-primary" onClick={() => fileInput.current?.click()} disabled={busy}>
              Choose files
            </button>
            <button type="button" className="btn-ghost" disabled={busy} onClick={() => withBusy(api.loadSampleDocuments)}>
              <Sparkles className="h-4 w-4" aria-hidden />
              Load sample documents
            </button>
          </div>
          {busy ? <div className="mt-3"><Spinner label="Indexing…" /></div> : null}
        </div>
      </div>

      {state?.documents?.length ? (
        <section>
          <SectionTitle
            title={`${state.count} document${state.count === 1 ? '' : 's'} indexed`}
            description={`${state.chunks} retrievable passages.`}
          />
          <div className="card divide-y" style={{ borderColor: 'var(--border)' }}>
            {state.documents.map((d) => (
              <div key={d.id} className="flex flex-wrap items-center gap-3 p-3">
                <FileText className="h-4 w-4 shrink-0 text-ink-muted" aria-hidden />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="truncate text-sm font-medium text-ink">{d.filename}</span>
                    <Badge tone="neutral">{titleCase(d.doc_type || '')}</Badge>
                  </div>
                  <div className="text-xs text-ink-muted">
                    {d.chunk_count} chunks · {d.word_count?.toLocaleString()} words ·{' '}
                    {new Date(d.created_at).toLocaleString()}
                  </div>
                </div>
                <button
                  type="button"
                  className="btn-ghost !px-2"
                  aria-label={`Delete ${d.filename}`}
                  disabled={busy}
                  onClick={() => withBusy(() => api.deleteDocument(d.id))}
                >
                  <Trash2 className="h-4 w-4" aria-hidden />
                </button>
              </div>
            ))}
          </div>
        </section>
      ) : (
        <EmptyState
          icon={FileText}
          title="No documents yet"
          message="The investigation works without them, using structured data alone — but documents are what let it corroborate or contradict an explanation with written evidence."
        />
      )}

      {isAnalyst ? (
        <section>
          <SectionTitle
            eyebrow="Analyst tools"
            title="Retrieval inspector"
            description="See exactly what the retrieval layer returns for a query, with its scores. This is the same index the Investigate and Contest stages use."
          />
          <form onSubmit={search} className="card card-pad">
            <div className="flex gap-2">
              <input
                className="field"
                placeholder="e.g. supply disruption stockout allocation"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                aria-label="Retrieval query"
              />
              <button type="submit" className="btn-primary" disabled={busy}>
                <Search className="h-4 w-4" aria-hidden />
                Search
              </button>
            </div>
            {results ? (
              <div className="mt-4">
                <p className="mb-2 text-xs text-ink-muted">
                  {results.method} · {results.indexed_chunks} chunks indexed
                </p>
                <ul className="space-y-2">
                  {results.results?.map((r) => (
                    <li key={r.chunk_id} className="rounded-lg border px-3 py-2" style={{ borderColor: 'var(--border)' }}>
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        <span className="font-medium text-ink">
                          {r.document_name}
                          {r.heading ? ` § ${r.heading}` : ''}
                        </span>
                        <span className="tnum text-ink-muted">
                          score {r.score} · relative {r.relevance} · absolute {r.strength}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-ink-secondary">{r.text.slice(0, 320)}…</p>
                    </li>
                  ))}
                  {!results.results?.length ? <li className="text-sm text-ink-muted">No passage cleared the relevance floor.</li> : null}
                </ul>
              </div>
            ) : null}
          </form>
        </section>
      ) : (
        <Callout tone="info" title="Analyst view">
          Users with the Data Analyst role also get a retrieval inspector here, showing exactly which passages the
          system retrieves for a query and with what scores.
        </Callout>
      )}
    </div>
  )
}
