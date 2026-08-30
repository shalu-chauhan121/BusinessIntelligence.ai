import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AlertTriangle, ChevronRight, Database, Sparkles, Wrench } from 'lucide-react'
import QuestionBar from '../components/QuestionBar'
import { Badge, EmptyState, ErrorState, LoadingCard, SectionTitle } from '../components/ui'
import { api } from '../lib/api'

/**
 * Example questions, built from the KPIs this dataset actually has.
 *
 * Generic examples are worse than useless on an unfamiliar dataset — a hospital
 * user offered "why did revenue fall" learns nothing about what they can ask.
 */
function exampleQuestions(meta) {
  const catalogue = meta?.schema?.kpi_catalogue || []
  if (!catalogue.length) return []
  const [first, second] = catalogue
  const examples = []
  if (first) examples.push(`Why did ${first.label.toLowerCase()} change last quarter?`)
  if (first && second) {
    examples.push(
      `Why did ${first.label.toLowerCase()} fall even though ${second.label.toLowerCase()} rose?`
    )
  }
  return examples
}

// One narrative source: the model's own prose in `answer`. Every other status
// still renders that same field (it is populated even on `truncated`), with a
// banner above it saying the run did not finish cleanly.
const STATUS_BANNER = {
  max_turns_exhausted: {
    tone: 'warning',
    label: 'Ran out of turns before concluding',
    detail: 'The answer below reflects what the model had established when its turn budget ran out.',
  },
  truncated: {
    tone: 'warning',
    label: 'Answer cut off',
    detail: 'The model’s final turn hit the output-token limit — the answer below may be incomplete.',
  },
  refused: {
    tone: 'critical',
    label: 'Declined to answer',
    detail: 'A safety classifier declined this question. Try rephrasing rather than resubmitting unchanged.',
  },
  llm_required: {
    tone: 'critical',
    label: 'No reasoning model configured',
    detail: 'This deployment has no ANTHROPIC_API_KEY set, so the agent loop cannot run.',
  },
}

export default function InvestigationPage() {
  const { investigationId } = useParams()
  const [meta, setMeta] = useState(null)
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [legacy, setLegacy] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const examples = useMemo(() => exampleQuestions(meta), [meta])

  useEffect(() => {
    api.activeDataset().then(setMeta).catch((e) => setError(e))
  }, [])

  useEffect(() => {
    if (!investigationId) return
    setBusy(true)
    setLegacy(null)
    api
      .getInvestigation(investigationId)
      .then((data) => {
        if (data.status === 'legacy_format') {
          setLegacy(data)
          setResult(null)
        } else {
          setResult(data)
          setQuestion(data.question || '')
        }
      })
      .catch(setError)
      .finally(() => setBusy(false))
  }, [investigationId])

  const ask = useCallback(async (asked) => {
    const text = (asked ?? question).trim()
    if (!text) return
    setBusy(true)
    setError(null)
    setLegacy(null)
    setQuestion(text)
    try {
      // Every ending of the loop is HTTP 200 with a typed `status` — a
      // question the model could not converge on is a normal branch here,
      // rendered by `AnswerCard`, not a caught error.
      const data = await api.askAgent({ question: text, persist: true })
      setResult(data)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }, [question])

  if (error && error.status === 409) {
    return (
      <EmptyState
        icon={Database}
        title="No business data yet"
        message="An investigation needs data to investigate. Upload a CSV or load the sample company first."
        action={
          <Link className="btn-primary" to="/data">
            Go to Business data
          </Link>
        }
      />
    )
  }

  return (
    <div className="space-y-5">
      <SectionTitle
        eyebrow="Question-driven investigation"
        title="Ask a business question"
        description="Ask what you actually want to know. A reasoning model decides which analysis tools to call, pulls real numbers from your data, and writes the answer itself — every figure in it comes from a tool call you can inspect below."
      />

      <QuestionBar
        value={question}
        onChange={setQuestion}
        onSubmit={ask}
        busy={busy}
        examples={examples}
      />

      {error && error.status !== 409 ? (
        <ErrorState
          title="The question could not be answered"
          message={error.message}
          onRetry={() => ask()}
        />
      ) : null}

      {busy && !result && !legacy ? <LoadingCard label="Working out which tools to call…" lines={6} /> : null}

      {!result && !legacy && !busy ? (
        <EmptyState
          icon={Sparkles}
          title="Ask anything about this data"
          message="Describe what happened and what puzzles you about it. Every number in the answer is computed by a tool call against your uploaded data."
        />
      ) : null}

      {legacy ? (
        <EmptyState
          icon={AlertTriangle}
          title="Saved before this page's current format"
          message={`"${legacy.question || 'This investigation'}" was saved by an earlier version of this product and can no longer be rendered. Ask the question again to get a fresh answer.`}
          action={
            <button type="button" className="btn-primary" onClick={() => ask(legacy.question)}>
              Ask it again
            </button>
          }
        />
      ) : null}

      {result ? (
        <>
          <AnswerCard result={result} />
          <EvidenceTrail evidence={result.evidence} />
          {result.telemetry ? <TelemetryResult telemetry={result.telemetry} /> : null}
          <EngineFooter engine={result.engine} />
        </>
      ) : null}
    </div>
  )
}

function AnswerCard({ result }) {
  const banner = STATUS_BANNER[result.status]
  const kpis = result.kpis_used || []
  const periods = result.periods_used || []
  return (
    <div className="space-y-3">
      {banner ? (
        <div
          className="rounded-lg border-l-4 px-3 py-2.5 text-sm"
          style={{
            borderColor: banner.tone === 'critical' ? 'var(--status-critical)' : 'var(--status-warning)',
            background: 'var(--plane)',
          }}
        >
          <div className="font-semibold text-ink">{banner.label}</div>
          <div className="text-ink-secondary">{banner.detail}</div>
        </div>
      ) : null}

      <div className="card card-pad">
        <div className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Answer</div>
        {result.answer ? (
          <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-ink-secondary">{result.answer}</p>
        ) : (
          <p className="mt-2 text-sm text-ink-muted">No answer was produced.</p>
        )}

        {kpis.length || periods.length ? (
          <div className="mt-4 flex flex-wrap gap-2">
            {kpis.map((k) => (
              <Badge key={k} tone="neutral">
                {k}
              </Badge>
            ))}
            {periods.map((p, i) => (
              <Badge key={i} tone="neutral">
                {typeof p === 'string' ? p : JSON.stringify(p)}
              </Badge>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

/** How the answer was reached — collapsed by default, everyone can open it. */
function EvidenceTrail({ evidence = [] }) {
  if (!evidence.length) return null
  return (
    <details className="card card-pad">
      <summary className="flex cursor-pointer select-none items-center gap-2 text-sm font-semibold text-ink">
        <Wrench className="h-4 w-4 text-ink-muted" aria-hidden />
        How I worked this out
        <span className="tnum text-xs font-normal text-ink-muted">
          ({evidence.length} tool {evidence.length === 1 ? 'call' : 'calls'})
        </span>
      </summary>
      <ol className="mt-3 space-y-2">
        {evidence.map((step) => (
          <li key={step.step} className="rounded-lg border" style={{ borderColor: 'var(--border)' }}>
            <details>
              <summary
                className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-sm"
                style={{ color: step.is_error ? 'var(--status-critical)' : 'var(--text-primary)' }}
              >
                <ChevronRight className="h-3.5 w-3.5 shrink-0 text-ink-muted" aria-hidden />
                <span className="tnum text-ink-muted">{step.step}.</span>
                <code className="font-medium">{step.tool}</code>
                {step.is_error ? <Badge tone="critical">error</Badge> : null}
              </summary>
              <div className="space-y-2 border-t px-3 py-2" style={{ borderColor: 'var(--border)' }}>
                {step.args && Object.keys(step.args).length ? (
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-muted">Arguments</div>
                    <pre className="mt-1 overflow-x-auto rounded-md p-2 text-xs" style={{ background: 'var(--plane)' }}>
                      {JSON.stringify(step.args, null, 2)}
                    </pre>
                  </div>
                ) : null}
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-muted">Result</div>
                  <pre className="mt-1 overflow-x-auto rounded-md p-2 text-xs" style={{ background: 'var(--plane)' }}>
                    {JSON.stringify(step.result, null, 2)}
                  </pre>
                </div>
              </div>
            </details>
          </li>
        ))}
      </ol>
    </details>
  )
}

function TelemetryResult({ telemetry }) {
  const processing = telemetry.processing || {}
  const llm = processing.llm || {}
  const nonLlm = processing.non_llm || {}
  const errors = telemetry.errors || []
  return (
    <div className="card card-pad">
      <SectionTitle eyebrow="Runtime telemetry" title="This insight request" description="Model usage comes from the provider when available; no prompts, model output or raw business data are stored." />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="End-to-end latency" value={`${telemetry.latency_ms || 0} ms`} />
        <Stat label="Model calls" value={telemetry.model_calls || 0} />
        <Stat label="Model" value={telemetry.model_name || 'No model call'} />
        <Stat label="LLM cache" value={telemetry.cache_hit ? 'Cache hit' : telemetry.cache_miss ? 'Cache miss' : 'Not used'} />
        <Stat label="Prompt tokens" value={(telemetry.prompt_tokens || telemetry.input_tokens || 0).toLocaleString()} />
        <Stat label="Completion tokens" value={(telemetry.completion_tokens || telemetry.output_tokens || 0).toLocaleString()} />
        <Stat label="Total tokens" value={(telemetry.total_tokens || 0).toLocaleString()} />
        <Stat label="Estimated cost" value={`$${Number(telemetry.estimated_cost || 0).toFixed(3)}`} />
        <Stat label="Request status" value={telemetry.status || 'unknown'} />
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <Stat label="LLM Processing" value={`${llm.duration_ms || 0} ms · ${llm.step_count || 0} calls`} />
        <Stat label="Non-LLM Processing" value={`${nonLlm.duration_ms || 0} ms · ${nonLlm.step_count || 0} steps`} />
      </div>
      <p className="mt-3 text-xs text-ink-muted">
        Started {formatTelemetryTime(telemetry.start_time)} · Ended {formatTelemetryTime(telemetry.end_time)} · Trace ID: {telemetry.trace_id}
      </p>
      {errors.length ? (
        <p className="mt-2 text-xs text-delta-bad">Tracked errors: {errors.map((error) => `${error.step}: ${error.error_type}`).join(' · ')}</p>
      ) : null}
    </div>
  )
}

function formatTelemetryTime(value) {
  return value ? new Date(value).toLocaleTimeString() : 'not recorded'
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg px-3 py-2" style={{ background: 'var(--plane)' }}>
      <div className="text-xs text-ink-muted">{label}</div>
      <div className="tnum text-sm font-semibold text-ink">{value}</div>
    </div>
  )
}

function EngineFooter({ engine }) {
  if (!engine) return null
  return (
    <p className="text-xs text-ink-muted">
      {engine.turns} tool-loop {engine.turns === 1 ? 'turn' : 'turns'} · completed in {engine.seconds}s ·
      {' '}reasoning layer: {engine.model} · numbers computed by the structured analysis layer.
    </p>
  )
}
