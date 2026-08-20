import { useCallback, useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { Database, Play, Quote, Sparkles } from 'lucide-react'
import TrendChart from '../components/charts/TrendChart'
import DriverChart from '../components/charts/DriverChart'
import HypothesisCard from '../components/HypothesisCard'
import Recommendations from '../components/Recommendations'
import TimeframePicker from '../components/TimeframePicker'
import { AnalystOnly, Badge, Callout, EmptyState, ErrorState, LoadingCard, SectionTitle, Spinner } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import useAnalysisSettings from '../hooks/useAnalysisSettings'
import { api } from '../lib/api'
import { STAGES, VERDICT_COPY, formatDelta, formatValue, titleCase } from '../lib/format'

export default function InvestigationPage() {
  const { investigationId } = useParams()
  const { isAnalyst } = useAuth()
  const [settings, update] = useAnalysisSettings()
  const [stage, setStage] = useState('observe')
  const [meta, setMeta] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.activeDataset().then(setMeta).catch((e) => setError(e))
  }, [])

  useEffect(() => {
    if (!investigationId) return
    setBusy(true)
    api
      .getInvestigation(investigationId)
      .then((data) => {
        setResult(data)
        setStage('act')
      })
      .catch(setError)
      .finally(() => setBusy(false))
  }, [investigationId])

  const run = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const data = await api.runInvestigation({
        kpi: settings.kpi,
        year: settings.year,
        quarter: settings.quarter,
        comparison: settings.comparison,
        use_llm: true,
        persist: true,
      })
      setResult(data)
      setStage('observe')
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }, [settings])

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
        eyebrow="Four-stage investigation"
        title="Observe → Investigate → Contest → Act"
        description="Each stage answers a different question. They are kept separate on purpose: proposing an explanation and trying to break it are not the same job."
        right={
          <button type="button" className="btn-primary" onClick={run} disabled={busy}>
            {busy ? <Spinner label="Running…" /> : (
              <>
                <Play className="h-4 w-4" aria-hidden />
                Run investigation
              </>
            )}
          </button>
        }
      />

      {meta ? (
        <TimeframePicker
          timeframes={meta.timeframes}
          year={settings.year ?? meta.timeframes?.at(-1)?.year}
          quarter={settings.quarter ?? meta.timeframes?.at(-1)?.quarter}
          kpi={settings.kpi}
          kpiOptions={meta.schema?.kpi_catalogue || []}
          comparison={settings.comparison}
          onChange={update}
          busy={busy}
        />
      ) : null}

      {error && error.status !== 409 ? (
        <ErrorState title="The investigation could not be completed" message={error.message} onRetry={run} />
      ) : null}

      {busy && !result ? <LoadingCard label="Observing, investigating, contesting and acting…" lines={6} /> : null}

      {!result && !busy ? (
        <EmptyState
          icon={Sparkles}
          title="Ready when you are"
          message="Pick a KPI and a quarter above, then run the investigation. Every number in the result is computed from your uploaded data, and every quotation comes from a document you uploaded."
          action={
            <button type="button" className="btn-primary" onClick={run}>
              <Play className="h-4 w-4" aria-hidden />
              Run investigation
            </button>
          }
        />
      ) : null}

      {result ? (
        <>
          <StageNav stage={stage} setStage={setStage} result={result} />
          {stage === 'observe' ? <ObserveStage result={result} isAnalyst={isAnalyst} /> : null}
          {stage === 'investigate' ? <InvestigateStage result={result} isAnalyst={isAnalyst} /> : null}
          {stage === 'contest' ? <ContestStage result={result} isAnalyst={isAnalyst} /> : null}
          {stage === 'act' ? <ActStage result={result} isAnalyst={isAnalyst} /> : null}
          <EngineFooter engine={result.engine} />
        </>
      ) : null}
    </div>
  )
}

function StageNav({ stage, setStage, result }) {
  const counts = {
    observe: result.observe?.top_drivers?.length || 0,
    investigate: result.investigate?.hypotheses?.length || 0,
    contest: result.contest?.ranking?.length || 0,
    act: result.act?.recommendations?.length || 0,
  }
  return (
    <nav className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4" aria-label="Investigation stages">
      {STAGES.map((s, i) => {
        const active = stage === s.key
        return (
          <button
            key={s.key}
            type="button"
            onClick={() => setStage(s.key)}
            aria-current={active ? 'step' : undefined}
            className="card card-pad text-left transition-shadow hover:shadow-sm"
            style={active ? { outline: '2px solid var(--series-1)', outlineOffset: '-1px' } : undefined}
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-ink-muted">Stage {i + 1}</span>
              <span className="tnum text-[11px] text-ink-muted">{counts[s.key]}</span>
            </div>
            <div className="mt-1 text-sm font-semibold text-ink">{s.title}</div>
            <div className="text-xs text-ink-secondary">{s.question}</div>
          </button>
        )
      })}
    </nav>
  )
}

function ObserveStage({ result, isAnalyst }) {
  const obs = result.observe
  const verdict = VERDICT_COPY[obs.verdict] || VERDICT_COPY.within_normal_variation
  return (
    <div className="space-y-4">
      <div className="card card-pad">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-wider text-ink-muted">
              {obs.kpi_label} · {obs.timeframe.pretty} vs {obs.baseline_timeframe.pretty}
            </div>
            <div className="mt-1 flex items-baseline gap-3">
              <span className="tnum text-3xl font-semibold text-ink">
                {formatValue(obs.current_value, obs.unit, { compact: true })}
              </span>
              <span
                className="tnum text-lg font-semibold"
                style={{ color: obs.is_unfavourable ? 'var(--delta-bad)' : 'var(--delta-good)' }}
              >
                {formatDelta(obs.change_pct)}
              </span>
            </div>
          </div>
          <Badge tone={verdict.tone}>{verdict.label}</Badge>
        </div>
        <p className="mt-2 text-sm text-ink-secondary">{verdict.blurb}</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card card-pad">
          <TrendChart observation={obs} />
        </div>
        <div className="card card-pad">
          <SectionTitle title="Drivers of the change" description="Members that moved the KPI more than their own size implies are the real drivers." />
          <ul className="space-y-2">
            {obs.top_drivers?.map((d) => (
              <li key={`${d.dimension}-${d.name}`} className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2" style={{ borderColor: 'var(--border)' }}>
                <div>
                  <div className="text-sm font-medium text-ink">{d.name}</div>
                  <div className="text-xs text-ink-muted">{titleCase(d.dimension)}</div>
                </div>
                <div className="flex items-center gap-3">
                  {d.is_disproportionate ? <Badge tone="critical">{d.over_index?.toFixed(1)}×</Badge> : null}
                  <span className="tnum text-sm font-semibold text-ink">{d.contribution_pct?.toFixed(0)}%</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {isAnalyst && obs.drivers ? (
        <div className="grid gap-4 lg:grid-cols-2">
          {Object.entries(obs.drivers).map(([dimension, rows]) => (
            <div key={dimension} className="card card-pad">
              <DriverChart dimension={titleCase(dimension)} rows={rows} unit={obs.unit} higherIsBetter={obs.higher_is_better} />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function InvestigateStage({ result, isAnalyst }) {
  const inv = result.investigate
  return (
    <div className="space-y-4">
      <Callout tone="info" title="How these were produced">
        {inv.method_note}
      </Callout>

      <div className="flex flex-wrap gap-2">
        <Badge tone="neutral">{inv.hypotheses.length} competing explanations</Badge>
        {inv.considered_count ? <Badge tone="neutral">{inv.considered_count} considered</Badge> : null}
        <Badge tone="neutral">{inv.documents_indexed} document chunks indexed</Badge>
        <Badge tone={inv.llm_used ? 'good' : 'warning'}>
          {inv.llm_used ? 'LLM framing enabled' : 'Deterministic framing'}
        </Badge>
        {Object.entries(inv.focus || {}).map(([dim, member]) => (
          <Badge key={dim} tone="critical">
            Focus: {member} ({dim})
          </Badge>
        ))}
      </div>

      {!inv.rag_available ? (
        <Callout tone="warning" title="No documents uploaded">
          Hypotheses are currently tested against structured data only. Upload operations reports, customer feedback or
          market notes on the Documents page so the system can corroborate — or contradict — them with written evidence.
        </Callout>
      ) : null}

      <div className="space-y-3">
        {inv.hypotheses.map((h, i) => (
          <article key={h.key} className="card card-pad">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-sm font-semibold text-ink sm:text-base">
                {i + 1}. {h.title}
              </h3>
              <Badge tone="neutral">{titleCase(h.family)}</Badge>
            </div>
            <p className="mt-1 text-sm text-ink-secondary">{h.statement}</p>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <Stat label="Structured tests" value={h.evidence?.length || 0} />
              <Stat label="Document passages" value={h.documentary_evidence?.length || 0} />
              <Stat label="Known gaps" value={h.missing?.length || 0} />
            </div>
            {h.documentary_evidence?.length ? (
              <div className="mt-3 space-y-1.5">
                {h.documentary_evidence.slice(0, 1).map((d) => (
                  <p key={d.chunk_id} className="flex gap-2 text-xs italic text-ink-secondary">
                    <Quote className="mt-0.5 h-3 w-3 shrink-0 text-ink-muted" aria-hidden />
                    “{d.quote}” — {d.source}
                  </p>
                ))}
              </div>
            ) : null}
          </article>
        ))}
      </div>

      <AnalystOnly show={isAnalyst} title="Explanations considered but not carried forward">
        <ul className="space-y-1 text-sm text-ink-secondary">
          {(inv.not_carried_forward || []).map((r) => (
            <li key={r.key}>
              <span className="font-medium text-ink">{r.title}</span> — {r.reason}
            </li>
          ))}
          {!inv.not_carried_forward?.length ? <li>Every applicable explanation was carried forward.</li> : null}
        </ul>
      </AnalystOnly>
    </div>
  )
}

function ContestStage({ result, isAnalyst }) {
  const contest = result.contest
  return (
    <div className="space-y-4">
      <div className="card card-pad">
        <SectionTitle
          title="Ranked by evidence, after being challenged"
          description="Each explanation was tested for timing, consistency across the business, counterexamples and contradictory documents."
        />
        <ol className="space-y-2">
          {contest.ranking.map((r) => (
            <li key={r.key} className="flex items-center gap-3">
              <span className="tnum w-5 text-sm text-ink-muted">{r.rank}</span>
              <span className="min-w-0 flex-1 truncate text-sm text-ink">{r.title}</span>
              <span className="tnum text-sm font-semibold text-ink">{r.confidence}%</span>
              <span className="w-28 text-xs text-ink-muted">{titleCase(r.band)}</span>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-xs text-ink-muted">{contest.confidence_disclaimer}</p>
      </div>

      {contest.ambiguity_note ? (
        <Callout tone="warning" title="The evidence does not settle this">
          {contest.ambiguity_note}
        </Callout>
      ) : null}

      <div className="space-y-3">
        {contest.hypotheses.map((h, i) => (
          <HypothesisCard key={h.key} hypothesis={h} rank={i + 1} isAnalyst={isAnalyst} defaultOpen={i === 0} />
        ))}
      </div>

      {contest.unresolved_questions?.length ? (
        <div className="card card-pad">
          <SectionTitle title="What we still do not know" description="Gaps that would most change the conclusion if filled." />
          <ul className="space-y-1.5 text-sm text-ink-secondary">
            {contest.unresolved_questions.map((q, i) => (
              <li key={i}>· {q}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )
}

function ActStage({ result }) {
  const act = result.act
  const n = act.narrative
  const story = act.llm_story && !act.llm_story.error ? act.llm_story : null

  return (
    <div className="space-y-4">
      <div className="card card-pad">
        <div className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Executive summary</div>
        <h3 className="mt-1 text-lg font-semibold text-ink">{n.headline}</h3>
        {story ? (
          <p className="mt-2 text-sm text-ink-secondary">{story.executive_summary}</p>
        ) : (
          <div className="mt-2 space-y-2 text-sm text-ink-secondary">
            <p>{n.what_changed}</p>
            <p>{n.how_significant}</p>
            <p>{n.what_drove_it}</p>
            <p>{n.leading_explanation}</p>
          </div>
        )}

        {story ? (
          <dl className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-ink-muted">What changed</dt>
              <dd className="mt-1 text-sm text-ink-secondary">{story.what_changed}</dd>
            </div>
            <div>
              <dt className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Why it likely happened</dt>
              <dd className="mt-1 text-sm text-ink-secondary">{story.why_it_likely_happened}</dd>
            </div>
          </dl>
        ) : null}
      </div>

      {(story?.what_we_cannot_yet_say || n.what_we_are_not_sure_about?.length) ? (
        <Callout tone="warning" title="What we cannot yet say">
          {story?.what_we_cannot_yet_say ? (
            <p>{story.what_we_cannot_yet_say}</p>
          ) : (
            <ul className="mt-1 space-y-1">
              {n.what_we_are_not_sure_about.map((u, i) => (
                <li key={i}>· {u}</li>
              ))}
            </ul>
          )}
        </Callout>
      ) : null}

      <SectionTitle title="Recommended next steps" description="Each is tied to the evidence that justifies it, and states what would change it." />
      <Recommendations recommendations={act.recommendations} limits={act.limits} />

      {story?.question_to_ask_the_team ? (
        <Callout tone="info" title="A question this analysis cannot answer">
          {story.question_to_ask_the_team}
        </Callout>
      ) : null}
    </div>
  )
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
      Pipeline: {engine.pipeline.join(' → ')} · completed in {engine.total_seconds}s ·{' '}
      {engine.llm?.enabled ? `reasoning layer: ${engine.llm.model}` : 'reasoning layer: deterministic fallback'} ·
      numbers computed by the structured analysis layer.
    </p>
  )
}
