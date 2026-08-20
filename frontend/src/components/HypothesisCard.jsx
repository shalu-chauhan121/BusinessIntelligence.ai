import { useState } from 'react'
import { ChevronDown, Clock, GitCompareArrows, ShieldAlert } from 'lucide-react'
import ConfidenceMeter from './ConfidenceMeter'
import OnsetChart from './charts/OnsetChart'
import ReasoningTrail from './ReasoningTrail'
import { AnalystOnly, Badge, Callout } from './ui'
import { DocumentEvidence, MissingEvidence, StructuredEvidence } from './EvidencePanel'
import { titleCase } from '../lib/format'

const STATUS_TONE = {
  well_supported: 'good',
  partially_supported: 'warning',
  weakly_supported: 'serious',
  not_supported: 'critical',
}

const TEMPORAL_TONE = {
  cause_precedes_kpi: 'good',
  simultaneous: 'warning',
  kpi_precedes_cause: 'critical',
  undetermined: 'neutral',
  not_applicable: 'neutral',
}

export default function HypothesisCard({ hypothesis, rank, isAnalyst, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const scoring = hypothesis.scoring || {}
  const contest = hypothesis.contest || {}
  const temporal = contest.temporal || {}
  const mechanism = contest.mechanism || {}
  const consistency = contest.consistency || {}
  const contradicting = contest.contradictory_evidence || []

  return (
    <article className="card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-start gap-3 p-4 text-left sm:p-5"
      >
        <span
          className="tnum mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold"
          style={{ background: 'var(--plane)', color: 'var(--text-secondary)' }}
        >
          {rank}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-ink sm:text-base">{hypothesis.title}</h3>
            <Badge tone={STATUS_TONE[scoring.status] || 'neutral'}>{titleCase(scoring.status || '')}</Badge>
            {scoring.causally_consistent ? <Badge tone="good">Timing holds</Badge> : null}
            {temporal.status === 'kpi_precedes_cause' ? (
              <Badge tone="critical" icon={Clock}>Started too late</Badge>
            ) : null}
            {mechanism.reverse_causation_suspected ? (
              <Badge tone="serious" icon={ShieldAlert}>Reverse causation risk</Badge>
            ) : null}
          </div>
          <p className="mt-1 text-sm text-ink-secondary">{hypothesis.statement}</p>
          {hypothesis.analyst_note ? (
            <p className="mt-1 text-xs text-ink-muted">{hypothesis.analyst_note}</p>
          ) : null}
        </div>

        <div className="hidden shrink-0 sm:block">
          <ConfidenceMeter confidence={scoring.confidence} band={scoring.confidence_band} compact />
        </div>
        <ChevronDown
          className={`mt-1 h-4 w-4 shrink-0 text-ink-muted transition-transform ${open ? 'rotate-180' : ''}`}
          aria-hidden
        />
      </button>

      <div className="px-4 pb-3 sm:hidden">
        <ConfidenceMeter confidence={scoring.confidence} band={scoring.confidence_band} />
      </div>

      {open ? (
        <div className="border-t px-4 pb-5 pt-4 sm:px-5" style={{ borderColor: 'var(--border)' }}>
          {scoring.cap_reason ? (
            <div className="mb-4">
              <Callout tone="critical" title="Confidence was capped">{scoring.cap_reason}</Callout>
            </div>
          ) : null}

          <div className="mb-4">
            <Callout tone="info" title="What this confidence means">
              {scoring.causal_claim}
            </Callout>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Supporting evidence — measured
              </h4>
              <StructuredEvidence
                items={(hypothesis.evidence || []).filter((e) => e.stance !== 'contradicting')}
                showWeights={isAnalyst}
              />

              <h4 className="mb-2 mt-5 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Supporting evidence — documents
              </h4>
              <DocumentEvidence
                items={hypothesis.documentary_evidence}
                emptyMessage="Nothing in your uploaded documents supports this explanation."
              />
            </section>

            <section>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Contradictory evidence
              </h4>
              <div className="space-y-3">
                <div
                  className="rounded-lg border px-3 py-2"
                  style={{ borderColor: 'var(--border)' }}
                >
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                    <Clock className="h-3.5 w-3.5" aria-hidden />
                    Temporal precedence
                  </div>
                  <p className="mt-1 text-sm text-ink-secondary">{temporal.detail}</p>
                  {temporal.status ? (
                    <div className="mt-1.5">
                      <Badge tone={TEMPORAL_TONE[temporal.status] || 'neutral'}>{titleCase(temporal.status)}</Badge>
                    </div>
                  ) : null}
                </div>

                <div className="rounded-lg border px-3 py-2" style={{ borderColor: 'var(--border)' }}>
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                    <GitCompareArrows className="h-3.5 w-3.5" aria-hidden />
                    Consistency across the business
                  </div>
                  <p className="mt-1 text-sm text-ink-secondary">
                    {consistency.correlation?.interpretation || consistency.detail || 'Not applicable.'}
                  </p>
                  {consistency.counterexamples?.length ? (
                    <p className="mt-1 text-sm" style={{ color: 'var(--status-critical)' }}>
                      Counterexamples:{' '}
                      {consistency.counterexamples.map((c) => c.member).join(', ')} — the KPI fell there without the
                      proposed cause moving.
                    </p>
                  ) : null}
                </div>

                {mechanism.declared_risk ? (
                  <div className="rounded-lg border px-3 py-2" style={{ borderColor: 'var(--border)' }}>
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
                      <ShieldAlert className="h-3.5 w-3.5" aria-hidden />
                      Reverse-causation screen
                    </div>
                    <p className="mt-1 text-sm text-ink-secondary">{mechanism.declared_risk}</p>
                    <p className="mt-1 text-sm text-ink-secondary">{mechanism.conclusion}</p>
                  </div>
                ) : null}

                <DocumentEvidence
                  items={contradicting}
                  stance="contradicting"
                  emptyMessage="No passage in your documents argues against this explanation."
                />
              </div>

              <h4 className="mb-2 mt-5 text-xs font-semibold uppercase tracking-wider text-ink-muted">
                Missing evidence
              </h4>
              <MissingEvidence items={hypothesis.missing} />
            </section>
          </div>

          {contest.temporal_series?.kpi?.length ? (
            <AnalystOnly show={isAnalyst} title="Weekly onset comparison">
              <OnsetChart
                kpiLabel="KPI"
                causeLabel={temporal.cause_metric_label || 'Proposed cause'}
                kpiSeries={contest.temporal_series.kpi}
                causeSeries={contest.temporal_series.cause}
                kpiOnset={temporal.kpi_onset_week}
                causeOnset={temporal.cause_onset_week}
              />
              {mechanism.lead_lag ? (
                <p className="mt-2 text-xs text-ink-secondary">{mechanism.lead_lag.detail}</p>
              ) : null}
            </AnalystOnly>
          ) : null}

          <AnalystOnly show={isAnalyst} title="How this confidence was computed">
            <p className="mb-2 text-xs text-ink-secondary tnum">
              support {scoring.support_score} · against {scoring.against_score} · missing-evidence penalty{' '}
              {scoring.missing_penalty}
            </p>
            <table className="w-full text-left text-xs">
              <thead className="text-ink-muted">
                <tr>
                  <th className="py-1 pr-3 font-medium">Side</th>
                  <th className="py-1 pr-3 font-medium">Source</th>
                  <th className="py-1 pr-3 font-medium">Evidence</th>
                  <th className="py-1 text-right font-medium">Weight</th>
                </tr>
              </thead>
              <tbody className="text-ink-secondary">
                {(scoring.score_ledger || []).map((row, i) => (
                  <tr key={i}>
                    <td className="py-1 pr-3" style={{ color: row.side === 'against' ? 'var(--status-critical)' : 'var(--status-good)' }}>
                      {row.side}
                    </td>
                    <td className="py-1 pr-3">{titleCase(row.source)}</td>
                    <td className="py-1 pr-3">{row.label}</td>
                    <td className="tnum py-1 text-right">{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AnalystOnly>

          <details className="mt-3 rounded-lg border" style={{ borderColor: 'var(--border)' }}>
            <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">
              Reasoning trail
            </summary>
            <div className="border-t px-3 py-3" style={{ borderColor: 'var(--border)' }}>
              <ReasoningTrail steps={hypothesis.reasoning_trail} />
            </div>
          </details>
        </div>
      ) : null}
    </article>
  )
}
