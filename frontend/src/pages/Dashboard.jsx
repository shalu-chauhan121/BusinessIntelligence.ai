import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Database, Info, TriangleAlert } from 'lucide-react'
import DriverChart from '../components/charts/DriverChart'
import TrendChart from '../components/charts/TrendChart'
import KpiCard from '../components/KpiCard'
import TimeframePicker from '../components/TimeframePicker'
import { AnalystOnly, Badge, Callout, EmptyState, ErrorState, LoadingCard, SectionTitle } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import useAnalysisSettings from '../hooks/useAnalysisSettings'
import { api } from '../lib/api'
import { VERDICT_COPY, formatDelta, formatValue, titleCase } from '../lib/format'

export default function Dashboard() {
  const { isAnalyst } = useAuth()
  const [settings, update] = useAnalysisSettings()
  const [state, setState] = useState({ data: null, error: null, loading: true })

  const load = useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }))
    try {
      const [data, telemetry] = await Promise.all([
        api.dashboard(settings),
        api.telemetrySummary().catch(() => null),
      ])
      data.telemetry = telemetry
      setState({ data, error: null, loading: false })
      const tf = data.observe?.timeframe
      if (tf && (settings.year !== tf.year || settings.quarter !== tf.quarter)) {
        update({ year: tf.year, quarter: tf.quarter })
      }
    } catch (error) {
      setState({ data: null, error, loading: false })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.year, settings.quarter, settings.kpi, settings.comparison])

  useEffect(() => {
    load()
  }, [load])

  if (state.loading && !state.data) return <LoadingCard label="Analysing your data…" lines={5} />

  if (state.error) {
    const noData = state.error.status === 409
    return noData ? (
      <EmptyState
        icon={Database}
        title="No business data yet"
        message="Upload a business metrics CSV, or load the bundled sample company, and the dashboard will be computed from it."
        action={
          <Link className="btn-primary" to="/data">
            Go to Business data
          </Link>
        }
      />
    ) : (
      <ErrorState title="Could not load the dashboard" message={state.error.message} onRetry={load} />
    )
  }

  const { observe: obs, timeframes, schema, dataset, telemetry } = state.data
  const verdict = VERDICT_COPY[obs.verdict] || VERDICT_COPY.within_normal_variation
  const primary = obs.kpi_scoreboard?.find((k) => k.is_primary)

  return (
    <div className="space-y-5">
      <SectionTitle
        eyebrow="Stage 1 — Observe"
        title="What actually changed?"
        description={`${dataset.filename} · ${dataset.rows?.toLocaleString()} rows · ${dataset.grain} grain`}
        right={
          <Link className="btn-primary" to="/investigation">
            Run full investigation
            <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
        }
      />

      <TimeframePicker
        timeframes={timeframes}
        year={settings.year ?? obs.timeframe.year}
        quarter={settings.quarter ?? obs.timeframe.quarter}
        kpi={settings.kpi}
        kpiOptions={schema.kpi_catalogue}
        comparison={settings.comparison}
        onChange={update}
        busy={state.loading}
      />

      <RuntimeTelemetry telemetry={telemetry} />

      {/* headline */}
      <div className="card card-pad">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="text-xs font-medium uppercase tracking-wider text-ink-muted">
              {obs.kpi_label} · {obs.timeframe.pretty} vs {obs.baseline_timeframe.pretty}
            </div>
            <div className="mt-1 flex flex-wrap items-baseline gap-3">
              <span className="tnum text-3xl font-semibold text-ink sm:text-4xl">
                {formatValue(obs.current_value, obs.unit, { compact: true })}
              </span>
              <span
                className="tnum text-lg font-semibold"
                style={{ color: obs.is_unfavourable ? 'var(--delta-bad)' : 'var(--delta-good)' }}
              >
                {formatDelta(obs.change_pct)}
              </span>
            </div>
            <p className="mt-2 max-w-2xl text-sm text-ink-secondary">
              {verdict.blurb}
              {obs.significance?.explanation ? ` ${obs.significance.explanation}` : ''}
            </p>
          </div>
          <Badge tone={verdict.tone} icon={obs.anomaly ? TriangleAlert : Info}>
            {verdict.label}
          </Badge>
        </div>

        <AnalystOnly show={isAnalyst} title="Significance method">
          <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
            {[
              ['Method', titleCase(obs.significance.method || '')],
              ['Robust z-score', obs.significance.robust_z?.toFixed(2)],
              ['Threshold', `±${obs.significance.z_threshold}`],
              ['Median historical change', formatDelta(obs.significance.median_historical_change_pct)],
              ['Robust sigma', formatDelta(obs.significance.robust_sigma_pct)],
              ['History points', obs.significance.history_points],
              ['Same-quarter points', obs.significance.same_quarter_points],
              ['Materiality floor', `±${obs.significance.material_threshold_pct}%`],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs text-ink-muted">{k}</dt>
                <dd className="tnum text-sm text-ink">{v ?? '—'}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-ink-secondary">{obs.significance.statistical_power}</p>
          {obs.significance.dispersion_note ? (
            <p className="mt-1 text-xs text-ink-secondary">{obs.significance.dispersion_note}</p>
          ) : null}
        </AnalystOnly>
      </div>

      {/* kpi scoreboard */}
      <section>
        <SectionTitle
          title="All KPIs for this period"
          description="Click a card to make that KPI the subject of the analysis."
        />
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {obs.kpi_scoreboard?.map((item) => (
            <KpiCard
              key={item.key}
              item={item}
              active={item.is_primary}
              onSelect={(key) => update({ kpi: key })}
            />
          ))}
        </div>
      </section>

      {/* charts */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card card-pad">
          <TrendChart observation={obs} />
        </div>
        <div className="card card-pad">
          {obs.top_drivers?.length ? (
            <>
              <SectionTitle
                title="What drove the change"
                description="Ranked by how much each part of the business moved the KPI relative to its own size."
              />
              <ul className="space-y-2">
                {obs.top_drivers.map((d) => (
                  <li
                    key={`${d.dimension}-${d.name}`}
                    className="flex items-center justify-between gap-3 rounded-lg border px-3 py-2"
                    style={{ borderColor: 'var(--border)' }}
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="truncate text-sm font-medium text-ink">{d.name}</span>
                        {d.is_disproportionate ? (
                          <Badge tone="critical">{d.over_index?.toFixed(1)}× its size</Badge>
                        ) : (
                          <span className="text-[11px] text-ink-muted">in line with its size</span>
                        )}
                      </div>
                      <div className="text-xs text-ink-muted">{titleCase(d.dimension)}</div>
                    </div>
                    <div className="text-right">
                      <div className="tnum text-sm font-semibold text-ink">{d.contribution_pct?.toFixed(0)}%</div>
                      <div className="tnum text-xs" style={{ color: 'var(--delta-bad)' }}>
                        {formatDelta(d.change_pct)}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <Callout tone="info" title="No dimension columns">
              Add columns such as region, product, channel or segment to your CSV to see which part of the business
              moved the KPI.
            </Callout>
          )}
        </div>
      </div>

      {isAnalyst && obs.drivers && Object.keys(obs.drivers).length ? (
        <section>
          <SectionTitle
            eyebrow="Analyst detail"
            title="Full driver decomposition"
            description="Contributions sum to 100% of the change within each dimension."
          />
          <div className="grid gap-4 lg:grid-cols-2">
            {Object.entries(obs.drivers).map(([dimension, rows]) => (
              <div key={dimension} className="card card-pad">
                <DriverChart
                  dimension={titleCase(dimension)}
                  rows={rows}
                  unit={obs.unit}
                  higherIsBetter={obs.higher_is_better}
                />
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {obs.data_warnings?.length ? (
        <Callout tone="warning" title="Data quality notes">
          <ul className="mt-1 space-y-1">
            {obs.data_warnings.map((w, i) => (
              <li key={i}>· {w}</li>
            ))}
          </ul>
        </Callout>
      ) : null}
    </div>
  )
}

function RuntimeTelemetry({ telemetry }) {
  const t = telemetry || {}
  const formatTokens = (value) => value >= 1000 ? `${(value / 1000).toFixed(value >= 100000 ? 0 : 1)}K` : (value || 0).toLocaleString()
  const formatCost = (value) => `$${Number(value || 0).toFixed(3)}`
  const stats = [
    ['Avg latency', `${t.average_latency_ms || 0} ms`],
    ['P95 latency', `${t.p95_latency_ms || 0} ms`],
    ['Model calls', `${t.average_model_calls || 0} / request`],
    ['Total tokens', formatTokens(t.total_tokens)],
    ['Avg tokens', formatTokens(t.average_total_tokens)],
    ['Estimated cost', formatCost(t.estimated_total_cost)],
    ['Cost / insight', formatCost(t.average_cost_per_request)],
  ]
  return (
    <section className="card card-pad">
      <SectionTitle
        eyebrow="Efficiency"
        title="Runtime Telemetry"
        description={`${t.total_requests || 0} insight requests tracked. Token and cost values are estimates when provider usage or pricing is unavailable.`}
      />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
        {stats.map(([label, value]) => (
          <div key={label} className="rounded-lg px-3 py-2" style={{ background: 'var(--plane)' }}>
            <div className="text-xs text-ink-muted">{label}</div>
            <div className="tnum mt-1 text-sm font-semibold text-ink">{value}</div>
          </div>
        ))}
      </div>
    </section>
  )
}
