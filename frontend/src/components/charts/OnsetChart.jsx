/**
 * The temporal-precedence picture: when did the KPI turn, and when did the
 * proposed cause turn?
 *
 * Two measures on different scales are NEVER given two y-axes here. Instead both
 * series are standardised against their OWN pre-period normal — plotted in robust
 * standard deviations from their own median — so a single axis carries both and
 * the question the chart exists to answer (which line moved first) stays readable.
 *
 * Standardising beats indexing to 100 here: a rate that starts near zero, such as
 * a stockout rate, produces a meaningless 1200-index spike that flattens the other
 * series off the chart. In sigma units both stay legible, and the units match the
 * onset test that produced the markers.
 */
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import ChartFrame, { Legend, Tooltip } from './ChartFrame'

function standardise(rows, baselineCount) {
  const values = rows.map((r) => r.value).filter((v) => typeof v === 'number' && !Number.isNaN(v))
  const base = values.slice(0, baselineCount)
  if (base.length < 3) return rows.map((r) => ({ ...r, indexed: null }))
  const sorted = [...base].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  const deviations = base.map((v) => Math.abs(v - median)).sort((a, b) => a - b)
  let sigma = 1.4826 * deviations[Math.floor(deviations.length / 2)]
  if (!sigma) {
    const mean = base.reduce((a, b) => a + b, 0) / base.length
    sigma = Math.sqrt(base.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(base.length - 1, 1))
  }
  if (!sigma) return rows.map((r) => ({ ...r, indexed: null }))
  return rows.map((r) => ({ ...r, indexed: typeof r.value === 'number' ? (r.value - median) / sigma : null }))
}

export default function OnsetChart({ kpiLabel, causeLabel, kpiSeries, causeSeries, kpiOnset, causeOnset, height = 260 }) {
  if (!kpiSeries?.length || !causeSeries?.length) return null
  const baselineCount = Math.max(6, Math.floor(kpiSeries.length / 3))
  const kpi = standardise(kpiSeries, baselineCount)
  const cause = standardise(causeSeries, baselineCount)
  const causeByWeek = new Map(cause.map((c) => [c.week, c.indexed]))
  const data = kpi.map((k) => ({ week: k.week, kpi: k.indexed, cause: causeByWeek.get(k.week) ?? null }))

  const table = (
    <table className="w-full text-left text-xs">
      <thead className="sticky top-0" style={{ background: 'var(--surface-1)' }}>
        <tr className="text-ink-muted">
          <th className="py-1 pr-3 font-medium">Week</th>
          <th className="py-1 pr-3 text-right font-medium">{kpiLabel} (indexed)</th>
          <th className="py-1 text-right font-medium">{causeLabel} (indexed)</th>
        </tr>
      </thead>
      <tbody className="text-ink-secondary">
        {data.map((d) => (
          <tr key={d.week}>
            <td className="py-1 pr-3">{d.week}</td>
            <td className="tnum py-1 pr-3 text-right">{d.kpi === null ? '—' : d.kpi.toFixed(1)}</td>
            <td className="tnum py-1 text-right">{d.cause === null ? '—' : d.cause.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )

  return (
    <ChartFrame
      title="Which moved first?"
      subtitle="Both series expressed as deviations from their own pre-period normal, in robust standard deviations, so a single scale carries both. The markers show where each series left its normal band."
      height={height}
      table={table}
      legend={
        <Legend
          items={[
            { label: kpiLabel, color: 'var(--series-1)' },
            { label: causeLabel, color: 'var(--series-2)' },
          ]}
        />
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 16, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="week" tickLine={false} axisLine={{ stroke: 'var(--baseline)' }} minTickGap={28} />
          <YAxis tickLine={false} axisLine={false} width={44} tickFormatter={(v) => `${v.toFixed(0)}σ`} />
          <ReferenceLine y={0} stroke="var(--baseline)" />
          {kpiOnset ? (
            <ReferenceLine
              x={kpiOnset}
              stroke="var(--series-1)"
              strokeDasharray="4 3"
              label={{ value: 'KPI turns', position: 'top', fill: 'var(--series-1)', fontSize: 10 }}
            />
          ) : null}
          {causeOnset ? (
            <ReferenceLine
              x={causeOnset}
              stroke="var(--series-2)"
              strokeDasharray="4 3"
              label={{ value: 'cause turns', position: 'insideBottom', fill: 'var(--series-2)', fontSize: 10 }}
            />
          ) : null}
          <Line type="monotone" dataKey="kpi" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="cause" stroke="var(--series-2)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <RTooltip
            cursor={{ stroke: 'var(--baseline)', strokeWidth: 1 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <Tooltip
                  title={label}
                  rows={payload.map((p) => ({
                    label: p.dataKey === 'kpi' ? kpiLabel : causeLabel,
                    value: p.value === null ? '—' : `${p.value.toFixed(1)}σ from its own normal`,
                    color: p.stroke,
                  }))}
                />
              ) : null
            }
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}
