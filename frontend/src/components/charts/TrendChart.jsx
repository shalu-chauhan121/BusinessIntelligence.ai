/**
 * KPI history by quarter, with the "normal range" the Observe stage computed
 * drawn as a band. A point outside the band is what "meaningful signal" means,
 * so the band is the explanation rather than decoration.
 *
 * One series, so no legend box is needed — the title names it. A reference band
 * and reference line are chrome, and are labelled directly.
 */
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import ChartFrame, { Legend, Tooltip } from './ChartFrame'
import { compactNumber, formatValue } from '../../lib/format'

export default function TrendChart({ observation, height = 260 }) {
  const series = observation?.series?.quarterly || []
  if (!series.length) return null

  const unit = observation.unit
  const current = observation.timeframe.label
  const baseline = observation.baseline_timeframe.label
  const range = observation.significance?.normal_range
  const expected = observation.significance?.expected_value

  const data = series.map((row) => ({
    period: row.period,
    value: row.value,
    isCurrent: row.period === current,
  }))

  const table = (
    <table className="w-full text-left text-xs">
      <thead className="sticky top-0" style={{ background: 'var(--surface-1)' }}>
        <tr className="text-ink-muted">
          <th className="py-1 pr-4 font-medium">Quarter</th>
          <th className="py-1 text-right font-medium">{observation.kpi_label}</th>
        </tr>
      </thead>
      <tbody className="text-ink-secondary">
        {data.map((d) => (
          <tr key={d.period} className={d.isCurrent ? 'font-semibold text-ink' : ''}>
            <td className="py-1 pr-4">{d.period}</td>
            <td className="tnum py-1 text-right">{formatValue(d.value, unit)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )

  return (
    <ChartFrame
      title={`${observation.kpi_label} by quarter`}
      subtitle={`Selected period ${current}, compared with ${baseline}. The shaded band is the range this KPI normally moves within for this comparison.`}
      height={height}
      table={table}
      legend={
        range ? (
          <Legend
            items={[
              { label: 'Actual', color: 'var(--series-1)' },
              { label: 'Expected', color: 'var(--text-muted)', dashed: true },
            ]}
          />
        ) : null
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
          <CartesianGrid vertical={false} strokeDasharray="0" />
          <XAxis dataKey="period" tickLine={false} axisLine={{ stroke: 'var(--baseline)' }} interval="preserveStartEnd" />
          <YAxis
            tickLine={false}
            axisLine={false}
            width={54}
            tickFormatter={(v) => (unit === 'percent' ? `${v.toFixed(0)}%` : compactNumber(v))}
          />
          {range ? (
            <ReferenceArea
              y1={range[0]}
              y2={range[1]}
              fill="var(--band-fill)"
              stroke="none"
              ifOverflow="extendDomain"
            />
          ) : null}
          {expected ? (
            <ReferenceLine
              y={expected}
              stroke="var(--text-muted)"
              strokeDasharray="4 4"
              label={{ value: 'expected', position: 'insideTopLeft', fill: 'var(--text-muted)', fontSize: 10 }}
            />
          ) : null}
          <Line
            type="monotone"
            dataKey="value"
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload, index } = props
              if (!payload?.isCurrent) return <circle key={index} cx={cx} cy={cy} r={2.5} fill="var(--series-1)" />
              return (
                <circle
                  key={index}
                  cx={cx}
                  cy={cy}
                  r={5}
                  fill={observation.anomaly ? 'var(--status-critical)' : 'var(--series-1)'}
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                />
              )
            }}
            activeDot={{ r: 5, stroke: 'var(--surface-1)', strokeWidth: 2 }}
            isAnimationActive={false}
          />
          <RTooltip
            cursor={{ stroke: 'var(--baseline)', strokeWidth: 1 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <Tooltip
                  title={label}
                  rows={[
                    { label: observation.kpi_label, value: formatValue(payload[0].value, unit), color: 'var(--series-1)' },
                    ...(range
                      ? [{ label: 'Normal range', value: `${formatValue(range[0], unit)} – ${formatValue(range[1], unit)}` }]
                      : []),
                  ]}
                />
              ) : null
            }
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}
