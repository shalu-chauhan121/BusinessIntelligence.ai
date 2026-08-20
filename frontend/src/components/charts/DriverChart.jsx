/**
 * Which parts of the business moved the KPI.
 *
 * The job is polarity + magnitude, so the encoding is a diverging pair around a
 * zero baseline: one hue for members that pushed the KPI in the favourable
 * direction, the other for members that pushed against it. Bars are horizontal
 * because the category labels are words.
 */
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from 'recharts'
import ChartFrame, { Legend, Tooltip } from './ChartFrame'
import { compactNumber, formatDelta, formatValue } from '../../lib/format'

export default function DriverChart({ dimension, rows, unit, higherIsBetter = true, height }) {
  const data = (rows || []).filter((r) => r.change_abs !== null && r.change_abs !== undefined)
  if (!data.length) return null

  const chartHeight = height || Math.max(140, data.length * 34 + 40)
  const colorFor = (value) => {
    const favourable = higherIsBetter ? value > 0 : value < 0
    return favourable ? 'var(--diverge-pos)' : 'var(--diverge-neg)'
  }

  const table = (
    <table className="w-full text-left text-xs">
      <thead className="sticky top-0" style={{ background: 'var(--surface-1)' }}>
        <tr className="text-ink-muted">
          <th className="py-1 pr-3 font-medium">{dimension}</th>
          <th className="py-1 pr-3 text-right font-medium">Change</th>
          <th className="py-1 pr-3 text-right font-medium">%</th>
          <th className="py-1 text-right font-medium">Share of change</th>
        </tr>
      </thead>
      <tbody className="text-ink-secondary">
        {data.map((r) => (
          <tr key={r.name}>
            <td className="py-1 pr-3">{r.name}</td>
            <td className="tnum py-1 pr-3 text-right">{formatValue(r.change_abs, unit, { compact: true })}</td>
            <td className="tnum py-1 pr-3 text-right">{formatDelta(r.change_pct)}</td>
            <td className="tnum py-1 text-right">
              {r.contribution_pct === null || r.contribution_pct === undefined
                ? '—'
                : `${r.contribution_pct.toFixed(0)}%`}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )

  return (
    <ChartFrame
      title={`Contribution by ${dimension}`}
      subtitle="How much of the total change each part of the business accounts for."
      height={chartHeight}
      table={table}
      legend={
        <Legend
          items={[
            { label: higherIsBetter ? 'Pulled the KPI down' : 'Pulled the KPI up', color: 'var(--diverge-neg)' },
            { label: higherIsBetter ? 'Pushed it up' : 'Pushed it down', color: 'var(--diverge-pos)' },
          ]}
        />
      }
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 4 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} />
          <XAxis
            type="number"
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => (unit === 'percent' ? `${v.toFixed(0)}%` : compactNumber(v))}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={112}
            tickLine={false}
            axisLine={false}
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
          />
          <ReferenceLine x={0} stroke="var(--baseline)" />
          <Bar dataKey="change_abs" radius={[0, 4, 4, 0]} isAnimationActive={false} maxBarSize={20}>
            {data.map((row) => (
              <Cell key={row.name} fill={colorFor(row.change_abs)} />
            ))}
            <LabelList
              dataKey="contribution_pct"
              position="right"
              formatter={(v) => (v === null || v === undefined ? '' : `${v.toFixed(0)}%`)}
              style={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            />
          </Bar>
          <RTooltip
            cursor={{ fill: 'var(--plane)' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const r = payload[0].payload
              return (
                <Tooltip
                  title={r.name}
                  rows={[
                    { label: 'Change', value: formatValue(r.change_abs, unit, { compact: true }), color: colorFor(r.change_abs) },
                    { label: 'Change %', value: formatDelta(r.change_pct) },
                    {
                      label: 'Share of total change',
                      value: r.contribution_pct === null || r.contribution_pct === undefined ? '—' : `${r.contribution_pct.toFixed(0)}%`,
                    },
                    ...(r.over_index ? [{ label: 'Versus its own size', value: `${r.over_index.toFixed(1)}x` }] : []),
                  ]}
                />
              )
            }}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  )
}
