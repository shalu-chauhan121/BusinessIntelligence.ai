import { useEffect, useState } from 'react'
import { LineChart } from 'lucide-react'
import { api } from '../../lib/api'
import { formatValue } from '../../lib/format'
import { Spinner } from '../ui'

/**
 * Compute a definition over recent periods without approving it.
 *
 * This is the check that stops a plausible-looking formula becoming
 * authoritative before anyone has seen the numbers it actually produces.
 */
export default function KpiPreview({ kpiId, datasetId, reloadKey }) {
  const [state, setState] = useState({ data: null, error: null, loading: true })

  useEffect(() => {
    let cancelled = false
    setState({ data: null, error: null, loading: true })
    api
      .kpiPreview(kpiId, datasetId)
      .then((data) => !cancelled && setState({ data, error: null, loading: false }))
      .catch((error) => !cancelled && setState({ data: null, error, loading: false }))
    return () => {
      cancelled = true
    }
  }, [kpiId, datasetId, reloadKey])

  if (state.loading) return <Spinner label="Computing a preview…" />

  if (state.error) {
    return (
      <p className="text-xs" style={{ color: 'var(--status-critical)' }}>
        This definition could not be computed against the data: {state.error.message}
      </p>
    )
  }

  const { data } = state
  const values = data.points.map((p) => p.value).filter((v) => v !== null)
  const max = Math.max(...values, 0)
  const min = Math.min(...values, 0)
  const span = max - min || 1

  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-xs text-ink-muted">
        <LineChart className="h-3.5 w-3.5" aria-hidden />
        Preview · {data.granularity} · not yet approved
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-ink-muted">
            <tr>
              <th className="py-1 pr-3 font-medium">Period</th>
              <th className="py-1 pr-3 font-medium">Value</th>
              <th className="py-1 font-medium" aria-hidden />
            </tr>
          </thead>
          <tbody className="text-ink-secondary">
            {data.points.map((p) => (
              <tr key={p.period}>
                <td className="py-1 pr-3 font-mono text-ink">{p.period}</td>
                <td className="tnum py-1 pr-3 text-ink">
                  {p.value === null ? '—' : formatValue(p.value, data.unit, { compact: true })}
                </td>
                <td className="py-1" style={{ width: '45%' }}>
                  {p.value === null ? null : (
                    <span
                      className="block h-1.5 rounded-full"
                      style={{
                        width: `${Math.max(2, ((p.value - min) / span) * 100)}%`,
                        background: 'var(--series-1)',
                      }}
                    />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-ink-muted">{data.note}</p>
    </div>
  )
}
