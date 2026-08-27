import { CheckCircle2, CircleAlert, CircleHelp, CircleOff, Layers } from 'lucide-react'
import { Badge, SectionTitle } from './ui'

const FRESHNESS = {
  current: { tone: 'good', icon: CheckCircle2, label: 'Current' },
  stale: { tone: 'warning', icon: CircleAlert, label: 'Stale' },
  missing: { tone: 'critical', icon: CircleOff, label: 'Missing' },
  freshness_unknown: { tone: 'neutral', icon: CircleHelp, label: 'Not asserted' },
}

const MODE = {
  overlap: 'Overlapping — compared, never summed',
  mixed: 'Mixed — stitched where sources differ, compared where they overlap',
  partition_or_complementary: 'Stitched — one source per cell',
}

/**
 * Per-source provenance for a canonical view reconciled from several sources.
 * Everything here is computed deterministically; none of it is an LLM judgement.
 * Renders nothing for an ordinary single-file dataset.
 */
export default function SourcePanel({ sources }) {
  if (!sources) return null
  const rows = sources.sources || []
  const modes = sources.measure_modes || {}
  const withheld = Object.entries(sources.withheld_kpis || {})
  const pending = Object.entries(sources.periods_pending || {})

  return (
    <div className="card card-pad space-y-3">
      <SectionTitle
        eyebrow="Reconciled business view"
        title="Where this data came from"
        description={
          `Stitched on ${(sources.join_keys || []).join(' × ') || 'the business grain'} at ` +
          `${sources.canonical_grain} grain` +
          (sources.grain_basis === 'contract'
            ? ' (the grain the KPI contract asks for).'
            : sources.grain_basis === 'contract_floored'
              ? ' — the contract asks for a finer grain than these sources can serve, and coarser data is never rolled down.'
              : ' (no contract grain declared yet, so the finest every source can serve).')
        }
      />

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wide text-ink-muted">
              <th className="pb-2 pr-3 font-medium">Source</th>
              <th className="pb-2 pr-3 font-medium">Last refresh</th>
              <th className="pb-2 pr-3 font-medium">Cadence</th>
              <th className="pb-2 pr-3 font-medium">Freshness</th>
              <th className="pb-2 pr-3 font-medium">Rollup</th>
              <th className="pb-2 font-medium">Contributes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => {
              const f = FRESHNESS[s.freshness] || FRESHNESS.freshness_unknown
              return (
                <tr key={s.source_id} className="border-t align-top" style={{ borderColor: 'var(--border)' }}>
                  <td className="py-2 pr-3 font-medium text-ink">{s.label || s.source_id}</td>
                  <td className="py-2 pr-3 text-ink-secondary">
                    <span className="tnum">{s.last_refresh_at || '—'}</span>
                    {/* the basis matters: an upload time says nothing about the source system */}
                    <div className="text-[11px] text-ink-muted">{s.last_refresh_basis}</div>
                  </td>
                  <td className="py-2 pr-3 text-ink-secondary">{s.refresh_cadence}</td>
                  <td className="py-2 pr-3"><Badge tone={f.tone} icon={f.icon}>{f.label}</Badge></td>
                  <td className="py-2 pr-3 text-[11px] text-ink-muted">{s.rollup_applied}</td>
                  <td className="py-2 text-ink-secondary">
                    {(s.fields_contributed || []).join(', ') || '—'}
                    {s.rows_dropped_as_future ? (
                      <div className="text-[11px] text-ink-muted">
                        {s.rows_dropped_as_future} row(s) dropped as future data
                      </div>
                    ) : null}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {rows.some((s) => s.note) ? (
        <ul className="space-y-1 text-xs text-ink-muted">
          {rows.filter((s) => s.note).map((s) => (
            <li key={s.source_id}>
              <span className="font-medium">{s.label || s.source_id}:</span> {s.note}
            </li>
          ))}
        </ul>
      ) : null}

      {pending.length ? (
        <div className="text-xs text-ink-muted">
          <span className="font-medium text-ink-secondary">Pending, not missing:</span>{' '}
          {pending.map(([measure, periods], i) => (
            <span key={measure}>
              {i > 0 ? '; ' : ''}
              <span className="font-medium">{measure}</span> has not caught up for the most
              recent {periods.length} period{periods.length === 1 ? '' : 's'}
            </span>
          ))}
        </div>
      ) : null}

      {withheld.length ? (
        <div
          className="rounded-lg border-l-4 px-3 py-2 text-sm"
          style={{ borderColor: 'var(--status-warning)', background: 'var(--plane)' }}
        >
          <div className="font-medium text-ink">
            {withheld.length} KPI{withheld.length === 1 ? '' : 's'} withheld
          </div>
          <p className="mt-0.5 text-xs text-ink-secondary">
            Withheld rather than computed from incomplete data — a partial total would look like a
            real one.
          </p>
          <ul className="mt-1 space-y-1 text-xs text-ink-secondary">
            {withheld.map(([kpi, reason]) => (
              <li key={kpi}>{reason}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {Object.keys(modes).length ? (
        <details className="text-xs text-ink-muted">
          <summary className="cursor-pointer select-none font-medium text-ink-secondary">
            <Layers className="mr-1 inline h-3.5 w-3.5" aria-hidden />
            How each measure was reconciled
          </summary>
          <ul className="mt-1.5 space-y-1">
            {Object.entries(modes).map(([measure, info]) => (
              <li key={measure} className="tnum">
                <span className="font-medium text-ink-secondary">{measure}</span> —{' '}
                {MODE[info.mode] || info.mode}
                {info.cells_corroborated ? `, ${info.cells_corroborated} corroborated` : ''}
                {info.cells_disputed ? `, ${info.cells_disputed} disputed` : ''}
                {' '}({(info.sources || []).join(', ')})
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  )
}
