/**
 * Shared chart chrome: title, optional subtitle, legend slot and a table view.
 *
 * The table view exists so identity is never carried by colour alone and so the
 * numbers are readable by a screen reader — every chart in this app has one.
 */
import { useState } from 'react'
import { Table2, LineChart as LineIcon } from 'lucide-react'

export default function ChartFrame({ title, subtitle, legend, table, height = 240, children }) {
  const [showTable, setShowTable] = useState(false)
  return (
    <figure className="m-0">
      <figcaption className="mb-2 flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-ink">{title}</div>
          {subtitle ? <div className="mt-0.5 text-xs text-ink-secondary">{subtitle}</div> : null}
        </div>
        <div className="flex items-center gap-3">
          {legend}
          {table ? (
            <button
              type="button"
              onClick={() => setShowTable((v) => !v)}
              className="inline-flex items-center gap-1 text-xs text-ink-muted hover:text-ink"
              aria-pressed={showTable}
            >
              {showTable ? <LineIcon className="h-3.5 w-3.5" /> : <Table2 className="h-3.5 w-3.5" />}
              {showTable ? 'Chart' : 'Table'}
            </button>
          ) : null}
        </div>
      </figcaption>

      {showTable && table ? (
        <div className="overflow-x-auto" style={{ maxHeight: height + 40 }}>
          {table}
        </div>
      ) : (
        <div style={{ width: '100%', height }}>{children}</div>
      )}
    </figure>
  )
}

export function Legend({ items }) {
  return (
    <ul className="flex flex-wrap items-center gap-3">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5 text-xs text-ink-secondary">
          <span
            aria-hidden
            className="inline-block rounded-[2px]"
            style={{
              width: it.dashed ? 14 : 10,
              height: it.dashed ? 0 : 10,
              borderTop: it.dashed ? `2px dashed ${it.color}` : undefined,
              background: it.dashed ? undefined : it.color,
            }}
          />
          {it.label}
        </li>
      ))}
    </ul>
  )
}

export function Tooltip({ title, rows }) {
  return (
    <div
      className="rounded-lg border px-3 py-2 text-xs shadow-lg"
      style={{ borderColor: 'var(--border)', background: 'var(--surface-1)' }}
    >
      <div className="mb-1 font-semibold text-ink">{title}</div>
      <ul className="space-y-0.5">
        {rows.map((r) => (
          <li key={r.label} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-ink-secondary">
              {r.color ? (
                <span className="inline-block h-2 w-2 rounded-[2px]" style={{ background: r.color }} aria-hidden />
              ) : null}
              {r.label}
            </span>
            <span className="tnum font-medium text-ink">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
