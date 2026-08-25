import { CalendarRange } from 'lucide-react'

/**
 * The timeframe control. Everything downstream — KPIs, drivers, hypotheses,
 * evidence and recommendations — is recomputed for the selected year and
 * quarter on the server; nothing is filtered client-side.
 */
export default function TimeframePicker({
  timeframes = [],
  year,
  quarter,
  comparison,
  kpi,
  kpiOptions = [],
  onChange,
  busy = false,
  // The dashboard still browses by KPI, which is a legitimate way to look
  // around. Investigation no longer starts from a dropdown — a question names
  // the measure — so it hides this column rather than offering two entry points.
  showKpi = true,
}) {
  const years = [...new Set(timeframes.map((t) => t.year))].sort((a, b) => b - a)
  const quarters = timeframes.filter((t) => t.year === Number(year)).map((t) => t.quarter).sort()

  return (
    <div className="card card-pad">
      <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-ink-muted">
        <CalendarRange className="h-3.5 w-3.5" aria-hidden />
        Analysis period
      </div>
      <div className={`grid gap-3 sm:grid-cols-2 ${showKpi ? 'lg:grid-cols-4' : 'lg:grid-cols-3'}`}>
        <div>
          <label className="label" htmlFor="tf-year">Year</label>
          <select
            id="tf-year"
            className="field"
            value={year ?? ''}
            disabled={busy || !years.length}
            onChange={(e) => onChange({ year: Number(e.target.value), quarter })}
          >
            {years.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="label" htmlFor="tf-quarter">Quarter</label>
          <select
            id="tf-quarter"
            className="field"
            value={quarter ?? ''}
            disabled={busy}
            onChange={(e) => onChange({ year, quarter: e.target.value ? Number(e.target.value) : null })}
          >
            {quarters.map((q) => (
              <option key={q} value={q}>Q{q}</option>
            ))}
            <option value="">Full year</option>
          </select>
        </div>

        {showKpi ? (
          <div>
            <label className="label" htmlFor="tf-kpi">KPI</label>
            <select
              id="tf-kpi"
              className="field"
              value={kpi ?? ''}
              disabled={busy || !kpiOptions.length}
              onChange={(e) => onChange({ kpi: e.target.value })}
            >
              {kpiOptions.map((k) => (
                <option key={k.key} value={k.key}>
                  {k.label}
                  {k.granularity ? ` · ${k.granularity}` : ''}
                </option>
              ))}
            </select>
          </div>
        ) : null}

        <div>
          <label className="label" htmlFor="tf-comparison">Compare with</label>
          <select
            id="tf-comparison"
            className="field"
            value={comparison}
            disabled={busy}
            onChange={(e) => onChange({ comparison: e.target.value })}
          >
            <option value="previous_period">Previous quarter</option>
            <option value="year_over_year">Same quarter last year</option>
          </select>
        </div>
      </div>
      <p className="mt-3 text-xs text-ink-muted">
        Every figure on this page is recomputed from your uploaded data for the selected period.
      </p>
    </div>
  )
}
