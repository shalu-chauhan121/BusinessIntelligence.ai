import { AlertTriangle } from 'lucide-react'
import { titleCase } from '../../lib/format'

const TIME_GRAINS = ['day', 'week', 'month', 'quarter', 'year']

const ROLLUP_COPY = {
  sum: 'Summed — a flow that accumulates over the period.',
  mean: 'Averaged — a level measured at a point in time, which must never be summed.',
  last: 'Last value in the period.',
  recompute_from_components:
    'Recomputed from its numerator and denominator at every level. Averaging period ratios would give the wrong answer.',
  not_aggregatable: 'Cannot be rolled up — report it only at its own grain.',
}

/**
 * Granularity is a first-class part of a KPI definition, so it gets a first-class
 * control. The inferred value is pre-filled and clearly labelled as inferred;
 * saving it is what marks the grain as confirmed by a person.
 */
export default function GranularityPicker({ granularity, aggregation, dimensions = [], onChange }) {
  const entity = granularity.entity_grain || []

  const toggleDimension = (dim) => {
    const next = entity.includes(dim) ? entity.filter((d) => d !== dim) : [...entity, dim]
    onChange({ ...granularity, entity_grain: next, declared_by: 'user' })
  }

  return (
    <div className="space-y-3">
      {granularity.requires_confirmation ? (
        <p
          className="flex items-start gap-2 text-xs"
          style={{ color: 'var(--status-warning)' }}
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          This grain was inferred from the data's own cadence, not confirmed. Set it before
          approving — a KPI compared at the wrong grain is wrong.
        </p>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className="label" htmlFor="time-grain">
            Time grain
          </label>
          <select
            id="time-grain"
            className="field"
            value={granularity.time_grain}
            onChange={(e) =>
              onChange({ ...granularity, time_grain: e.target.value, declared_by: 'user' })
            }
          >
            {TIME_GRAINS.map((g) => (
              <option key={g} value={g}>
                {titleCase(g)}
              </option>
            ))}
          </select>
        </div>

        <div>
          <span className="label">Aggregates by</span>
          <div className="field" style={{ pointerEvents: 'none' }}>
            {(aggregation?.rollup_policy || 'sum').replace(/_/g, ' ')}
          </div>
        </div>
      </div>

      <p className="text-xs text-ink-secondary">
        {ROLLUP_COPY[aggregation?.rollup_policy] || ROLLUP_COPY.sum}
      </p>

      <div>
        <span className="label">Entity grain</span>
        <p className="mb-1.5 text-xs text-ink-muted">
          The level this KPI is reported at. Leave every box clear for a whole-company measure.
        </p>
        <div className="flex flex-wrap gap-1.5">
          {dimensions.length === 0 ? (
            <span className="text-xs text-ink-muted">
              This dataset has no dimension columns, so the only grain available is the whole
              business.
            </span>
          ) : null}
          {dimensions.map((dim) => {
            const active = entity.includes(dim)
            return (
              <button
                key={dim}
                type="button"
                className="chip"
                aria-pressed={active}
                onClick={() => toggleDimension(dim)}
                style={
                  active
                    ? { background: 'var(--series-1)', color: '#fff', borderColor: 'var(--series-1)' }
                    : undefined
                }
              >
                {titleCase(dim)}
              </button>
            )
          })}
        </div>
      </div>

      <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
        <div className="flex justify-between gap-3">
          <dt className="text-ink-muted">Resulting grain</dt>
          <dd className="text-ink">
            {(entity.length ? entity.join('-') : 'company') + '-' + granularity.time_grain}
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-ink-muted">Native row grain</dt>
          <dd className="truncate text-ink-secondary">
            {(granularity.native_row_grain || []).join(', ') || '—'}
          </dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-ink-muted">Declared by</dt>
          <dd className="text-ink-secondary">{granularity.declared_by}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-ink-muted">Can roll up to</dt>
          <dd className="truncate text-ink-secondary">
            {(granularity.valid_rollups || []).join(', ') || '—'}
          </dd>
        </div>
      </dl>
    </div>
  )
}
