import { BookOpen, Boxes, Sigma, Sparkles, UserPen } from 'lucide-react'
import { SectionTitle } from '../ui'
import { StatusBadge } from './KpiDefinitionEditor'

const GROUPS = [
  {
    key: 'general_library',
    icon: BookOpen,
    title: 'From the general KPI library',
    blurb:
      'Broadly applicable KPIs that bound to real columns in this dataset. Library entries that could not bind are listed further down — they were considered, not missed.',
  },
  {
    key: 'discovered_atomic',
    icon: Boxes,
    title: 'Measured directly in this dataset',
    blurb: 'Columns that are KPIs in their own right, aggregated according to what they measure.',
  },
  {
    key: 'discovered_derived',
    icon: Sigma,
    title: 'Derived from combinations of fields',
    blurb:
      'Each of these passed a data check, not just a type check — a rate is only proposed where the numerator is genuinely contained by the denominator on the rows.',
  },
  {
    key: 'llm_suggested',
    icon: Sparkles,
    title: 'Suggested by semantic screening',
    blurb:
      'Proposed by the model and re-validated against the column list. Never authoritative without your approval.',
  },
  {
    key: 'user_defined',
    icon: UserPen,
    title: 'Defined by your team',
    blurb: 'Organisation-specific metrics that cannot be inferred from the data alone.',
  },
]

/** The review queue, grouped by where each definition came from. */
export default function KpiProposalList({ groups = {}, selectedId, onSelect }) {
  return (
    <div className="space-y-5">
      {GROUPS.map(({ key, icon: Icon, title, blurb }) => {
        const items = groups[key] || []
        if (!items.length) return null
        return (
          <section key={key}>
            <SectionTitle
              title={
                <span className="inline-flex items-center gap-2">
                  <Icon className="h-4 w-4 text-ink-muted" aria-hidden />
                  {title}
                </span>
              }
              description={blurb}
            />
            <ul className="space-y-1.5">
              {items.map((kpi) => (
                <li key={kpi.kpi_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(kpi.kpi_id)}
                    aria-pressed={selectedId === kpi.kpi_id}
                    className="card w-full px-3 py-2 text-left transition-shadow hover:shadow-md"
                    style={
                      selectedId === kpi.kpi_id
                        ? { outline: '2px solid var(--series-1)', outlineOffset: '-1px' }
                        : undefined
                    }
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="truncate text-sm font-medium text-ink">{kpi.name}</span>
                      <StatusBadge status={kpi.status} />
                    </div>
                    <div className="mt-0.5 truncate font-mono text-xs text-ink-muted">
                      {kpi.formula.kind === 'ratio'
                        ? `${kpi.formula.numerator_expression} / ${kpi.formula.denominator_expression}` +
                          (kpi.formula.scale !== 1 ? ` × ${kpi.formula.scale}` : '')
                        : `${kpi.formula.kind}(${kpi.formula.expression})`}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-muted">
                      <span className="chip">{kpi.granularity.label || 'grain not set'}</span>
                      <span className="chip">{kpi.unit}</span>
                      {kpi.granularity.requires_confirmation ? (
                        <span className="chip" style={{ color: 'var(--status-warning)' }}>
                          confirm grain
                        </span>
                      ) : null}
                    </div>
                    {kpi.relevance ? (
                      <p className="mt-1 line-clamp-2 text-xs text-ink-secondary">{kpi.relevance}</p>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )
      })}
    </div>
  )
}
