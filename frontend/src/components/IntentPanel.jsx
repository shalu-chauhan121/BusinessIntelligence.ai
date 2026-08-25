import { AlertTriangle, Info, Target } from 'lucide-react'
import { Badge } from './ui'

/**
 * How the system read the question.
 *
 * Shown above every result so a misreading is visible before it is acted on.
 * An assumed period or comparison is disclosed here rather than buried, because
 * a reader who does not know a default was applied cannot tell they were
 * answered about a different period than the one they meant.
 */
export default function IntentPanel({ intent, assumptions = [], onCorrect }) {
  if (!intent) return null

  const outcome = intent.outcome
  const period = intent.period
  const notes = assumptions.length ? assumptions : (intent.assumptions || [])

  return (
    <div className="card space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Target className="h-4 w-4 text-sky-400" aria-hidden />
        <span className="text-xs uppercase tracking-wide text-slate-500">Read as</span>

        {outcome ? (
          <Badge tone="info">{outcome.label || outcome.kpi_key}</Badge>
        ) : (
          <Badge tone="warn">no measure resolved</Badge>
        )}

        {period ? (
          <Badge>
            {period.quarter ? `Q${period.quarter} ${period.year}` : period.year}
            {period.comparison === 'year_over_year' ? ' vs a year ago' : ' vs the previous period'}
          </Badge>
        ) : null}

        {(intent.comparison_kpis || []).map((k) => (
          <Badge key={k.kpi_key} tone="muted">
            contrasted with {k.label || k.kpi_key}
          </Badge>
        ))}

        {onCorrect ? (
          <button
            type="button"
            className="ml-auto text-xs text-slate-400 underline underline-offset-2 hover:text-slate-200"
            onClick={onCorrect}
          >
            Not what I meant
          </button>
        ) : null}
      </div>

      {notes.length ? (
        <ul className="space-y-1">
          {notes.map((note) => (
            <li key={note} className="flex items-start gap-2 text-xs text-amber-300/90">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
              <span>{note}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {(intent.unmapped_terms || []).length ? (
        <p className="flex items-start gap-2 text-xs text-slate-500">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          <span>
            Nothing in this dataset measures{' '}
            <span className="text-slate-300">{intent.unmapped_terms.slice(0, 4).join(', ')}</span>,
            so those parts of the question were not investigated.
          </span>
        </p>
      ) : null}
    </div>
  )
}
