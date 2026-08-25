import { HelpCircle } from 'lucide-react'

/**
 * What the system shows instead of guessing.
 *
 * When a question cannot be resolved to a measure this dataset holds, or names
 * a period it does not cover, the investigation stops here. Picking the closest
 * KPI would produce a confident answer to a question nobody asked, which is a
 * worse outcome than asking which one was meant.
 */
export default function ClarificationPrompt({ ambiguities = [], onChoose, busy }) {
  if (!ambiguities.length) return null

  return (
    <div className="card space-y-4 border-amber-500/40">
      <div className="flex items-start gap-3">
        <HelpCircle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" aria-hidden />
        <div>
          <h3 className="text-sm font-semibold text-slate-100">
            One thing before this can run
          </h3>
          <p className="text-xs text-slate-400">
            The question could not be resolved to something this dataset measures. Rather than
            answer a nearby question, here is what needs settling.
          </p>
        </div>
      </div>

      {ambiguities.map((ambiguity, index) => (
        <div key={`${ambiguity.kind}-${index}`} className="space-y-2 border-t border-slate-800 pt-3">
          <p className="text-sm text-slate-200">{ambiguity.message}</p>

          {(ambiguity.candidates || []).length ? (
            <div className="flex flex-wrap gap-2">
              {ambiguity.candidates.map((candidate) => (
                <button
                  key={candidate.kpi_key}
                  type="button"
                  className="rounded-md border border-slate-700 px-3 py-1.5 text-xs text-slate-200 transition hover:border-sky-500 hover:text-sky-300 disabled:opacity-50"
                  disabled={busy}
                  onClick={() => onChoose?.(candidate)}
                >
                  {candidate.label || candidate.kpi_key}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  )
}
