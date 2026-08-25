import { useEffect, useState } from 'react'
import { Search, Sparkles } from 'lucide-react'
import { Spinner } from './ui'

/**
 * Where an investigation now starts.
 *
 * The old entry point was a KPI dropdown, which forced the user to translate a
 * business question into the system's vocabulary before asking it. This takes
 * the question as written and lets the KPI contract work out which measure it
 * is about.
 */
export default function QuestionBar({ value, onChange, onSubmit, busy, examples = [] }) {
  const [text, setText] = useState(value || '')

  useEffect(() => {
    setText(value || '')
  }, [value])

  const submit = (event) => {
    event?.preventDefault()
    const question = text.trim()
    if (question && !busy) onSubmit(question)
  }

  return (
    <form className="card space-y-3" onSubmit={submit}>
      <label className="block text-sm font-medium text-slate-200" htmlFor="investigation-question">
        What would you like to understand?
      </label>

      <div className="flex flex-col gap-2 sm:flex-row">
        <div className="relative flex-1">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500"
            aria-hidden
          />
          <input
            id="investigation-question"
            type="text"
            className="input w-full pl-9"
            placeholder="Why did profit fall in Q4 even though revenue held steady?"
            value={text}
            maxLength={500}
            disabled={busy}
            onChange={(e) => {
              setText(e.target.value)
              onChange?.(e.target.value)
            }}
          />
        </div>
        <button type="submit" className="btn-primary shrink-0" disabled={busy || !text.trim()}>
          {busy ? <Spinner label="Investigating…" /> : (
            <>
              <Sparkles className="h-4 w-4" aria-hidden />
              Investigate
            </>
          )}
        </button>
      </div>

      {examples.length ? (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-500">Try:</span>
          {examples.map((example) => (
            <button
              key={example}
              type="button"
              className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300 transition hover:border-slate-500 hover:text-slate-100 disabled:opacity-50"
              disabled={busy}
              onClick={() => {
                setText(example)
                onChange?.(example)
                onSubmit(example)
              }}
            >
              {example}
            </button>
          ))}
        </div>
      ) : null}

      <p className="text-xs text-slate-500">
        Name the measure, the period and anything you are contrasting against — “even though”,
        “despite”, “while”. Every number in the answer is computed from your uploaded data.
      </p>
    </form>
  )
}
