import { CONFIDENCE_BANDS } from '../lib/format'

/**
 * Evidence strength, drawn as a meter rather than a chart.
 *
 * The band label is always shown next to the bar: the number is not a
 * probability, so the words carry the meaning and the colour only reinforces it.
 */
export default function ConfidenceMeter({ confidence, band, compact = false }) {
  const meta = CONFIDENCE_BANDS[band] || CONFIDENCE_BANDS.insufficient
  return (
    <div className={compact ? 'w-32' : 'w-full'}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="tnum text-sm font-semibold text-ink">{confidence}%</span>
        {!compact ? <span className="text-xs text-ink-secondary">{meta.label}</span> : null}
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full"
        role="meter"
        aria-valuenow={confidence}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Evidence-based confidence: ${confidence} percent, ${meta.label}`}
        style={{ background: 'var(--gridline)' }}
      >
        <div className="h-full rounded-full" style={{ width: `${confidence}%`, background: meta.color }} />
      </div>
      {compact ? <div className="mt-1 text-[11px] text-ink-muted">{meta.label}</div> : null}
    </div>
  )
}
