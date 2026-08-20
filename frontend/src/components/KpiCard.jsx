import { ArrowDownRight, ArrowRight, ArrowUpRight } from 'lucide-react'
import { deltaTone, formatDelta, formatValue } from '../lib/format'

/**
 * A stat tile, not a chart: one number, its move, and whether that move is good
 * for THIS metric (a rising stockout rate is not good news).
 */
export default function KpiCard({ item, active = false, onSelect }) {
  const tone = deltaTone(item.change_pct, item.higher_is_better)
  const toneColor = {
    good: 'var(--delta-good)',
    bad: 'var(--delta-bad)',
    flat: 'var(--text-muted)',
  }[tone]
  const Icon = tone === 'flat' ? ArrowRight : item.change_pct > 0 ? ArrowUpRight : ArrowDownRight
  const Wrapper = onSelect ? 'button' : 'div'

  return (
    <Wrapper
      type={onSelect ? 'button' : undefined}
      onClick={onSelect ? () => onSelect(item.key) : undefined}
      aria-pressed={onSelect ? active : undefined}
      className={`card card-pad text-left transition-shadow ${onSelect ? 'hover:shadow-md' : ''}`}
      style={active ? { outline: '2px solid var(--series-1)', outlineOffset: '-1px' } : undefined}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-medium text-ink-secondary">{item.label}</span>
        {item.is_primary ? (
          <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--series-1)' }}>
            Primary
          </span>
        ) : null}
      </div>
      <div className="tnum mt-1.5 text-xl font-semibold text-ink sm:text-2xl">
        {formatValue(item.current, item.unit, { compact: true })}
      </div>
      <div className="mt-1 flex items-center gap-1 text-xs" style={{ color: toneColor }}>
        <Icon className="h-3.5 w-3.5" aria-hidden />
        <span className="tnum font-medium">{formatDelta(item.change_pct)}</span>
        <span className="text-ink-muted">vs baseline</span>
      </div>
    </Wrapper>
  )
}
