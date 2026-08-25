/** Presentation helpers. No business arithmetic happens in the browser. */

export function formatValue(value, unit = 'count', { compact = false } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (unit === 'percent') return `${value.toFixed(1)}%`
  if (unit === 'ratio') return value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  if (unit === 'currency') {
    if (compact && Math.abs(value) >= 1000) return compactNumber(value)
    return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
  }
  if (compact && Math.abs(value) >= 10000) return compactNumber(value)
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 })
}

export function compactNumber(value) {
  const abs = Math.abs(value)
  const sign = value < 0 ? '-' : ''
  if (abs >= 1e9) return `${sign}${(abs / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}K`
  return `${sign}${abs.toFixed(0)}`
}

export function formatDelta(value, { digits = 1 } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value > 0 ? '+' : ''}${value.toFixed(digits)}%`
}

/** Is a move good or bad for THIS metric? (revenue up = good, stockouts up = bad) */
export function deltaTone(change, higherIsBetter = true) {
  if (change === null || change === undefined || Math.abs(change) < 0.05) return 'flat'
  const good = change > 0 ? higherIsBetter : !higherIsBetter
  return good ? 'good' : 'bad'
}

export const CONFIDENCE_BANDS = {
  strong: { label: 'Strong evidence', color: 'var(--status-good)' },
  moderate: { label: 'Moderate evidence', color: 'var(--status-warning)' },
  weak: { label: 'Weak evidence', color: 'var(--status-serious)' },
  insufficient: { label: 'Insufficient evidence', color: 'var(--status-critical)' },
}

export const VERDICT_COPY = {
  newly_launched: {
    label: 'Newly launched',
    tone: 'warning',
    blurb: 'Current KPI value is available, but there is no prior history for trend analysis.',
  },
  sparse_history: {
    label: 'Sparse history',
    tone: 'warning',
    blurb: 'Current KPI value is available, but historical trend analysis is limited.',
  },
  meaningful_signal: {
    label: 'Meaningful signal',
    tone: 'critical',
    blurb: 'This change is outside the range this measure normally moves in.',
  },
  within_normal_variation: {
    label: 'Within normal variation',
    tone: 'good',
    blurb: 'This is the sort of movement this measure shows routinely. No investigation is warranted on this number alone.',
  },
  statistically_unusual_but_immaterial: {
    label: 'Unusual but immaterial',
    tone: 'warning',
    blurb: 'Statistically unusual, but too small to matter commercially.',
  },
}

export const STAGES = [
  { key: 'observe', title: 'Observe', question: 'What actually changed?' },
  { key: 'investigate', title: 'Investigate', question: 'What could explain it?' },
  { key: 'contest', title: 'Contest', question: 'What would disprove it?' },
  { key: 'act', title: 'Act', question: 'What should we do?' },
]

export function quarterLabel(year, quarter) {
  return quarter ? `Q${quarter} ${year}` : `FY ${year}`
}

export function titleCase(text = '') {
  return text.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
