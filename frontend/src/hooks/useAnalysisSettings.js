import { useCallback, useEffect, useState } from 'react'

const KEY = 'bi-analysis-settings'
const DEFAULTS = { year: null, quarter: null, kpi: 'revenue', comparison: 'previous_period' }

/**
 * The timeframe/KPI selection, shared between the dashboard and the
 * investigation workspace and remembered between visits.
 */
export default function useAnalysisSettings() {
  const [settings, setSettings] = useState(() => {
    try {
      return { ...DEFAULTS, ...JSON.parse(window.localStorage.getItem(KEY) || '{}') }
    } catch {
      return DEFAULTS
    }
  })

  useEffect(() => {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(settings))
    } catch {
      /* ignore unavailable storage */
    }
  }, [settings])

  const update = useCallback((patch) => setSettings((s) => ({ ...s, ...patch })), [])
  return [settings, update]
}
