import { useCallback, useEffect, useState } from 'react'

const KEY = 'bi-analysis-settings'
// kpi: null lets the backend pick the active dataset's own default KPI
// (pipeline.default_kpi) instead of assuming every dataset has 'revenue'.
const DEFAULTS = { year: null, quarter: null, kpi: null, comparison: 'previous_period' }

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
