/**
 * Thin REST client.
 *
 * A single place attaches the bearer token, so every request the app makes is
 * authenticated the same way whether the token came from Firebase or from the
 * backend's demo login.
 */
const BASE = import.meta.env.VITE_API_BASE_URL || ''

let tokenProvider = async () => null

export function setTokenProvider(fn) {
  tokenProvider = fn
}

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.status = status
    this.payload = payload
  }
}

async function request(path, { method = 'GET', body, form, signal } = {}) {
  const headers = {}
  const token = await tokenProvider()
  if (token) headers.Authorization = `Bearer ${token}`
  let payload
  if (form) {
    payload = form
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }

  let response
  try {
    response = await fetch(`${BASE}${path}`, { method, headers, body: payload, signal })
  } catch (err) {
    if (err.name === 'AbortError') throw err
    throw new ApiError(
      'Could not reach the API. Is the backend running on ' + (BASE || 'this origin') + '?',
      0,
      null,
    )
  }

  const text = await response.text()
  const data = text ? safeJson(text) : null
  if (!response.ok) {
    const detail = data?.detail || response.statusText || 'Request failed'
    throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), response.status, data)
  }
  return data
}

function safeJson(text) {
  try {
    return JSON.parse(text)
  } catch {
    return { raw: text }
  }
}

export const api = {
  // system + auth
  systemStatus: () => request('/api/system/status'),
  authConfig: () => request('/api/auth/config'),
  demoLogin: (payload) => request('/api/auth/demo-login', { method: 'POST', body: payload }),
  register: (payload) => request('/api/auth/register', { method: 'POST', body: payload }),
  me: () => request('/api/auth/me'),
  setRole: (role) => request('/api/auth/role', { method: 'PATCH', body: { role } }),

  // structured data
  listDatasets: () => request('/api/datasets'),
  activeDataset: () => request('/api/datasets/active'),
  uploadDataset: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/api/datasets', { method: 'POST', form })
  },
  loadSampleDataset: () => request('/api/datasets/load-sample', { method: 'POST' }),
  activateDataset: (id) => request(`/api/datasets/${id}/activate`, { method: 'POST' }),
  deleteDataset: (id) => request(`/api/datasets/${id}`, { method: 'DELETE' }),
  templateUrl: () => `${BASE}/api/datasets/template`,

  // unstructured data
  listDocuments: () => request('/api/documents'),
  uploadDocuments: (files) => {
    const form = new FormData()
    Array.from(files).forEach((f) => form.append('files', f))
    return request('/api/documents', { method: 'POST', form })
  },
  loadSampleDocuments: () => request('/api/documents/load-samples', { method: 'POST' }),
  deleteDocument: (id) => request(`/api/documents/${id}`, { method: 'DELETE' }),
  searchDocuments: (query, topK = 5) =>
    request('/api/documents/search', { method: 'POST', body: { query, top_k: topK } }),

  // analysis
  dashboard: ({ year, quarter, kpi, comparison } = {}) => {
    const qs = new URLSearchParams()
    if (year) qs.set('year', year)
    if (quarter) qs.set('quarter', quarter)
    if (kpi) qs.set('kpi', kpi)
    if (comparison) qs.set('comparison', comparison)
    return request(`/api/dashboard?${qs.toString()}`)
  },
  runInvestigation: (payload, signal) =>
    request('/api/investigations/run', { method: 'POST', body: payload, signal }),
  listInvestigations: () => request('/api/investigations'),
  getInvestigation: (id) => request(`/api/investigations/${id}`),
  deleteInvestigation: (id) => request(`/api/investigations/${id}`, { method: 'DELETE' }),
}
