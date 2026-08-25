import { useEffect, useState } from 'react'
import {
  BriefcaseBusiness,
  Cpu,
  Database,
  LineChart,
  Rocket,
  ShieldCheck,
  Stethoscope,
  Users,
  Wrench,
} from 'lucide-react'
import { Badge, Callout, LoadingCard, SectionTitle } from '../components/ui'
import { useAuth } from '../context/AuthContext'
import { api } from '../lib/api'
import { titleCase } from '../lib/format'

const ROLES = [
  {
    key: 'business_leader',
    label: 'Business Leader',
    icon: BriefcaseBusiness,
    blurb: 'What changed, whether it matters, the leading explanation, what to do next.',
  },
  {
    key: 'data_analyst',
    label: 'Data Analyst',
    icon: LineChart,
    blurb: 'Adds the statistical method, driver tables, the evidence ledger, weekly onset charts and the retrieval inspector.',
  },
]

// Icons keyed by persona — purely presentational, so this lives on the
// frontend rather than in the API response.
const PERSONA_ICONS = {
  business_analyst: LineChart,
  business_manager: Users,
  business_leader: BriefcaseBusiness,
  domain_specialist: Stethoscope,
  operational_user: Wrench,
}

export default function SettingsPage() {
  const { profile, changeRole, changePersona } = useAuth()
  const [status, setStatus] = useState(null)
  const [personas, setPersonas] = useState([])
  const [busy, setBusy] = useState(false)
  const [personaBusy, setPersonaBusy] = useState(false)

  useEffect(() => {
    api.systemStatus().then(setStatus).catch(() => setStatus(null))
    api.authConfig().then((c) => setPersonas(c.personas || [])).catch(() => setPersonas([]))
  }, [])

  return (
    <div className="space-y-5">
      <SectionTitle eyebrow="Account" title="Settings" />

      <div className="card card-pad">
        <div className="text-xs font-semibold uppercase tracking-wider text-ink-muted">Profile</div>
        <dl className="mt-2 grid gap-3 sm:grid-cols-3">
          <Item k="Name" v={profile?.display_name} />
          <Item k="Email" v={profile?.email} />
          <Item k="Organisation" v={profile?.organisation || '—'} />
        </dl>
      </div>

      <section>
        <SectionTitle
          title="Your role"
          description="The role decides what the API returns to you, not merely what the interface shows. Analyst-only fields are removed from the response for other roles."
        />
        <div className="grid gap-3 sm:grid-cols-2">
          {ROLES.map(({ key, label, icon: Icon, blurb }) => {
            const selected = profile?.role === key
            return (
              <button
                key={key}
                type="button"
                disabled={busy}
                onClick={async () => {
                  setBusy(true)
                  try {
                    await changeRole(key)
                  } finally {
                    setBusy(false)
                  }
                }}
                className="card card-pad text-left transition-shadow hover:shadow-sm"
                style={selected ? { outline: '2px solid var(--series-1)', outlineOffset: '-1px' } : undefined}
              >
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4" style={{ color: selected ? 'var(--series-1)' : 'var(--text-muted)' }} aria-hidden />
                  <span className="text-sm font-semibold text-ink">{label}</span>
                  {selected ? <Badge tone="good">Current</Badge> : null}
                </div>
                <p className="mt-1 text-sm text-ink-secondary">{blurb}</p>
              </button>
            )
          })}
        </div>
      </section>

      <section>
        <SectionTitle
          title="How investigations are written for you"
          description="Persona is separate from role and changes nothing about what the server sends — it decides how a finding is explained and what it recommends you do. Every persona sees the same evidence, the same ranking and the same confidence; only the framing and the advice differ."
        />
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {personas.map(({ key, label, description, action_horizon: horizon }) => {
            const Icon = PERSONA_ICONS[key] || Rocket
            const selected = profile?.persona === key
            return (
              <button
                key={key}
                type="button"
                disabled={personaBusy}
                onClick={async () => {
                  setPersonaBusy(true)
                  try {
                    await changePersona(key)
                  } finally {
                    setPersonaBusy(false)
                  }
                }}
                className="card card-pad text-left transition-shadow hover:shadow-sm"
                style={selected ? { outline: '2px solid var(--series-1)', outlineOffset: '-1px' } : undefined}
              >
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4" style={{ color: selected ? 'var(--series-1)' : 'var(--text-muted)' }} aria-hidden />
                  <span className="text-sm font-semibold text-ink">{label}</span>
                  {selected ? <Badge tone="good">Current</Badge> : null}
                </div>
                <p className="mt-1 text-sm text-ink-secondary">{description}</p>
                {horizon ? (
                  <p className="mt-1.5 text-xs text-ink-muted">Recommends for: {horizon.replace('_', ' ')}</p>
                ) : null}
              </button>
            )
          })}
        </div>
      </section>

      {profile?.permissions ? (
        <div className="card card-pad">
          <div className="text-xs font-semibold uppercase tracking-wider text-ink-muted">
            What your role can access
          </div>
          <ul className="mt-2 grid gap-1.5 sm:grid-cols-2">
            {Object.entries(profile.permissions).map(([k, v]) => (
              <li key={k} className="flex items-center gap-2 text-sm">
                <span
                  className="inline-block h-1.5 w-1.5 rounded-full"
                  style={{ background: v ? 'var(--status-good)' : 'var(--baseline)' }}
                  aria-hidden
                />
                <span className={v ? 'text-ink' : 'text-ink-muted line-through'}>{titleCase(k)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <section>
        <SectionTitle title="System" description="What this deployment is running." />
        {!status ? (
          <LoadingCard label="Reading system status…" lines={2} />
        ) : (
          <div className="grid gap-3 sm:grid-cols-3">
            <StatusCard
              icon={Database}
              title="Database"
              lines={[
                `Backend: ${status.database.backend}`,
                status.database.reachable ? 'Reachable' : 'Not reachable',
                status.database.atlas_ready ? 'MongoDB Atlas URI configured' : 'Local JSON document store',
              ]}
            />
            <StatusCard
              icon={ShieldCheck}
              title="Authentication"
              lines={[
                `Mode: ${status.auth.mode}`,
                status.auth.firebase_project ? `Firebase project: ${status.auth.firebase_project}` : 'Firebase project not set',
              ]}
            />
            <StatusCard
              icon={Cpu}
              title="Reasoning layer"
              lines={[
                status.llm.enabled ? `Anthropic ${status.llm.model}` : 'Deterministic fallback',
                status.llm.note,
              ]}
            />
          </div>
        )}
      </section>

      <Callout tone="info" title="A note on the numbers">
        The reasoning layer never computes business figures. KPIs, significance, driver contributions, confidence
        scores and monitoring thresholds are all produced by the deterministic analysis layer from your uploaded rows.
      </Callout>
    </div>
  )
}

function Item({ k, v }) {
  return (
    <div>
      <dt className="text-xs text-ink-muted">{k}</dt>
      <dd className="truncate text-sm text-ink">{v || '—'}</dd>
    </div>
  )
}

function StatusCard({ icon: Icon, title, lines }) {
  return (
    <div className="card card-pad">
      <div className="flex items-center gap-2 text-sm font-semibold text-ink">
        <Icon className="h-4 w-4 text-ink-muted" aria-hidden />
        {title}
      </div>
      <ul className="mt-1.5 space-y-0.5 text-xs text-ink-secondary">
        {lines.filter(Boolean).map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>
    </div>
  )
}
