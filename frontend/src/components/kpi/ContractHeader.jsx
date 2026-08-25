import { CheckCircle2, FileSearch, Lock, RefreshCw, ShieldCheck } from 'lucide-react'
import { Badge, SectionTitle, Spinner } from '../ui'

const STATUS_COPY = {
  draft: {
    tone: 'warning',
    label: 'Draft — under review',
    blurb:
      'These definitions are proposed, not authoritative. The dashboard keeps using the previous contract until this one is approved.',
  },
  provisional: {
    tone: 'neutral',
    label: 'Provisional — never reviewed',
    blurb:
      'Generated automatically from the general KPI library so the product works out of the box. Nobody has confirmed these definitions yet.',
  },
  proposed: { tone: 'warning', label: 'Proposed', blurb: 'Awaiting approval.' },
  approved: {
    tone: 'good',
    label: 'Approved — authoritative',
    blurb: 'Every stage of the analysis resolves KPIs through this contract.',
  },
  superseded: { tone: 'neutral', label: 'Superseded', blurb: 'A newer version is live.' },
}

/**
 * The contract's identity and state, and the one action that makes it
 * authoritative. Approval is deliberately blocked — with the reason shown —
 * while any conflict the system refused to resolve is still open.
 */
export default function ContractHeader({
  summary,
  datasetLabel,
  domains = [],
  screenedBy,
  busy = false,
  onDiscover,
  onApprove,
  canManage = false,
}) {
  const status = STATUS_COPY[summary.status] || STATUS_COPY.draft
  const counts = summary.counts_by_status || {}
  const blocked = summary.blocking_conflicts > 0
  const approved = counts.approved || 0

  return (
    <div className="card card-pad">
      <SectionTitle
        eyebrow="KPI contract"
        title={datasetLabel || 'KPI definitions'}
        description={status.blurb}
        right={
          canManage ? (
            <div className="flex flex-wrap gap-2">
              <button type="button" className="btn-ghost" onClick={onDiscover} disabled={busy}>
                <RefreshCw className="h-4 w-4" aria-hidden />
                Re-run discovery
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={onApprove}
                disabled={busy || blocked || approved === 0}
                title={
                  blocked
                    ? 'Resolve every blocking conflict first'
                    : approved === 0
                      ? 'Approve at least one KPI first'
                      : undefined
                }
              >
                <ShieldCheck className="h-4 w-4" aria-hidden />
                Approve contract
              </button>
            </div>
          ) : null
        }
      />

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Badge tone={status.tone} icon={summary.status === 'approved' ? CheckCircle2 : undefined}>
          {status.label}
        </Badge>
        <span className="chip">Version {summary.version}</span>
        <span className="chip">{summary.kpi_count} KPIs</span>
        {Object.entries(counts).map(([key, value]) => (
          <span key={key} className="chip">
            {value} {key.replace(/_/g, ' ')}
          </span>
        ))}
        {domains.map((d) => (
          <span key={d} className="chip">
            {d.replace(/_/g, ' ')}
          </span>
        ))}
        <span className="chip">
          <FileSearch className="mr-1 inline h-3 w-3" aria-hidden />
          {screenedBy === 'deterministic' ? 'Deterministic screening' : `Screened by ${screenedBy}`}
        </span>
      </div>

      {blocked ? (
        <p className="mt-3 flex items-start gap-2 text-xs" style={{ color: 'var(--status-critical)' }}>
          <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
          {summary.blocking_conflicts} conflict{summary.blocking_conflicts === 1 ? '' : 's'} must be
          resolved by a person before this contract can become authoritative. The system will not
          choose for you.
        </p>
      ) : null}

      {busy ? (
        <div className="mt-3">
          <Spinner label="Working…" />
        </div>
      ) : null}
    </div>
  )
}
