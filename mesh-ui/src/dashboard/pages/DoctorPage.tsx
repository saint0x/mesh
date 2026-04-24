import { useMemo } from 'react'
import { EmptyState, MutationButton, Stat, StatRow, StatusBadge } from '../primitives'
import { chartColors } from '../charts'
import { formatInteger } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function DoctorPage({ controller }: DashboardPageProps) {
  const report = controller.doctorReport
  const doctorState = controller.mutationState['doctor']?.state ?? 'idle'

  const counts = useMemo(() => {
    if (!report) return { ok: 0, warn: 0, fail: 0 }
    let ok = 0
    let warn = 0
    let fail = 0
    for (const check of report.checks) {
      if (check.status === 'ok') ok += 1
      else if (check.status === 'warn') warn += 1
      else fail += 1
    }
    return { ok, warn, fail }
  }, [report])

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Overall"
          value={
            report ? (
              <StatusBadge
                status={report.overall === 'ok' ? 'ok' : report.overall === 'warn' ? 'warn' : 'fail'}
                size="md"
              >
                {report.overall}
              </StatusBadge>
            ) : (
              '—'
            )
          }
          accent={
            report?.overall === 'ok'
              ? 'accent'
              : report?.overall === 'warn'
                ? 'warm'
                : report?.overall === 'fail'
                  ? 'danger'
                  : 'neutral'
          }
          caption={report ? `Last run ${report.generatedAt}` : 'Click "Run doctor" to start.'}
        />
        <Stat label="OK" value={formatInteger(counts.ok)} accent="accent" />
        <Stat label="Warnings" value={formatInteger(counts.warn)} accent="warm" />
        <Stat label="Failures" value={formatInteger(counts.fail)} accent={counts.fail > 0 ? 'danger' : 'neutral'} />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Doctor</div>
            <h3>Local health checks</h3>
            <p>
              Runs the same checks as <span className="mono">mesh doctor</span> on the CLI: device config,
              certificate posture, control-plane reachability, and runtime sanity.
            </p>
          </div>
          <div className="ms-section-head-actions">
            <MutationButton
              variant="primary"
              size="sm"
              state={doctorState}
              onClick={() => void controller.refreshDoctor()}
              pendingLabel="Running…"
            >
              Run doctor
            </MutationButton>
          </div>
        </div>

        {!report ? (
          <EmptyState
            title="No doctor report yet"
            hint="Run the doctor to fetch the latest local setup and reachability snapshot."
          />
        ) : (
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Check</span>
              <span>Status</span>
              <span>Detail</span>
              <span>Hint</span>
              <span className="num">Took</span>
            </div>
            {report.checks.map((check) => (
              <div key={check.id} className="ms-assignments-row">
                <span>
                  <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                    {check.label}
                  </strong>
                  <small className="mono" style={{ color: chartColors.textMuted, fontSize: 10 }}>
                    {check.id}
                  </small>
                </span>
                <span>
                  <StatusBadge status={check.status} />
                </span>
                <span style={{ fontSize: 11.5 }}>{check.detail}</span>
                <span style={{ fontSize: 11, color: chartColors.text }}>{check.hint ?? '—'}</span>
                <span className="num" style={{ fontSize: 11 }}>
                  {check.durationMs} ms
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
