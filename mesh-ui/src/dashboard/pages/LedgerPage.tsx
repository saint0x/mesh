import type { LedgerRecord } from '../../domain/dashboard'
import { formatDecimal, formatInteger } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function LedgerPage({ controller }: DashboardPageProps) {
  const creditsEarned = controller.networkLedger
    .filter((event: LedgerRecord) => (event.creditsAmount ?? 0) > 0)
    .reduce((sum: number, event: LedgerRecord) => sum + (event.creditsAmount ?? 0), 0)
  const creditsBurned = Math.abs(
    controller.networkLedger
      .filter((event: LedgerRecord) => (event.creditsAmount ?? 0) < 0)
      .reduce((sum: number, event: LedgerRecord) => sum + (event.creditsAmount ?? 0), 0),
  )
  const jobsStarted = controller.networkLedger.filter((event: LedgerRecord) => event.eventType === 'job_started').length
  const jobsCompleted = controller.networkLedger.filter((event: LedgerRecord) => event.eventType === 'job_completed').length

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Ledger summary</div>
            <h3>Authoritative control-plane accounting</h3>
          </div>
        </div>
        <div className="dashboard-stat-grid">
          <article><span>Total events</span><strong>{formatInteger(controller.networkLedger.length)}</strong></article>
          <article><span>Credits earned</span><strong>{formatDecimal(creditsEarned, 2)}</strong></article>
          <article><span>Credits burned</span><strong>{formatDecimal(creditsBurned, 2)}</strong></article>
          <article><span>Jobs started</span><strong>{formatInteger(jobsStarted)}</strong></article>
          <article><span>Jobs completed</span><strong>{formatInteger(jobsCompleted)}</strong></article>
          <article><span>Net credits</span><strong>{formatDecimal(creditsEarned - creditsBurned, 2)}</strong></article>
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Event feed</div>
            <h3>Append-only network ledger</h3>
          </div>
        </div>
        <div className="dashboard-list">
          {controller.networkLedger.map((event: LedgerRecord) => (
            <div key={event.id} className="dashboard-list-item static">
              <strong>{event.eventType.replaceAll('_', ' ')}</strong>
              <span>{event.createdAt} / {event.detail} / credit delta {formatDecimal(event.creditsAmount ?? 0, 2)}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
