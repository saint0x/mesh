import { formatDecimal, formatInteger } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function CreditsPage({ controller }: DashboardPageProps) {
  const creditEvents = controller.networkLedger.filter((event) => typeof event.creditsAmount === 'number')
  const earned = creditEvents
    .filter((event) => (event.creditsAmount ?? 0) > 0)
    .reduce((sum, event) => sum + (event.creditsAmount ?? 0), 0)
  const burned = Math.abs(
    creditEvents
      .filter((event) => (event.creditsAmount ?? 0) < 0)
      .reduce((sum, event) => sum + (event.creditsAmount ?? 0), 0),
  )
  const byDevice = new Map<string, number>()
  for (const event of creditEvents) {
    const key = event.deviceId ?? 'unattributed'
    byDevice.set(key, (byDevice.get(key) ?? 0) + (event.creditsAmount ?? 0))
  }

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Participation accounting</div>
            <h3>Mesh credits from the real ledger</h3>
          </div>
        </div>
        <div className="dashboard-stat-grid">
          <article><span>Credit events</span><strong>{formatInteger(creditEvents.length)}</strong></article>
          <article><span>Credits earned</span><strong>{formatDecimal(earned, 2)}</strong></article>
          <article><span>Credits burned</span><strong>{formatDecimal(burned, 2)}</strong></article>
          <article><span>Net credits</span><strong>{formatDecimal(earned - burned, 2)}</strong></article>
        </div>
        <div className="dashboard-empty">
          This page is intentionally non-financialized now. It reflects actual MeshNet participation accounting and device/job credit movement, not mock market pricing.
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Credits by device</div>
            <h3>Where credits were earned or burned</h3>
          </div>
        </div>
        <div className="dashboard-data-table">
          <div className="dashboard-data-head">
            <span>Device</span>
            <span>Net credits</span>
          </div>
          {Array.from(byDevice.entries()).map(([deviceId, netCredits]) => (
            <div key={deviceId} className="dashboard-data-row">
              <span className="row-primary">{deviceId}</span>
              <span>{formatDecimal(netCredits, 2)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel dashboard-panel dashboard-grid-span-2">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Credit events</div>
            <h3>Detailed mesh participation records</h3>
          </div>
        </div>
        <div className="dashboard-data-table">
          <div className="dashboard-data-head">
            <span>Event</span>
            <span>Job</span>
            <span>Device</span>
            <span>Credits</span>
            <span>Timestamp</span>
          </div>
          {creditEvents.map((event) => (
            <div key={event.id} className="dashboard-data-row">
              <span className="row-primary">{event.eventType}</span>
              <span>{event.jobId ?? 'n/a'}</span>
              <span>{event.deviceId ?? 'n/a'}</span>
              <span>{formatDecimal(event.creditsAmount ?? 0, 2)}</span>
              <span>{event.createdAt}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
