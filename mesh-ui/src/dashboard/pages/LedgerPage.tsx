import { useMemo } from 'react'
import { EmptyState, Stat, StatRow, StatusBadge } from '../primitives'
import { chartColors } from '../charts'
import { formatDecimal, formatInteger } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

function eventVariant(eventType: string): 'ok' | 'warn' | 'fail' | 'info' {
  if (eventType.includes('failed') || eventType.includes('error')) return 'fail'
  if (eventType.includes('cancel') || eventType.includes('release')) return 'warn'
  if (eventType.includes('completed') || eventType.includes('settled') || eventType.includes('earned'))
    return 'ok'
  return 'info'
}

export function LedgerPage({ controller }: DashboardPageProps) {
  const events = controller.networkLedger

  const summary = useMemo(() => {
    let earned = 0
    let burned = 0
    let started = 0
    let completed = 0
    for (const event of events) {
      const amount = event.creditsAmount ?? 0
      if (amount > 0) earned += amount
      else if (amount < 0) burned += Math.abs(amount)
      if (event.eventType === 'job_started') started += 1
      if (event.eventType === 'job_completed') completed += 1
    }
    return { earned, burned, started, completed }
  }, [events])

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Total events"
          value={formatInteger(events.length)}
          accent="cool"
          caption="On the selected network."
        />
        <Stat
          label="Credits earned"
          value={formatDecimal(summary.earned, 2)}
          accent="accent"
          caption={`${formatDecimal(summary.burned, 2)} credits burned`}
        />
        <Stat
          label="Jobs started"
          value={formatInteger(summary.started)}
          accent="warm"
          caption={`${formatInteger(summary.completed)} completed`}
        />
        <Stat
          label="Net credits"
          value={formatDecimal(summary.earned - summary.burned, 2)}
          accent={summary.earned >= summary.burned ? 'accent' : 'danger'}
          trend={summary.earned >= summary.burned ? 'up' : 'down'}
        />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Event feed</div>
            <h3>Append-only network ledger</h3>
            <p>Authoritative event trail from the local control-plane database.</p>
          </div>
        </div>
        {events.length === 0 ? (
          <EmptyState title="No ledger events on this network yet" />
        ) : (
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Type</span>
              <span>Job</span>
              <span>Device</span>
              <span className="num">Credits</span>
              <span>Detail</span>
            </div>
            {events.map((event) => (
              <div key={event.id} className="ms-assignments-row">
                <span>
                  <StatusBadge status={eventVariant(event.eventType)} dot={false}>
                    {event.eventType.replaceAll('_', ' ')}
                  </StatusBadge>
                </span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {event.jobId ? event.jobId.slice(0, 12) : '—'}
                </span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {event.deviceId ? event.deviceId.slice(0, 12) : '—'}
                </span>
                <span
                  className="num"
                  style={{
                    color:
                      typeof event.creditsAmount === 'number' && event.creditsAmount !== 0
                        ? event.creditsAmount > 0
                          ? chartColors.settled
                          : chartColors.danger
                        : chartColors.text,
                  }}
                >
                  {typeof event.creditsAmount === 'number'
                    ? `${event.creditsAmount > 0 ? '+' : ''}${formatDecimal(event.creditsAmount, 2)}`
                    : '—'}
                </span>
                <span style={{ fontSize: 11 }}>
                  <span style={{ display: 'block', color: chartColors.text }}>{event.detail}</span>
                  <small style={{ color: chartColors.textMuted, fontSize: 10 }}>{event.createdAt}</small>
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
