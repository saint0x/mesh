import { useMemo } from 'react'
import {
  BulletList,
  ChartTooltipFoot,
  ChartTooltipHeader,
  ChartTooltipRow,
  Heatmap,
  LifecycleArea,
  Sparkline,
  chartColors,
  formatChartNumber,
} from '../charts'
import { EmptyState, Stat, StatRow } from '../primitives'
import { formatDecimal, formatInteger } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

interface DeviceCreditTotals {
  earned: number
  burned: number
  history: number[]
}

export function CreditsPage({ controller }: DashboardPageProps) {
  const jobs = controller.networkJobs
  const ledger = controller.networkLedger

  const totals = useMemo(() => {
    let reserved = 0
    let settled = 0
    let released = 0
    for (const job of jobs) {
      reserved += job.reservedCredits
      settled += job.settledCredits
      released += job.releasedCredits
    }
    let earned = 0
    let burned = 0
    for (const event of ledger) {
      const amount = event.creditsAmount ?? 0
      if (amount > 0) earned += amount
      else if (amount < 0) burned += Math.abs(amount)
    }
    return { reserved, settled, released, earned, burned, net: earned - burned }
  }, [jobs, ledger])

  // Build a 24-bucket lifecycle area: cumulative reserved/settled/released across the
  // ledger events ordered by createdAt.
  const lifecycle = useMemo(() => {
    const buckets = 24
    const labels: string[] = []
    const reserved: number[] = []
    const settled: number[] = []
    const released: number[] = []
    if (ledger.length === 0) {
      return { labels, reserved, settled, released }
    }
    const sorted = [...ledger].sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1))
    const first = sorted[0]
    const last = sorted[sorted.length - 1]
    if (!first || !last) return { labels, reserved, settled, released }
    const start = Date.parse(first.createdAt)
    const end = Date.parse(last.createdAt)
    const span = Math.max(1, end - start)
    const stepMs = span / buckets

    let r = 0
    let s = 0
    let rel = 0
    let cursor = 0
    for (let i = 0; i < buckets; i++) {
      const bucketEnd = start + stepMs * (i + 1)
      while (cursor < sorted.length) {
        const event = sorted[cursor]
        if (!event) break
        const eventTime = Date.parse(event.createdAt)
        if (eventTime > bucketEnd) break
        const amount = event.creditsAmount ?? 0
        const type = event.eventType
        if (type.includes('reserve')) r += Math.abs(amount)
        else if (type.includes('release')) rel += Math.abs(amount)
        else if (amount > 0) s += amount
        else s += Math.abs(amount)
        cursor += 1
      }
      labels.push(new Date(start + stepMs * i).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
      reserved.push(r)
      settled.push(s)
      released.push(rel)
    }
    return { labels, reserved, settled, released }
  }, [ledger])

  const byDevice = useMemo(() => {
    const map = new Map<string, DeviceCreditTotals>()
    const sorted = [...ledger].sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1))
    for (const event of sorted) {
      const id = event.deviceId ?? 'unattributed'
      const current = map.get(id) ?? { earned: 0, burned: 0, history: [] }
      const amount = event.creditsAmount ?? 0
      if (amount > 0) current.earned += amount
      else if (amount < 0) current.burned += Math.abs(amount)
      current.history.push(current.earned - current.burned)
      map.set(id, current)
    }
    return Array.from(map.entries())
      .map(([id, totals]) => ({ id, ...totals }))
      .sort((a, b) => Math.abs(b.earned - b.burned) - Math.abs(a.earned - a.burned))
  }, [ledger])

  const bulletRows = useMemo(
    () =>
      jobs
        .filter((job) => job.reservedCredits > 0)
        .slice(0, 12)
        .map((job) => {
          const outstanding = Math.max(
            0,
            job.reservedCredits - job.settledCredits - job.releasedCredits,
          )
          return {
            id: job.id,
            label: job.id.slice(0, 12),
            reservedCap: job.reservedCredits,
            settled: job.settledCredits,
            released: job.releasedCredits,
            outstanding,
          }
        }),
    [jobs],
  )

  // Activity heatmap: 7 days × 24 hours, derived from ledger event timestamps.
  const heatmap = useMemo(() => {
    const grid: number[][] = Array.from({ length: 7 }, () => Array.from({ length: 24 }, () => 0))
    for (const event of ledger) {
      const date = new Date(event.createdAt)
      if (Number.isNaN(date.valueOf())) continue
      const day = (date.getDay() + 6) % 7 // Mon=0
      const hour = date.getHours()
      const row = grid[day]
      if (!row) continue
      row[hour] = (row[hour] ?? 0) + 1
    }
    return grid
  }, [ledger])

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Reserved"
          value={formatChartNumber(totals.reserved)}
          accent="cool"
          caption="Held at job admission across the network."
        />
        <Stat
          label="Settled"
          value={formatChartNumber(totals.settled)}
          accent="accent"
          caption="Attributed after assignments completed."
        />
        <Stat
          label="Released"
          value={formatChartNumber(totals.released)}
          accent="violet"
          caption="Returned to submitters from unused reservation."
        />
        <Stat
          label="Net earned"
          value={formatDecimal(totals.net, 2)}
          accent={totals.net >= 0 ? 'accent' : 'danger'}
          caption={`${formatDecimal(totals.earned, 2)} earned · ${formatDecimal(totals.burned, 2)} burned`}
          trend={totals.net >= 0 ? 'up' : 'down'}
          delta={totals.net >= 0 ? '+ ledger flow' : '− ledger flow'}
        />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Reservation lifecycle</div>
            <h3>Reserved → settled → released over time</h3>
            <p>Cumulative credit flow derived from the local ledger event stream.</p>
          </div>
        </div>
        <div className="ms-chart-card">
          {lifecycle.labels.length > 0 ? (
            <LifecycleArea
              labels={lifecycle.labels}
              series={[
                { key: 'settled', label: 'settled', color: chartColors.settled, values: lifecycle.settled },
                { key: 'released', label: 'released', color: chartColors.released, values: lifecycle.released },
                { key: 'reserved', label: 'reserved', color: chartColors.reserved, values: lifecycle.reserved },
              ]}
              renderTooltip={(idx) => (
                <>
                  <ChartTooltipHeader>{lifecycle.labels[idx]}</ChartTooltipHeader>
                  <ChartTooltipRow
                    color={chartColors.settled}
                    label="settled"
                    value={formatDecimal(lifecycle.settled[idx] ?? 0, 2)}
                  />
                  <ChartTooltipRow
                    color={chartColors.released}
                    label="released"
                    value={formatDecimal(lifecycle.released[idx] ?? 0, 2)}
                  />
                  <ChartTooltipRow
                    color={chartColors.reserved}
                    label="reserved"
                    value={formatDecimal(lifecycle.reserved[idx] ?? 0, 2)}
                  />
                  <ChartTooltipFoot>cumulative across the bucket</ChartTooltipFoot>
                </>
              )}
            />
          ) : (
            <EmptyState
              title="No ledger activity yet"
              hint="Submit jobs and the lifecycle chart will populate from the local ledger."
            />
          )}
        </div>
      </section>

      <div className="ms-credit-grid">
        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">By job</div>
              <h3>Per-job reservation bullets</h3>
              <p>Top 12 jobs ranked by reserved credit volume.</p>
            </div>
          </div>
          {bulletRows.length > 0 ? (
            <div className="ms-chart-card">
              <BulletList
                rows={bulletRows}
                onSelect={(row) => controller.setSelectedJobId(row.id)}
                {...(controller.selectedJob ? { selectedId: controller.selectedJob.id } : {})}
                renderTooltip={(row) => (
                  <>
                    <ChartTooltipHeader>{row.label}</ChartTooltipHeader>
                    <ChartTooltipRow color={chartColors.settled} label="settled" value={formatDecimal(row.settled, 2)} />
                    <ChartTooltipRow color={chartColors.released} label="released" value={formatDecimal(row.released, 2)} />
                    <ChartTooltipRow
                      color={chartColors.outstanding}
                      label="outstanding"
                      value={formatDecimal(row.outstanding, 2)}
                    />
                    <ChartTooltipRow
                      color={chartColors.reserved}
                      label="reserved cap"
                      value={formatDecimal(row.reservedCap, 2)}
                      outlined
                    />
                  </>
                )}
              />
            </div>
          ) : (
            <EmptyState title="No jobs with reserved credits yet" />
          )}
        </section>

        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">By device</div>
              <h3>Earned vs burned</h3>
              <p>Credit flow per device with a 30-event sparkline.</p>
            </div>
          </div>
          {byDevice.length > 0 ? (
            <div className="ms-assignments">
              <div className="ms-assignments-head">
                <span>Device</span>
                <span className="num">Earned</span>
                <span className="num">Burned</span>
                <span className="num">Net</span>
                <span>Trend</span>
              </div>
              {byDevice.map((row) => (
                <div key={row.id} className="ms-assignments-row">
                  <span className="mono" style={{ fontSize: 11 }}>
                    {row.id.slice(0, 14)}
                  </span>
                  <span className="num" style={{ color: chartColors.settled }}>
                    +{formatDecimal(row.earned, 2)}
                  </span>
                  <span className="num" style={{ color: chartColors.danger }}>
                    −{formatDecimal(row.burned, 2)}
                  </span>
                  <span
                    className="num"
                    style={{ color: row.earned - row.burned >= 0 ? chartColors.settled : chartColors.danger }}
                  >
                    {formatDecimal(row.earned - row.burned, 2)}
                  </span>
                  <span>
                    <Sparkline
                      data={row.history.slice(-30)}
                      width={86}
                      height={20}
                      color={row.earned - row.burned >= 0 ? chartColors.settled : chartColors.danger}
                      fillColor={
                        row.earned - row.burned >= 0
                          ? 'rgba(102,240,192,0.18)'
                          : 'rgba(255,107,107,0.18)'
                      }
                      renderTooltip={(idx, value) => (
                        <>
                          <ChartTooltipHeader>{row.id.slice(0, 14)}</ChartTooltipHeader>
                          <ChartTooltipRow
                            color={chartColors.settled}
                            label={`event ${idx + 1}`}
                            value={formatDecimal(value, 2)}
                          />
                        </>
                      )}
                    />
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState title="No credit attribution yet" />
          )}
        </section>
      </div>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Activity</div>
            <h3>Ledger events · 7 days × 24 hours</h3>
            <p>When credit-bearing events land in the local ledger.</p>
          </div>
        </div>
        <div className="ms-chart-card">
          <Heatmap
            rows={heatmap}
            rowLabels={['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']}
            columnLabels={Array.from({ length: 24 }, (_, i) => `${i}h`)}
            renderTooltip={(row, column, value) => (
              <>
                <ChartTooltipHeader>
                  {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][row]} · {String(column).padStart(2, '0')}:00
                </ChartTooltipHeader>
                <ChartTooltipRow color={chartColors.settled} label="events" value={formatInteger(value)} />
              </>
            )}
          />
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Audit trail</div>
            <h3>Credit-bearing ledger events</h3>
            <p>{formatInteger(ledger.filter((event) => typeof event.creditsAmount === 'number').length)} events on this network.</p>
          </div>
        </div>
        <div className="ms-assignments">
          <div className="ms-assignments-head">
            <span>Event</span>
            <span>Job</span>
            <span>Device</span>
            <span className="num">Credits</span>
            <span>Metadata</span>
          </div>
          {ledger
            .filter((event) => typeof event.creditsAmount === 'number')
            .map((event) => (
              <div key={event.id} className="ms-assignments-row">
                <span style={{ color: chartColors.textStrong }}>{event.eventType.replaceAll('_', ' ')}</span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {event.jobId ? event.jobId.slice(0, 12) : '—'}
                </span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {event.deviceId ? event.deviceId.slice(0, 12) : '—'}
                </span>
                <span
                  className="num"
                  style={{
                    color: (event.creditsAmount ?? 0) >= 0 ? chartColors.settled : chartColors.danger,
                  }}
                >
                  {(event.creditsAmount ?? 0) >= 0 ? '+' : ''}
                  {formatDecimal(event.creditsAmount ?? 0, 2)}
                </span>
                <span className="ms-credit-metadata">{event.metadata ? JSON.stringify(event.metadata) : '—'}</span>
              </div>
            ))}
        </div>
      </section>
    </div>
  )
}
