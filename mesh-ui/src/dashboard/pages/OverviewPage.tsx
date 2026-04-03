import { startTransition } from 'react'
import type { DeviceRecord, JobRecord, LedgerRecord, OverviewTab } from '../../domain/dashboard'
import { formatBytes, formatDecimal, formatInteger, formatLatency } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

const overviewTabs: Array<{ id: OverviewTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'operations', label: 'Operations' },
  { id: 'runtime', label: 'Runtime' },
]

export function OverviewPage({ controller, onNavigateSection }: DashboardPageProps) {
  const selectedNetwork = controller.selectedNetwork
  if (!selectedNetwork) return null

  const topology = controller.selectedTopology
  const localDevice = controller.networkDevices.find((device) => device.localDevice)
  const recentFailures = controller.networkJobs.filter((job) => job.error).slice(0, 3)
  const earnedCredits = controller.networkLedger
    .filter((event) => (event.creditsAmount ?? 0) > 0)
    .reduce((total, event) => total + (event.creditsAmount ?? 0), 0)
  const burnedCredits = Math.abs(
    controller.networkLedger
      .filter((event) => (event.creditsAmount ?? 0) < 0)
      .reduce((total, event) => total + (event.creditsAmount ?? 0), 0),
  )

  return (
    <div className="dashboard-stack">
      <section className="dashboard-subnav">
        {overviewTabs.map((tab) => (
          <button
            key={tab.id}
            className={controller.overviewTab === tab.id ? 'dashboard-chip active' : 'dashboard-chip'}
            onClick={() => controller.setOverviewTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </section>

      <section className="dashboard-metric-row">
        <article className="panel dashboard-metric-card">
          <div>
            <span>Healthy devices</span>
            <strong>{controller.summary.healthyDevices}</strong>
          </div>
          <small>{controller.networkDevices.length} devices registered in this mesh.</small>
        </article>
        <article className="panel dashboard-metric-card">
          <div>
            <span>Ready artifacts</span>
            <strong>{controller.summary.activeModels}</strong>
          </div>
          <small>{controller.networkModels.length} models mapped to the selected network.</small>
        </article>
        <article className="panel dashboard-metric-card">
          <div>
            <span>Running jobs</span>
            <strong>{controller.summary.runningJobs}</strong>
          </div>
          <small>{controller.networkJobs.length} total jobs in the current filtered view.</small>
        </article>
        <article className="panel dashboard-metric-card">
          <div>
            <span>Net credits</span>
            <strong>{formatDecimal(controller.summary.netCredits, 2)}</strong>
          </div>
          <small>Derived from the authoritative mesh ledger.</small>
        </article>
      </section>

      <div className="dashboard-overview-grid">
        <section className="panel dashboard-panel dashboard-table-panel">
          <div className="dashboard-panel-head">
            <div>
              <h3>Mesh posture</h3>
              <p className="dashboard-panel-copy">Real network scheduling, topology, and accounting state.</p>
            </div>
          </div>
          <div className="snapshot-grid">
            <article>
              <span>Preferred path</span>
              <strong>{selectedNetwork.preferredPath}</strong>
              <small>{selectedNetwork.attachments.length} configured relay attachment(s).</small>
            </article>
            <article>
              <span>Ring status</span>
              <strong>{topology?.ringStable ? 'stable' : 'pending'}</strong>
              <small>{topology?.workers.length ?? 0} workers present in topology data.</small>
            </article>
            <article>
              <span>Credits earned</span>
              <strong>{formatDecimal(earnedCredits, 2)}</strong>
              <small>{formatDecimal(burnedCredits, 2)} credits burned on this network.</small>
            </article>
            <article>
              <span>Local device</span>
              <strong>{localDevice?.name ?? 'not present in selected network'}</strong>
              <small>{controller.state.settings.controlPlaneUrl ?? 'No control-plane URL configured'}</small>
            </article>
          </div>
        </section>

        <aside className="dashboard-side-stack">
          <section className="panel dashboard-panel">
            <div className="dashboard-panel-head">
              <div>
                <h3>Recent failures</h3>
                <p className="dashboard-panel-copy">Latest job errors surfaced from the control-plane database.</p>
              </div>
            </div>
            <div className="dashboard-event-list">
              {recentFailures.length > 0 ? recentFailures.map((job) => (
                <button
                  key={job.id}
                  className="dashboard-event-item"
                  onClick={() => {
                    controller.setSelectedJobId(job.id)
                    startTransition(() => onNavigateSection('jobs'))
                  }}
                >
                  <strong>{job.id}</strong>
                  <span>{job.error}</span>
                  <small>{job.completedAt ?? job.createdAt}</small>
                </button>
              )) : (
                <div className="dashboard-empty">No failed jobs in the current network.</div>
              )}
            </div>
          </section>

          <section className="panel dashboard-panel">
            <div className="dashboard-panel-head">
              <div>
                <h3>Local runtime</h3>
                <p className="dashboard-panel-copy">What this machine is currently contributing.</p>
              </div>
            </div>
            {localDevice ? (
              <div className="snapshot-grid">
                <article>
                  <span>Provider</span>
                  <strong>{localDevice.capabilities.defaultExecutionProvider}</strong>
                  <small>{localDevice.capabilities.executionProviders.filter((provider) => provider.available).length} available providers.</small>
                </article>
                <article>
                  <span>Contributed memory</span>
                  <strong>{localDevice.contributedMemoryBytes ? formatBytes(localDevice.contributedMemoryBytes) : 'n/a'}</strong>
                  <small>{localDevice.capabilities.tier} capability tier.</small>
                </article>
                <article>
                  <span>Ring position</span>
                  <strong>{localDevice.ringPosition ?? 'n/a'}</strong>
                  <small>{localDevice.shardModelId ?? 'No active shard model'}</small>
                </article>
                <article>
                  <span>Certificate</span>
                  <strong>{localDevice.certificateStatus}</strong>
                  <small>{localDevice.identityStatus} identity status.</small>
                </article>
              </div>
            ) : (
              <div className="dashboard-empty">The local device is not part of this selected network.</div>
            )}
          </section>
        </aside>
      </div>

      <div className="dashboard-overview-bottom">
        <section className="panel dashboard-panel dashboard-table-panel">
          <div className="dashboard-panel-head">
            <div>
              <h3>Recent jobs</h3>
              <p className="dashboard-panel-copy">Latest execution records with authoritative timing and assignment counts.</p>
            </div>
            <button className="dashboard-outline-button" onClick={() => startTransition(() => onNavigateSection('jobs'))}>
              View all jobs
            </button>
          </div>
          <div className="dashboard-data-table">
            <div className="dashboard-data-head">
              <span>Job</span>
              <span>Model</span>
              <span>Status</span>
              <span>Assignments</span>
              <span>Runtime</span>
            </div>
            {controller.networkJobs.slice(0, 5).map((job: JobRecord) => (
              <button
                key={job.id}
                className="dashboard-data-row"
                onClick={() => {
                  controller.setSelectedJobId(job.id)
                  startTransition(() => onNavigateSection('jobs'))
                }}
              >
                <span className="row-primary">{job.id}</span>
                <span>{job.modelId}</span>
                <span><span className={`status-badge ${job.status}`}>{job.status}</span></span>
                <span>{formatInteger(job.assignments.length)}</span>
                <span>{job.executionTimeMs > 0 ? formatLatency(job.executionTimeMs) : 'n/a'}</span>
              </button>
            ))}
          </div>
        </section>

        <aside className="dashboard-side-stack">
          <section className="panel dashboard-panel">
            <div className="dashboard-panel-head">
              <div>
                <h3>Devices</h3>
                <p className="dashboard-panel-copy">Health, provider, and connectivity posture.</p>
              </div>
            </div>
            <div className="health-list">
              {controller.networkDevices.slice(0, 4).map((device: DeviceRecord) => (
                <button
                  key={device.id}
                  className="health-row"
                  onClick={() => {
                    controller.setSelectedDeviceId(device.id)
                    startTransition(() => onNavigateSection('devices'))
                  }}
                >
                  <div>
                    <strong>{device.name}</strong>
                    <span>{device.capabilities.defaultExecutionProvider} • {device.connectivityState?.activePath ?? device.status}</span>
                  </div>
                  <div className={`health-pill ${device.health}`}>{device.health}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="panel dashboard-panel">
            <div className="dashboard-panel-head">
              <div>
                <h3>Recent ledger events</h3>
                <p className="dashboard-panel-copy">Last append-only network events.</p>
              </div>
            </div>
            <div className="dashboard-event-list">
              {controller.networkLedger.slice(0, 4).map((event: LedgerRecord) => (
                <div key={event.id} className="dashboard-event-item">
                  <strong>{event.eventType.replaceAll('_', ' ')}</strong>
                  <span>{event.detail}</span>
                  <small>{event.createdAt}</small>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  )
}
