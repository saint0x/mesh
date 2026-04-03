import { formatSchedulingPolicy, type NetworkRecord } from '../../domain/dashboard'
import type { DashboardPageProps } from '../lib/pageProps'

export function NetworksPage({ controller }: DashboardPageProps) {
  const selectedNetwork = controller.selectedNetwork
  if (!selectedNetwork) return null

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Network registry</div>
            <h3>Control-plane network inventory</h3>
          </div>
        </div>
        <div className="dashboard-list">
          {controller.state.networks.map((network: NetworkRecord) => (
            <button
              key={network.id}
              className={network.id === selectedNetwork.id ? 'dashboard-list-item active' : 'dashboard-list-item'}
              onClick={() => controller.setSelectedNetworkId(network.id)}
            >
              <strong>{network.name}</strong>
              <span>{network.preferredPath} / {formatSchedulingPolicy(network.schedulingPolicy)}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Scheduling and connectivity</div>
            <h3>{selectedNetwork.name}</h3>
          </div>
        </div>
        <div className="dashboard-detail-grid">
          <article><span>Owner</span><strong>{selectedNetwork.owner}</strong></article>
          <article><span>Created</span><strong>{selectedNetwork.createdAt}</strong></article>
          <article><span>Preferred path</span><strong>{selectedNetwork.preferredPath}</strong></article>
          <article><span>Relay attachments</span><strong>{selectedNetwork.attachments.length}</strong></article>
        </div>
        <div className="snapshot-grid">
          <article>
            <span>Submitter soft cap</span>
            <strong>{selectedNetwork.schedulingPolicy.submitterActiveJobSoftCap}</strong>
            <small>Maximum active jobs per submitter before fairness backpressure.</small>
          </article>
          <article>
            <span>Model soft cap divisor</span>
            <strong>{selectedNetwork.schedulingPolicy.modelActiveJobSoftCapDivisor}</strong>
            <small>Controls how quickly one model can monopolize active slots.</small>
          </article>
          <article>
            <span>Capacity soft cap divisor</span>
            <strong>{selectedNetwork.schedulingPolicy.capacityUnitSoftCapDivisor}</strong>
            <small>Weights dispatch fairness against assigned capacity units.</small>
          </article>
          <article>
            <span>Tier capacity units</span>
            <strong>
              {selectedNetwork.schedulingPolicy.tierCapacityUnits.tier0}/
              {selectedNetwork.schedulingPolicy.tierCapacityUnits.tier1}/
              {selectedNetwork.schedulingPolicy.tierCapacityUnits.tier2}/
              {selectedNetwork.schedulingPolicy.tierCapacityUnits.tier3}/
              {selectedNetwork.schedulingPolicy.tierCapacityUnits.tier4}
            </strong>
            <small>Tier0 through Tier4 assigned capacity units.</small>
          </article>
        </div>

        <div className="dashboard-data-table">
          <div className="dashboard-data-head">
            <span>Attachment kind</span>
            <span>Endpoint</span>
            <span>Priority</span>
          </div>
          {selectedNetwork.attachments.length > 0 ? selectedNetwork.attachments.map((attachment) => (
            <div key={`${attachment.kind}-${attachment.endpoint}`} className="dashboard-data-row">
              <span className="row-primary">{attachment.kind}</span>
              <span>{attachment.endpoint}</span>
              <span>{attachment.priority}</span>
            </div>
          )) : (
            <div className="dashboard-empty">This network currently has no relay attachments configured.</div>
          )}
        </div>
      </section>
    </div>
  )
}
