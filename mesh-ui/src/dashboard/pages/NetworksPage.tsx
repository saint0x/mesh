import { EmptyState, Stat, StatRow, StatusBadge } from '../primitives'
import { chartColors } from '../charts'
import { formatSchedulingPolicy } from '../../domain/dashboard'
import type { DashboardPageProps } from '../lib/pageProps'

export function NetworksPage({ controller }: DashboardPageProps) {
  const selectedNetwork = controller.selectedNetwork
  if (!selectedNetwork) {
    return (
      <section className="panel dashboard-panel">
        <EmptyState title="No networks registered" hint="Use mesh device init to register one." />
      </section>
    )
  }

  const policy = selectedNetwork.schedulingPolicy

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat label="Networks" value={controller.state.networks.length} accent="cool" />
        <Stat
          label="Selected"
          value={selectedNetwork.name}
          accent="accent"
          caption={`Owner ${selectedNetwork.owner}`}
        />
        <Stat
          label="Preferred path"
          value={
            <StatusBadge status="info" dot={false}>
              {selectedNetwork.preferredPath}
            </StatusBadge>
          }
          accent="violet"
        />
        <Stat
          label="Relay attachments"
          value={selectedNetwork.attachments.length}
          accent={selectedNetwork.attachments.length > 0 ? 'warm' : 'neutral'}
        />
      </StatRow>

      <div className="ms-credit-grid">
        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Registry</div>
              <h3>Networks on this device</h3>
            </div>
          </div>
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Name</span>
              <span>Owner</span>
              <span>Path</span>
              <span>Policy</span>
              <span className="num">Relays</span>
            </div>
            {controller.state.networks.map((network) => (
              <button
                key={network.id}
                type="button"
                className="ms-assignments-row"
                style={{
                  background:
                    network.id === selectedNetwork.id
                      ? 'rgba(102,240,192,0.07)'
                      : 'transparent',
                  border: 'none',
                  textAlign: 'left',
                  width: '100%',
                  cursor: 'pointer',
                }}
                onClick={() => controller.setSelectedNetworkId(network.id)}
              >
                <span>
                  <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                    {network.name}
                  </strong>
                  <small className="mono" style={{ color: chartColors.textMuted, fontSize: 10 }}>
                    {network.id.slice(0, 18)}
                  </small>
                </span>
                <span>{network.owner}</span>
                <span>{network.preferredPath}</span>
                <span style={{ fontSize: 11 }}>{formatSchedulingPolicy(network.schedulingPolicy)}</span>
                <span className="num">{network.attachments.length}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Scheduling policy</div>
              <h3>{selectedNetwork.name}</h3>
            </div>
          </div>
          <div className="ms-drawer-lifecycle">
            <div className="cell">
              <span>Submitter cap</span>
              <strong>{policy.submitterActiveJobSoftCap}</strong>
            </div>
            <div className="cell">
              <span>Model divisor</span>
              <strong>{policy.modelActiveJobSoftCapDivisor}</strong>
            </div>
            <div className="cell">
              <span>Capacity divisor</span>
              <strong>{policy.capacityUnitSoftCapDivisor}</strong>
            </div>
            <div className="cell">
              <span>Created</span>
              <strong style={{ fontSize: 12 }}>{selectedNetwork.createdAt}</strong>
            </div>
          </div>
          <div className="ms-section-head" style={{ marginTop: 18 }}>
            <div className="ms-section-head-copy">
              <h4 style={{ margin: 0, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', color: chartColors.accent }}>
                Tier capacity units
              </h4>
            </div>
          </div>
          <div className="ms-drawer-lifecycle">
            {(['tier0', 'tier1', 'tier2', 'tier3', 'tier4'] as const).map((key) => (
              <div className="cell" key={key}>
                <span>{key}</span>
                <strong>{policy.tierCapacityUnits[key]}</strong>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Connectivity</div>
            <h3>Relay attachments</h3>
            <p>Configured relay endpoints for this network in priority order.</p>
          </div>
        </div>
        {selectedNetwork.attachments.length === 0 ? (
          <EmptyState title="No relay attachments configured" />
        ) : (
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Kind</span>
              <span>Endpoint</span>
              <span className="num">Priority</span>
            </div>
            {selectedNetwork.attachments.map((attachment) => (
              <div
                key={`${attachment.kind}-${attachment.endpoint}`}
                className="ms-assignments-row"
                style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 2fr) 80px' }}
              >
                <span>
                  <StatusBadge status="info" dot={false}>
                    {attachment.kind}
                  </StatusBadge>
                </span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {attachment.endpoint}
                </span>
                <span className="num">{attachment.priority}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
