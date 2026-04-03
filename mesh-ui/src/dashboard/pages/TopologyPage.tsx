import type { DashboardPageProps } from '../lib/pageProps'

export function TopologyPage({ controller }: DashboardPageProps) {
  const topology = controller.selectedTopology
  if (!topology) {
    return (
      <section className="panel dashboard-panel">
        <div className="dashboard-empty">No topology data is available for the selected network yet.</div>
      </section>
    )
  }

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Ring topology</div>
            <h3>{topology.networkId}</h3>
          </div>
        </div>
        <div className="dashboard-detail-grid">
          <article><span>Source</span><strong>{topology.source}</strong></article>
          <article><span>Ring stable</span><strong>{topology.ringStable ? 'yes' : 'no'}</strong></article>
          <article><span>Workers</span><strong>{topology.workers.length}</strong></article>
          <article><span>Punch plans</span><strong>{topology.punchPlans.length}</strong></article>
        </div>

        <div className="dashboard-data-table">
          <div className="dashboard-data-head">
            <span>Position</span>
            <span>Worker</span>
            <span>Neighbors</span>
            <span>Shard range</span>
            <span>Tensor endpoints</span>
          </div>
          {topology.workers.map((worker) => (
            <div key={worker.deviceId} className="dashboard-data-row">
              <span className="row-primary">{worker.position ?? 'n/a'}</span>
              <span>{worker.deviceName}</span>
              <span>{worker.leftNeighborId ?? 'n/a'} / {worker.rightNeighborId ?? 'n/a'}</span>
              <span>
                {worker.shardColumnStart !== null && worker.shardColumnStart !== undefined
                  ? `${worker.shardColumnStart} - ${worker.shardColumnEnd ?? 'n/a'}`
                  : 'n/a'}
              </span>
              <span>{worker.tensorPlaneEndpoints.join(', ') || worker.activeEndpoint || 'n/a'}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Punch plans</div>
            <h3>Direct-path coordination</h3>
          </div>
        </div>
        {topology.punchPlans.length > 0 ? (
          <div className="dashboard-data-table">
            <div className="dashboard-data-head">
              <span>Source</span>
              <span>Target</span>
              <span>Reason</span>
              <span>Strategy</span>
              <span>Rendezvous</span>
            </div>
            {topology.punchPlans.map((plan) => (
              <div key={`${plan.sourceDeviceId}-${plan.targetDeviceId}-${plan.issuedAtMs}`} className="dashboard-data-row">
                <span className="row-primary">{plan.sourceDeviceId}</span>
                <span>{plan.targetDeviceId}</span>
                <span>{plan.reason}</span>
                <span>{plan.strategy}</span>
                <span>{plan.relayRendezvousRequired ? 'required' : 'not required'}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="dashboard-empty">No live punch plans were returned for this topology snapshot.</div>
        )}
      </section>
    </div>
  )
}
