import { useMemo, useState } from 'react'
import {
  ChartTooltipFoot,
  ChartTooltipHeader,
  ChartTooltipRow,
  RingTopology,
  chartColors,
  type RingChartNode,
  type RingNodeStatus,
} from '../charts'
import { Drawer, EmptyState, Stat, StatRow, StatusBadge } from '../primitives'
import { formatBytes } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'
import type { TopologyWorkerRecord } from '../../domain/dashboard'

function workerStatus(worker: TopologyWorkerRecord): RingNodeStatus {
  const status = worker.status.toLowerCase()
  if (status === 'online' || status === 'active' || status === 'healthy') return 'healthy'
  if (status === 'syncing' || status === 'pending') return 'syncing'
  if (status === 'offline' || status === 'unreachable') return 'offline'
  return 'unknown'
}

function loadFromShardSpan(worker: TopologyWorkerRecord): number {
  if (worker.shardColumnStart == null || worker.shardColumnEnd == null) return 0.5
  const span = Math.max(0, worker.shardColumnEnd - worker.shardColumnStart)
  return Math.max(0.1, Math.min(1, span / 64))
}

export function TopologyPage({ controller }: DashboardPageProps) {
  const topology = controller.selectedTopology
  const [selectedWorkerId, setSelectedWorkerId] = useState<string | null>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const workers = useMemo(
    () => (topology ? [...topology.workers].sort((a, b) => (a.position ?? 0) - (b.position ?? 0)) : []),
    [topology],
  )

  const ringNodes: RingChartNode[] = useMemo(
    () =>
      workers.map((worker, i) => ({
        id: worker.deviceId,
        label:
          worker.deviceName.length > 14
            ? `${worker.deviceName.slice(0, 12)}…`
            : worker.deviceName,
        position: worker.position ?? i,
        status: workerStatus(worker),
        load: loadFromShardSpan(worker),
      })),
    [workers],
  )

  const healthy = useMemo(
    () => workers.filter((worker) => workerStatus(worker) === 'healthy').length,
    [workers],
  )

  const totalContributed = useMemo(
    () => workers.reduce((sum, worker) => sum + (worker.contributedMemoryBytes ?? 0), 0),
    [workers],
  )

  if (!topology) {
    return (
      <section className="panel dashboard-panel">
        <EmptyState
          title="No topology data yet"
          hint="Run a ring on this network and the topology snapshot will appear here."
        />
      </section>
    )
  }

  const selectedWorker = workers.find((worker) => worker.deviceId === selectedWorkerId)

  const openWorker = (node: RingChartNode) => {
    setSelectedWorkerId(node.id)
    setDrawerOpen(true)
  }

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Workers"
          value={workers.length}
          accent="cool"
          caption={`${healthy} healthy · ${workers.length - healthy} other`}
        />
        <Stat
          label="Ring stable"
          value={topology.ringStable ? 'yes' : 'no'}
          accent={topology.ringStable ? 'accent' : 'warm'}
          caption={`Source: ${topology.source}`}
        />
        <Stat
          label="Punch plans"
          value={topology.punchPlans.length}
          accent={topology.punchPlans.length > 0 ? 'warm' : 'neutral'}
          caption={
            topology.punchPlans.length > 0
              ? 'Live direct-path coordination requested.'
              : 'No active hole-punching attempts.'
          }
        />
        <Stat
          label="Contributed memory"
          value={totalContributed > 0 ? formatBytes(totalContributed) : '—'}
          accent="violet"
          caption="Sum of memory pledged by ring members."
        />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Ring topology</div>
            <h3>{topology.networkId}</h3>
            <p>Click a worker to open its assignment, neighbors, and tensor endpoints.</p>
          </div>
          <div className="ms-section-head-actions">
            <StatusBadge status={topology.ringStable ? 'stable' : 'pending'} />
          </div>
        </div>
        {ringNodes.length === 0 ? (
          <EmptyState title="Empty ring" hint="No workers have joined this network's ring yet." />
        ) : (
          <div className="ms-chart-card">
            <RingTopology
              nodes={ringNodes}
              onSelectNode={openWorker}
              {...(selectedWorkerId ? { selectedId: selectedWorkerId } : {})}
              centerCaption="workers"
              showCrossChords={ringNodes.length >= 8}
              renderTooltip={(node) => {
                const worker = workers.find((w) => w.deviceId === node.id)
                return (
                  <>
                    <ChartTooltipHeader>{node.label}</ChartTooltipHeader>
                    <ChartTooltipRow color={chartColors.cool} label="position" value={`#${node.position}`} />
                    <ChartTooltipRow
                      color={
                        node.status === 'healthy'
                          ? chartColors.settled
                          : node.status === 'syncing'
                            ? chartColors.outstanding
                            : chartColors.danger
                      }
                      label="status"
                      value={node.status}
                    />
                    {worker?.shardColumnStart != null ? (
                      <ChartTooltipRow
                        color={chartColors.released}
                        label="shards"
                        value={`${worker.shardColumnStart}–${worker.shardColumnEnd ?? '?'}`}
                      />
                    ) : null}
                    {worker?.contributedMemoryBytes ? (
                      <ChartTooltipRow
                        color={chartColors.outstanding}
                        label="memory"
                        value={formatBytes(worker.contributedMemoryBytes)}
                      />
                    ) : null}
                    {worker?.activePath ? (
                      <ChartTooltipFoot>via {worker.activePath}</ChartTooltipFoot>
                    ) : null}
                  </>
                )
              }}
            />
          </div>
        )}
      </section>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Punch plans</div>
            <h3>Direct-path coordination</h3>
            <p>Hole-punching requests issued by the control-plane to bring peers onto direct paths.</p>
          </div>
        </div>
        {topology.punchPlans.length === 0 ? (
          <EmptyState
            title="No live punch plans"
            hint="The mesh is connected via existing direct paths or relays."
          />
        ) : (
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Source</span>
              <span>Target</span>
              <span>Reason</span>
              <span>Strategy</span>
              <span>Rendezvous</span>
            </div>
            {topology.punchPlans.map((plan) => (
              <div
                key={`${plan.sourceDeviceId}-${plan.targetDeviceId}-${plan.issuedAtMs}`}
                className="ms-assignments-row"
              >
                <span className="mono" style={{ fontSize: 11 }}>
                  {plan.sourceDeviceId.slice(0, 12)}
                </span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {plan.targetDeviceId.slice(0, 12)}
                </span>
                <span>{plan.reason}</span>
                <span>{plan.strategy}</span>
                <span>
                  <StatusBadge
                    status={plan.relayRendezvousRequired ? 'warn' : 'ok'}
                    dot={false}
                  >
                    {plan.relayRendezvousRequired ? 'relay required' : 'direct'}
                  </StatusBadge>
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      <Drawer
        open={drawerOpen && selectedWorker !== undefined}
        onClose={() => setDrawerOpen(false)}
        eyebrow={selectedWorker ? `Position #${selectedWorker.position ?? '?'}` : 'Worker'}
        title={selectedWorker?.deviceName ?? null}
      >
        {selectedWorker ? (
          <>
            <section className="ms-drawer-section">
              <h4>Identity</h4>
              <div className="ms-drawer-lifecycle">
                <div className="cell">
                  <span>Device ID</span>
                  <strong className="mono" style={{ fontSize: 12 }}>
                    {selectedWorker.deviceId.slice(0, 14)}
                  </strong>
                </div>
                <div className="cell">
                  <span>Peer ID</span>
                  <strong className="mono" style={{ fontSize: 12 }}>
                    {selectedWorker.peerId ? selectedWorker.peerId.slice(0, 14) : '—'}
                  </strong>
                </div>
                <div className="cell">
                  <span>Status</span>
                  <strong>
                    <StatusBadge status={selectedWorker.status} />
                  </strong>
                </div>
                <div className="cell">
                  <span>Memory</span>
                  <strong>
                    {selectedWorker.contributedMemoryBytes
                      ? formatBytes(selectedWorker.contributedMemoryBytes)
                      : '—'}
                  </strong>
                </div>
              </div>
            </section>

            <section className="ms-drawer-section">
              <h4>Shard assignment</h4>
              <div className="ms-drawer-lifecycle">
                <div className="cell">
                  <span>Column start</span>
                  <strong>{selectedWorker.shardColumnStart ?? '—'}</strong>
                </div>
                <div className="cell">
                  <span>Column end</span>
                  <strong>{selectedWorker.shardColumnEnd ?? '—'}</strong>
                </div>
                <div className="cell">
                  <span>Left neighbor</span>
                  <strong className="mono" style={{ fontSize: 12 }}>
                    {selectedWorker.leftNeighborId ? selectedWorker.leftNeighborId.slice(0, 12) : '—'}
                  </strong>
                </div>
                <div className="cell">
                  <span>Right neighbor</span>
                  <strong className="mono" style={{ fontSize: 12 }}>
                    {selectedWorker.rightNeighborId ? selectedWorker.rightNeighborId.slice(0, 12) : '—'}
                  </strong>
                </div>
              </div>
            </section>

            <section className="ms-drawer-section">
              <h4>Tensor plane endpoints</h4>
              {selectedWorker.tensorPlaneEndpoints.length === 0 ? (
                <EmptyState title="No tensor endpoints registered" />
              ) : (
                <div className="ms-assignments">
                  {selectedWorker.tensorPlaneEndpoints.map((endpoint) => (
                    <div key={endpoint} className="ms-assignments-row" style={{ gridTemplateColumns: '1fr' }}>
                      <span className="mono" style={{ fontSize: 11 }}>
                        {endpoint}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              {selectedWorker.activeEndpoint ? (
                <div style={{ marginTop: 8, fontSize: 11, color: chartColors.text }}>
                  Active path:{' '}
                  <strong style={{ color: chartColors.textStrong }}>
                    {selectedWorker.activePath ?? 'unknown'}
                  </strong>{' '}
                  → <span className="mono">{selectedWorker.activeEndpoint}</span>
                </div>
              ) : null}
            </section>
          </>
        ) : null}
      </Drawer>
    </div>
  )
}
