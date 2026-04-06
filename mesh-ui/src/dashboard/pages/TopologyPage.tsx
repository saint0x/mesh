import { useMemo } from 'react'
import { formatBytes } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'
import type { TopologyWorkerRecord } from '../../domain/dashboard'

const RING_CX = 200
const RING_CY = 190
const RING_R = 140
const NODE_R = 38

function nodePosition(index: number, total: number) {
  const angle = (2 * Math.PI * index) / total - Math.PI / 2
  return { x: RING_CX + RING_R * Math.cos(angle), y: RING_CY + RING_R * Math.sin(angle) }
}

function RingVisualization({ workers }: { workers: TopologyWorkerRecord[] }) {
  const sorted = useMemo(
    () => [...workers].sort((a, b) => (a.position ?? 0) - (b.position ?? 0)),
    [workers],
  )
  const n = sorted.length
  if (n === 0) return null

  const positions = sorted.map((_, i) => nodePosition(i, n))

  return (
    <div className="dashboard-topology">
      <svg viewBox="0 0 400 380" width="100%" height="100%" style={{ minHeight: 340 }}>
        {/* Ring circle (subtle) */}
        <circle
          cx={RING_CX}
          cy={RING_CY}
          r={RING_R}
          fill="none"
          stroke="var(--line)"
          strokeWidth={1}
          strokeDasharray="6 4"
        />

        {/* Neighbor edges */}
        {positions.map((pos, i) => {
          const next = positions[(i + 1) % n]!
          return (
            <line
              key={`edge-${i}`}
              x1={pos.x}
              y1={pos.y}
              x2={next.x}
              y2={next.y}
              stroke="rgba(102, 240, 192, 0.3)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
          )
        })}

        {/* Nodes */}
        {sorted.map((worker, i) => {
          const pos = positions[i]!
          const isHealthy = worker.status === 'online' || worker.status === 'active'
          const fillColor = isHealthy
            ? 'color-mix(in srgb, var(--accent) 14%, transparent)'
            : 'rgba(255, 97, 97, 0.1)'
          const strokeColor = isHealthy
            ? 'rgba(102, 240, 192, 0.4)'
            : 'rgba(255, 97, 97, 0.3)'
          const shortName = worker.deviceName.length > 14
            ? worker.deviceName.slice(0, 12) + '…'
            : worker.deviceName

          return (
            <g key={worker.deviceId}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={NODE_R}
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={1.5}
              />
              {isHealthy && (
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={NODE_R}
                  fill="none"
                  stroke="rgba(102, 240, 192, 0.08)"
                  strokeWidth={8}
                />
              )}
              <text
                x={pos.x}
                y={pos.y - 6}
                textAnchor="middle"
                fill="var(--text-strong)"
                fontSize={9}
                fontWeight={600}
                fontFamily="'Instrument Sans', sans-serif"
              >
                {shortName}
              </text>
              <text
                x={pos.x}
                y={pos.y + 8}
                textAnchor="middle"
                fill="var(--text-muted)"
                fontSize={8}
                fontFamily="'Instrument Sans', sans-serif"
              >
                pos {worker.position ?? '?'}
              </text>
              <text
                x={pos.x}
                y={pos.y + 19}
                textAnchor="middle"
                fill="var(--text-muted)"
                fontSize={7}
                fontFamily="'Instrument Sans', sans-serif"
              >
                {worker.contributedMemoryBytes ? formatBytes(worker.contributedMemoryBytes) : ''}
              </text>
            </g>
          )
        })}

        {/* Center label */}
        <text
          x={RING_CX}
          y={RING_CY - 6}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize={10}
          fontWeight={600}
          fontFamily="'Space Grotesk', sans-serif"
          letterSpacing="0.08em"
        >
          RING
        </text>
        <text
          x={RING_CX}
          y={RING_CY + 10}
          textAnchor="middle"
          fill="var(--text-strong)"
          fontSize={16}
          fontWeight={700}
          fontFamily="'Space Grotesk', sans-serif"
          letterSpacing="-0.02em"
        >
          {n} nodes
        </text>
      </svg>
    </div>
  )
}

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
    <div className="dashboard-stack">
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

          <RingVisualization workers={topology.workers} />
        </section>

        <section className="panel dashboard-panel">
          <div className="dashboard-panel-head">
            <div>
              <div className="eyebrow">Workers</div>
              <h3>Ring member detail</h3>
            </div>
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
                <span>{worker.leftNeighborId ? worker.leftNeighborId.split('-')[0] : 'n/a'} / {worker.rightNeighborId ? worker.rightNeighborId.split('-')[0] : 'n/a'}</span>
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
      </div>

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
                <span className="row-primary">{plan.sourceDeviceId.split('-')[0]}</span>
                <span>{plan.targetDeviceId.split('-')[0]}</span>
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
