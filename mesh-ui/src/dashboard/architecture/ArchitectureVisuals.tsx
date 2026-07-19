import type { ArchitectureNodeId, ArchitectureRoute, ArchitectureSlide } from './slides'

interface NodeDefinition {
  id: ArchitectureNodeId
  label: string
  x: number
  y: number
  kind: 'core' | 'support' | 'worker'
}

const nodeMap: Record<ArchitectureNodeId, NodeDefinition> = {
  client: { id: 'client', label: 'Client', x: 92, y: 94, kind: 'support' },
  ui: { id: 'ui', label: 'Mesh UI', x: 92, y: 344, kind: 'support' },
  control: { id: 'control', label: 'Control Plane', x: 270, y: 110, kind: 'core' },
  planner: { id: 'planner', label: 'Planner', x: 270, y: 218, kind: 'core' },
  db: { id: 'db', label: 'State DB', x: 270, y: 334, kind: 'core' },
  scheduler: { id: 'scheduler', label: 'Scheduler', x: 452, y: 104, kind: 'core' },
  relay: { id: 'relay', label: 'Relay', x: 452, y: 344, kind: 'support' },
  tensor: { id: 'tensor', label: 'Tensor Plane', x: 452, y: 220, kind: 'core' },
  agentA: { id: 'agentA', label: 'Agent A', x: 648, y: 96, kind: 'worker' },
  agentB: { id: 'agentB', label: 'Agent B', x: 778, y: 220, kind: 'worker' },
  agentC: { id: 'agentC', label: 'Agent C', x: 648, y: 344, kind: 'worker' },
  zip: { id: 'zip', label: 'ZIP Runtime', x: 648, y: 220, kind: 'core' },
  kv: { id: 'kv', label: 'KV + Checkpoints', x: 778, y: 96, kind: 'support' },
  ledger: { id: 'ledger', label: 'Ledger', x: 778, y: 344, kind: 'support' },
}

const routeTones: Record<ArchitectureRoute['tone'], { stroke: string; glow: string }> = {
  accent: { stroke: '#67f0c0', glow: 'rgba(103, 240, 192, 0.55)' },
  warm: { stroke: '#ffbd59', glow: 'rgba(255, 189, 89, 0.5)' },
  cool: { stroke: '#6bb8ff', glow: 'rgba(107, 184, 255, 0.44)' },
}

export function ArchitectureVisual({
  slide,
  index,
}: {
  slide: ArchitectureSlide
  index: number
}) {
  const focused = new Set(slide.focusNodes)

  return (
    <div className="architecture-visual-shell">
      <svg
        className="architecture-visual"
        viewBox="0 0 870 430"
        role="img"
        aria-label={`${slide.title} architecture flow diagram`}
      >
        <defs>
          <linearGradient id={`mesh-flow-${index}`} x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%" stopColor="#6bb8ff" />
            <stop offset="50%" stopColor="#67f0c0" />
            <stop offset="100%" stopColor="#ffbd59" />
          </linearGradient>
          <radialGradient id={`mesh-halo-${index}`} cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="rgba(103, 240, 192, 0.42)" />
            <stop offset="100%" stopColor="rgba(103, 240, 192, 0)" />
          </radialGradient>
        </defs>

        <g opacity="0.28">
          {Array.from({ length: 12 }, (_, gridIndex) => (
            <line
              key={`vertical-${gridIndex}`}
              x1={gridIndex * 72}
              y1="0"
              x2={gridIndex * 72}
              y2="430"
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="1"
            />
          ))}
          {Array.from({ length: 7 }, (_, gridIndex) => (
            <line
              key={`horizontal-${gridIndex}`}
              x1="0"
              y1={gridIndex * 70}
              x2="870"
              y2={gridIndex * 70}
              stroke="rgba(255,255,255,0.08)"
              strokeWidth="1"
            />
          ))}
        </g>

        <path
          d="M 565 96 C 722 132, 722 304, 565 344"
          fill="none"
          stroke="rgba(255,255,255,0.11)"
          strokeDasharray="8 10"
          strokeWidth="1.4"
        />
        <path
          d="M 565 344 C 612 238, 702 202, 778 96"
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeDasharray="8 10"
          strokeWidth="1.1"
        />
        <path
          d="M 540 220 C 600 220, 614 220, 648 220"
          fill="none"
          stroke={`url(#mesh-flow-${index})`}
          strokeDasharray="10 8"
          strokeWidth="2"
          className="architecture-flow-pulse"
        />

        {slide.routes.map((route, routeIndex) => (
          <RoutePath key={`${route.from}-${route.to}-${routeIndex}`} route={route} />
        ))}

        {Object.values(nodeMap).map((node) => (
          <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
            {focused.has(node.id) ? (
              <circle r="38" fill={`url(#mesh-halo-${index})`} className="architecture-node-glow" />
            ) : null}
            <rect
              x={node.kind === 'worker' ? -48 : -58}
              y="-26"
              width={node.kind === 'worker' ? 96 : 116}
              height="52"
              rx="18"
              fill={focused.has(node.id) ? 'rgba(16, 34, 53, 0.96)' : 'rgba(10, 20, 34, 0.78)'}
              stroke={focused.has(node.id) ? 'rgba(103, 240, 192, 0.7)' : 'rgba(255,255,255,0.14)'}
              strokeWidth={focused.has(node.id) ? 1.6 : 1}
            />
            <circle
              cx={node.kind === 'worker' ? -28 : -38}
              cy="0"
              r="7"
              fill={node.kind === 'worker' ? '#6bb8ff' : focused.has(node.id) ? '#67f0c0' : '#ffbd59'}
            />
            <text
              x={node.kind === 'worker' ? -16 : -26}
              y="-2"
              fill="#edf7ff"
              fontSize="12"
              fontWeight="700"
              letterSpacing="0.02em"
            >
              {node.label}
            </text>
            <text
              x={node.kind === 'worker' ? -16 : -26}
              y="14"
              fill="rgba(179,196,212,0.86)"
              fontSize="10"
            >
              {describeNode(node.id)}
            </text>
          </g>
        ))}
      </svg>

      <div className="architecture-visual-caption">
        <span>Active seam</span>
        <strong>{slide.platformDetail}</strong>
      </div>
    </div>
  )
}

function RoutePath({ route }: { route: ArchitectureRoute }) {
  const source = nodeMap[route.from]
  const target = nodeMap[route.to]
  const dx = target.x - source.x
  const curve = Math.max(28, Math.abs(dx) * 0.22)
  const midX = (source.x + target.x) / 2
  const midY = (source.y + target.y) / 2
  const tone = routeTones[route.tone]

  return (
    <g className="architecture-route">
      <path
        d={`M ${source.x} ${source.y} C ${source.x + curve} ${source.y}, ${target.x - curve} ${target.y}, ${target.x} ${target.y}`}
        fill="none"
        stroke={tone.stroke}
        strokeOpacity="0.86"
        strokeWidth="2.1"
        strokeDasharray="8 8"
        style={{ filter: `drop-shadow(0 0 10px ${tone.glow})` }}
        className="architecture-route-line"
      />
      <rect
        x={midX - 42}
        y={midY - 12}
        width="84"
        height="24"
        rx="12"
        fill="rgba(5, 10, 18, 0.88)"
        stroke="rgba(255,255,255,0.08)"
      />
      <text
        x={midX}
        y={midY + 4}
        textAnchor="middle"
        fill="#d9e9f4"
        fontSize="10.5"
        fontWeight="600"
      >
        {route.label}
      </text>
    </g>
  )
}

function describeNode(nodeId: ArchitectureNodeId) {
  switch (nodeId) {
    case 'client':
      return 'prompt intent'
    case 'ui':
      return 'operator lens'
    case 'control':
      return 'API authority'
    case 'planner':
      return 'phase planning'
    case 'db':
      return 'durable truth'
    case 'scheduler':
      return 'lease logic'
    case 'relay':
      return 'NAT fallback'
    case 'tensor':
      return 'hot data plane'
    case 'agentA':
    case 'agentB':
    case 'agentC':
      return 'mesh worker'
    case 'zip':
      return 'execution core'
    case 'kv':
      return 'session state'
    case 'ledger':
      return 'credits trail'
  }
}
