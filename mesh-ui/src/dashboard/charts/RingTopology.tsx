import { useMemo, useRef, useState, type ReactNode } from 'react'
import { ChartTooltipLayer, useChartTooltip } from './Tooltip'
import { containerCoordsFromEvent } from './chartHelpers'
import { chartColors } from './chartTokens'

export type RingNodeStatus = 'healthy' | 'syncing' | 'offline' | 'unknown'

export interface RingChartNode {
  id: string
  label: string
  position: number
  status: RingNodeStatus
  load?: number // 0..1
}

export interface RingTopologyProps {
  nodes: RingChartNode[]
  height?: number
  centerLabel?: string
  centerValue?: string
  centerCaption?: string
  renderTooltip?: (node: RingChartNode) => ReactNode
  onSelectNode?: (node: RingChartNode) => void
  selectedId?: string
  showCrossChords?: boolean
}

/**
 * Ring diagram with shard arcs, neighbor edges, optional cross-chords,
 * interactive nodes, and a recharts-class floating tooltip.
 */
export function RingTopology({
  nodes,
  height = 320,
  centerLabel = 'RING',
  centerValue,
  centerCaption = 'workers',
  renderTooltip,
  onSelectNode,
  selectedId,
  showCrossChords = false,
}: RingTopologyProps) {
  const W = 620
  const H = height
  const cx = W / 2
  const cy = H / 2
  const radius = Math.min(W, H) * 0.32
  const nodeRadius = 18

  const containerRef = useRef<HTMLDivElement>(null)
  const tooltip = useChartTooltip()
  const [hoverId, setHoverId] = useState<string | null>(null)

  const points = useMemo(() => {
    if (nodes.length === 0) return []
    return nodes.map((n, i) => {
      const angle = (i / nodes.length) * Math.PI * 2 - Math.PI / 2
      return {
        ...n,
        angle,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
      }
    })
  }, [cx, cy, nodes, radius])

  if (nodes.length === 0) {
    return <div className="chart-empty">Ring topology not available yet.</div>
  }

  const showNode = (node: RingChartNode, e: React.MouseEvent) => {
    const cont = containerRef.current
    if (!cont) return
    const { x, y } = containerCoordsFromEvent(cont, e)
    setHoverId(node.id)
    tooltip.show(x, y, renderTooltip ? renderTooltip(node) : null)
  }

  const hideNode = () => {
    setHoverId(null)
    tooltip.hide()
  }

  return (
    <div className="chart-host chart-ring" ref={containerRef}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={height}>
        <defs>
          <radialGradient id="ring-halo" cx="50%" cy="50%" r="50%">
            <stop offset="60%" stopColor="rgba(102,240,192,0)" />
            <stop offset="100%" stopColor="rgba(102,240,192,0.05)" />
          </radialGradient>
          <filter id="ring-glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <circle cx={cx} cy={cy} r={radius + 42} fill="url(#ring-halo)" />
        {/* Shard arcs */}
        {points.map((p, i) => {
          const next = points[(i + 1) % points.length]
          if (!next) return null
          const a1 = p.angle
          const a2 = next.angle - 0.04
          const r = radius + 28
          const x1 = cx + Math.cos(a1) * r
          const y1 = cy + Math.sin(a1) * r
          const x2 = cx + Math.cos(a2) * r
          const y2 = cy + Math.sin(a2) * r
          const color =
            p.status === 'healthy'
              ? chartColors.settled
              : p.status === 'syncing'
                ? chartColors.outstanding
                : p.status === 'offline'
                  ? chartColors.danger
                  : chartColors.textMuted
          return (
            <path
              key={`arc-${p.id}`}
              d={`M${x1},${y1} A${r},${r} 0 0 1 ${x2},${y2}`}
              fill="none"
              stroke={color}
              strokeWidth={3}
              strokeLinecap="round"
              opacity={0.55}
            />
          )
        })}
        {/* Neighbor edges (curved towards center) */}
        {points.map((p, i) => {
          const next = points[(i + 1) % points.length]
          if (!next) return null
          const mx = (p.x + next.x) / 2
          const my = (p.y + next.y) / 2
          const cxc = cx + (mx - cx) * 0.55
          const cyc = cy + (my - cy) * 0.55
          return (
            <path
              key={`edge-${p.id}`}
              d={`M${p.x},${p.y} Q${cxc},${cyc} ${next.x},${next.y}`}
              fill="none"
              stroke="rgba(255,255,255,0.12)"
              strokeWidth={1}
            />
          )
        })}
        {/* Optional cross chords */}
        {showCrossChords
          ? points.map((p, i) => {
              const target = points[(i + Math.floor(points.length / 3)) % points.length]
              if (!target) return null
              return (
                <line
                  key={`chord-${p.id}`}
                  x1={p.x}
                  y1={p.y}
                  x2={target.x}
                  y2={target.y}
                  stroke="rgba(106,169,255,0.08)"
                  strokeWidth={0.6}
                />
              )
            })
          : null}
        {/* Nodes */}
        {points.map((p) => {
          const color =
            p.status === 'healthy'
              ? chartColors.settled
              : p.status === 'syncing'
                ? chartColors.outstanding
                : p.status === 'offline'
                  ? chartColors.danger
                  : chartColors.textMuted
          const isHover = hoverId === p.id
          const isSelected = selectedId === p.id
          const load = Math.max(0, Math.min(1, p.load ?? 0))
          const inner = nodeRadius - 5
          const circumference = 2 * Math.PI * inner
          return (
            <g
              key={p.id}
              onMouseEnter={(e) => showNode(p, e)}
              onMouseMove={(e) => showNode(p, e)}
              onMouseLeave={hideNode}
              onClick={() => onSelectNode?.(p)}
              style={{ cursor: onSelectNode ? 'pointer' : 'default' }}
            >
              <circle
                cx={p.x}
                cy={p.y}
                r={nodeRadius + (isHover || isSelected ? 4 : 0)}
                fill={chartColors.panel}
                stroke={color}
                strokeWidth={isSelected ? 2.4 : 1.6}
                filter={isHover || isSelected ? 'url(#ring-glow)' : undefined}
              />
              {p.load !== undefined ? (
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={inner}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeDasharray={`${load * circumference} ${circumference}`}
                  transform={`rotate(-90 ${p.x} ${p.y})`}
                  opacity={0.8}
                />
              ) : null}
              <text
                x={p.x}
                y={p.y + 3}
                textAnchor="middle"
                fontSize={9}
                fill={chartColors.textStrong}
                fontWeight={600}
              >
                {p.position}
              </text>
              <text
                x={cx + Math.cos(p.angle) * (radius + 58)}
                y={cy + Math.sin(p.angle) * (radius + 58) + 3}
                textAnchor="middle"
                fontSize={9}
                fill={chartColors.text}
                fontFamily="ui-monospace, Menlo, monospace"
              >
                {p.label}
              </text>
            </g>
          )
        })}
        <text x={cx} y={cy - 10} textAnchor="middle" fontSize={10} fill={chartColors.text}>
          {centerLabel}
        </text>
        <text x={cx} y={cy + 9} textAnchor="middle" fontSize={18} fill={chartColors.textStrong} fontWeight={700}>
          {centerValue ?? nodes.length}
        </text>
        <text x={cx} y={cy + 26} textAnchor="middle" fontSize={9} fill={chartColors.text}>
          {centerCaption}
        </text>
      </svg>
      <ChartTooltipLayer
        state={tooltip.state}
        containerRef={containerRef}
        estimatedWidth={220}
        estimatedHeight={180}
      />
    </div>
  )
}
