import { useRef, useState, type ReactNode } from 'react'
import { ChartTooltipLayer, useChartTooltip } from './Tooltip'
import { containerCoordsFromEvent, svgCoordsFromEvent } from './chartHelpers'
import { chartColors, formatChartNumber } from './chartTokens'

export interface BarsSeries {
  key: string
  label: string
  color: string
  values: number[]
}

export interface ComposedBarsProps {
  labels: string[]
  bars: BarsSeries[] // stacked in order
  line?: { key: string; label: string; color: string; values: number[] }
  height?: number
  renderTooltip?: (index: number) => ReactNode
  xTickInterval?: number
}

/**
 * Stacked bars with optional secondary-axis line overlay and recharts-class tooltip.
 */
export function ComposedBars({
  labels,
  bars,
  line,
  height = 240,
  renderTooltip,
  xTickInterval,
}: ComposedBarsProps) {
  const W = 620
  const H = height
  const PAD = { l: 42, r: line ? 42 : 14, t: 16, b: 26 }
  const innerW = W - PAD.l - PAD.r
  const innerH = H - PAD.t - PAD.b
  const count = labels.length

  let maxBars = 0
  for (let i = 0; i < count; i++) {
    let sum = 0
    for (const b of bars) sum += b.values[i] ?? 0
    if (sum > maxBars) maxBars = sum
  }
  maxBars = maxBars * 1.08 || 1

  const maxLine = line ? (Math.max(...line.values.map((v) => v), 0) || 1) * 1.1 : 1

  const barW = count > 0 ? innerW / count - 1 : 0
  const xStart = (i: number) => PAD.l + (i / Math.max(count, 1)) * innerW
  const yRight = (v: number) => PAD.t + innerH - (v / maxLine) * innerH

  const linePath = line
    ? line.values
        .map((v, i) => `${i === 0 ? 'M' : 'L'}${xStart(i) + barW / 2},${yRight(v)}`)
        .join(' ')
    : ''

  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltip = useChartTooltip()
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (count === 0) return
    const svg = svgRef.current
    const cont = containerRef.current
    if (!svg || !cont) return
    const { svgX } = svgCoordsFromEvent(svg, e, W, H)
    const ratio = (svgX - PAD.l) / innerW
    const idx = Math.max(0, Math.min(count - 1, Math.floor(ratio * count)))
    setHoverIdx(idx)
    const { x, y } = containerCoordsFromEvent(cont, e)
    tooltip.show(x, y, renderTooltip ? renderTooltip(idx) : null)
  }

  const handleLeave = () => {
    setHoverIdx(null)
    tooltip.hide()
  }

  if (count === 0) {
    return <div className="chart-empty">No data in this range.</div>
  }

  const tickStep = xTickInterval ?? Math.max(1, Math.floor(count / 8))

  return (
    <div className="chart-host" ref={containerRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height={height}
        onMouseMove={handleMove}
        onMouseLeave={handleLeave}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={PAD.t + innerH * (1 - f)}
              y2={PAD.t + innerH * (1 - f)}
              stroke={chartColors.grid}
            />
            <text
              x={PAD.l - 6}
              y={PAD.t + innerH * (1 - f) + 3}
              fill={chartColors.text}
              fontSize={9}
              textAnchor="end"
            >
              {formatChartNumber(maxBars * f)}
            </text>
            {line ? (
              <text
                x={W - PAD.r + 6}
                y={PAD.t + innerH * (1 - f) + 3}
                fill={chartColors.text}
                fontSize={9}
              >
                {formatChartNumber(maxLine * f)}
              </text>
            ) : null}
          </g>
        ))}
        {labels.map((_, i) => {
          let stack = 0
          return bars.map((b, bi) => {
            const value = b.values[i] ?? 0
            const h = (value / maxBars) * innerH
            const yy = PAD.t + innerH - stack - h
            stack += h
            const isTop = bi === bars.length - 1
            const isHover = hoverIdx === i
            return (
              <rect
                key={`${b.key}-${i}`}
                x={xStart(i)}
                y={yy}
                width={barW}
                height={h}
                fill={b.color}
                opacity={isHover ? 1 : 0.9}
                rx={isTop ? 1.6 : 0}
              />
            )
          })
        })}
        {line ? (
          <g>
            <path d={linePath} fill="none" stroke={line.color} strokeWidth={1.8} />
            {hoverIdx !== null ? (
              <circle
                cx={xStart(hoverIdx) + barW / 2}
                cy={yRight(line.values[hoverIdx] ?? 0)}
                r={3.2}
                fill={line.color}
                stroke={chartColors.panel}
                strokeWidth={1.3}
                pointerEvents="none"
              />
            ) : null}
          </g>
        ) : null}
        {labels.map((label, i) => {
          if (i % tickStep !== 0 && i !== count - 1) return null
          return (
            <text
              key={i}
              x={xStart(i) + barW / 2}
              y={H - 8}
              fill={chartColors.text}
              fontSize={9}
              textAnchor="middle"
            >
              {label}
            </text>
          )
        })}
        {hoverIdx !== null ? (
          <line
            x1={xStart(hoverIdx) + barW / 2}
            x2={xStart(hoverIdx) + barW / 2}
            y1={PAD.t}
            y2={PAD.t + innerH}
            stroke="rgba(255,255,255,0.28)"
            strokeDasharray="3 4"
            pointerEvents="none"
          />
        ) : null}
      </svg>
      <ChartTooltipLayer
        state={tooltip.state}
        containerRef={containerRef}
        estimatedWidth={210}
        estimatedHeight={140}
      />
    </div>
  )
}
