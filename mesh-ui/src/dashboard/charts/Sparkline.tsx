import { useRef, useState, type ReactNode } from 'react'
import { ChartTooltipLayer, useChartTooltip } from './Tooltip'
import { containerCoordsFromEvent, svgCoordsFromEvent } from './chartHelpers'
import { chartColors } from './chartTokens'

export interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  fillColor?: string
  renderTooltip?: (index: number, value: number) => ReactNode
}

/**
 * Tiny inline-SVG sparkline with recharts-class tooltip and snapping dot.
 * Cheap enough to use ~hundreds in a single table.
 */
export function Sparkline({
  data,
  width = 96,
  height = 22,
  color = chartColors.settled,
  fillColor = 'rgba(102,240,192,0.18)',
  renderTooltip,
}: SparklineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const tooltip = useChartTooltip()
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)

  if (data.length === 0) {
    return (
      <div
        className="chart-sparkline-empty"
        style={{ width, height, display: 'inline-block' }}
        aria-hidden
      />
    )
  }

  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const v of data) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const range = max - min || 1
  const step = data.length === 1 ? 0 : width / (data.length - 1)
  const yOf = (v: number) => height - 2 - ((v - min) / range) * (height - 4)
  const path = data
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${yOf(v).toFixed(1)}`)
    .join(' ')
  const fillPath = `${path} L${width},${height} L0,${height} Z`
  const last = data[data.length - 1] ?? 0
  const lastY = yOf(last)

  const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current
    const cont = containerRef.current
    if (!svg || !cont) return
    const { svgX } = svgCoordsFromEvent(svg, e, width, height)
    const idx = step === 0 ? 0 : Math.max(0, Math.min(data.length - 1, Math.round(svgX / step)))
    setHoverIdx(idx)
    const { x, y } = containerCoordsFromEvent(cont, e)
    const value = data[idx] ?? 0
    tooltip.show(x, y, renderTooltip ? renderTooltip(idx, value) : null)
  }

  const handleLeave = () => {
    setHoverIdx(null)
    tooltip.hide()
  }

  return (
    <div
      className="chart-host chart-sparkline"
      ref={containerRef}
      style={{ display: 'inline-block', verticalAlign: 'middle' }}
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handleMove}
        onMouseLeave={handleLeave}
        style={{ display: 'block' }}
      >
        <path d={fillPath} fill={fillColor} />
        <path d={path} fill="none" stroke={color} strokeWidth={1.3} />
        <circle cx={width} cy={lastY} r={1.6} fill={color} />
        {hoverIdx !== null ? (
          <g pointerEvents="none">
            <line
              x1={hoverIdx * step}
              x2={hoverIdx * step}
              y1={0}
              y2={height}
              stroke="rgba(255,255,255,0.4)"
              strokeWidth={0.6}
            />
            <circle
              cx={hoverIdx * step}
              cy={yOf(data[hoverIdx] ?? 0)}
              r={2}
              fill={color}
              stroke={chartColors.panel}
              strokeWidth={1}
            />
          </g>
        ) : null}
      </svg>
      <ChartTooltipLayer
        state={tooltip.state}
        containerRef={containerRef}
        estimatedWidth={180}
        estimatedHeight={80}
      />
    </div>
  )
}
