import { useMemo, useRef, useState, type ReactNode } from 'react'
import { ChartTooltipLayer, useChartTooltip } from './Tooltip'
import { containerCoordsFromEvent, svgCoordsFromEvent } from './chartHelpers'
import { chartColors, formatChartNumber } from './chartTokens'

export interface LifecycleSeries {
  key: string
  label: string
  color: string
  values: number[]
}

export interface LifecycleAreaProps {
  labels: string[]
  series: LifecycleSeries[] // stacked bottom-to-top in order
  height?: number
  referenceLine?: { value: number; label?: string; color?: string }
  renderTooltip?: (index: number) => ReactNode
  xTickInterval?: number
}

/**
 * Stacked area chart with recharts-class tooltip, vertical guide, per-series dots.
 * Pure inline SVG. Generic over N series.
 */
export function LifecycleArea({
  labels,
  series,
  height = 240,
  referenceLine,
  renderTooltip,
  xTickInterval,
}: LifecycleAreaProps) {
  const W = 620
  const H = height
  const PAD = { l: 44, r: 14, t: 18, b: 24 }
  const innerW = W - PAD.l - PAD.r
  const innerH = H - PAD.t - PAD.b

  const count = labels.length
  const stackedTops = useMemo(() => {
    if (count === 0) return [] as number[][]
    const result: number[][] = []
    for (let i = 0; i < count; i++) {
      let running = 0
      const row: number[] = []
      for (const s of series) {
        running += s.values[i] ?? 0
        row.push(running)
      }
      result.push(row)
    }
    return result
  }, [count, series])

  const max = useMemo(() => {
    let m = 0
    for (const row of stackedTops) for (const v of row) if (v > m) m = v
    if (referenceLine) m = Math.max(m, referenceLine.value)
    return m * 1.08 || 1
  }, [referenceLine, stackedTops])

  const x = (i: number) => (count <= 1 ? PAD.l : PAD.l + (i / (count - 1)) * innerW)
  const y = (v: number) => PAD.t + innerH - (v / max) * innerH

  const tickStep = xTickInterval ?? Math.max(1, Math.floor(count / 8))

  const paths = useMemo(() => {
    if (count === 0) return [] as string[]
    const out: string[] = []
    for (let si = 0; si < series.length; si++) {
      const topPoints: string[] = []
      const basePoints: string[] = []
      for (let i = 0; i < count; i++) {
        const topRow = stackedTops[i] ?? []
        topPoints.push(`${i === 0 ? 'M' : 'L'}${x(i)},${y(topRow[si] ?? 0)}`)
        const baseRow = stackedTops[count - 1 - i] ?? []
        const baseVal = si === 0 ? 0 : baseRow[si - 1] ?? 0
        basePoints.push(`L${x(count - 1 - i)},${y(baseVal)}`)
      }
      out.push(`${topPoints.join(' ')} ${basePoints.join(' ')} Z`)
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stackedTops, series, count, max])

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
    const idx = Math.max(0, Math.min(count - 1, Math.round(ratio * (count - 1))))
    setHoverIdx(idx)
    const { x: cx, y: cy } = containerCoordsFromEvent(cont, e)
    const content = renderTooltip ? renderTooltip(idx) : null
    tooltip.show(cx, cy, content)
  }

  const handleLeave = () => {
    setHoverIdx(null)
    tooltip.hide()
  }

  if (count === 0) {
    return <div className="chart-empty">No data in this range.</div>
  }

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
        <defs>
          {series.map((s) => (
            <linearGradient key={s.key} id={`chart-grad-${s.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity={0.65} />
              <stop offset="100%" stopColor={s.color} stopOpacity={0.03} />
            </linearGradient>
          ))}
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line
            key={f}
            x1={PAD.l}
            x2={W - PAD.r}
            y1={PAD.t + innerH * (1 - f)}
            y2={PAD.t + innerH * (1 - f)}
            stroke={chartColors.grid}
          />
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <text
            key={f}
            x={PAD.l - 8}
            y={PAD.t + innerH * (1 - f) + 3}
            fill={chartColors.text}
            fontSize={9}
            textAnchor="end"
          >
            {formatChartNumber(max * f)}
          </text>
        ))}
        {paths.map((d, i) => {
          const seriesEntry = series[i]
          if (!seriesEntry) return null
          return (
            <path
              key={seriesEntry.key}
              d={d}
              fill={`url(#chart-grad-${seriesEntry.key})`}
              stroke={seriesEntry.color}
              strokeWidth={1.4}
            />
          )
        })}
        {referenceLine ? (
          <g>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={y(referenceLine.value)}
              y2={y(referenceLine.value)}
              stroke={referenceLine.color ?? chartColors.outstanding}
              strokeDasharray="3 4"
            />
            {referenceLine.label ? (
              <text
                x={W - PAD.r - 4}
                y={y(referenceLine.value) - 4}
                fill={referenceLine.color ?? chartColors.outstanding}
                fontSize={9}
                textAnchor="end"
              >
                {referenceLine.label}
              </text>
            ) : null}
          </g>
        ) : null}
        {labels.map((label, i) => {
          if (i % tickStep !== 0 && i !== count - 1) return null
          return (
            <text
              key={i}
              x={x(i)}
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
          <g pointerEvents="none">
            <line
              x1={x(hoverIdx)}
              x2={x(hoverIdx)}
              y1={PAD.t}
              y2={PAD.t + innerH}
              stroke="rgba(255,255,255,0.28)"
              strokeDasharray="3 4"
            />
            {series.map((s, si) => {
              const row = stackedTops[hoverIdx] ?? []
              const topValue = row[si] ?? 0
              return (
                <circle
                  key={s.key}
                  cx={x(hoverIdx)}
                  cy={y(topValue)}
                  r={3.2}
                  fill={s.color}
                  stroke={chartColors.panel}
                  strokeWidth={1.3}
                />
              )
            })}
          </g>
        ) : null}
      </svg>
      <ChartTooltipLayer
        state={tooltip.state}
        containerRef={containerRef}
        estimatedWidth={220}
        estimatedHeight={150}
      />
    </div>
  )
}
