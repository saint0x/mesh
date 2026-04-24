import { useRef, useState, type ReactNode } from 'react'
import { ChartTooltipLayer, useChartTooltip } from './Tooltip'
import { containerCoordsFromEvent } from './chartHelpers'
import { chartColors } from './chartTokens'

export interface HeatmapProps {
  rows: number[][]
  rowLabels?: string[]
  columnLabels?: string[]
  height?: number
  max?: number
  colorScale?: (value: number) => string
  renderTooltip?: (row: number, column: number, value: number) => ReactNode
}

const defaultColorScale = (v: number) => `rgba(102,240,192,${(0.05 + v * 0.85).toFixed(3)})`

export function Heatmap({
  rows,
  rowLabels,
  columnLabels,
  height = 220,
  max,
  colorScale = defaultColorScale,
  renderTooltip,
}: HeatmapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const tooltip = useChartTooltip()
  const [hoverKey, setHoverKey] = useState<string | null>(null)

  const rowCount = rows.length
  const colCount = rows[0]?.length ?? 0
  if (rowCount === 0 || colCount === 0) {
    return <div className="chart-empty">No activity data in this range.</div>
  }

  const cellW = 22
  const cellH = 22
  const gap = 3
  const padL = 44
  const padR = 12
  const padT = 8
  const padB = 18
  const W = colCount * (cellW + gap) + padL + padR
  const H = rowCount * (cellH + gap) + padT + padB

  let computedMax = max ?? 0
  if (!max) {
    for (const row of rows) for (const v of row) if (v > computedMax) computedMax = v
  }
  const denom = computedMax || 1

  const showCell = (row: number, column: number, value: number, e: React.MouseEvent) => {
    const cont = containerRef.current
    if (!cont) return
    const { x, y } = containerCoordsFromEvent(cont, e)
    setHoverKey(`${row}-${column}`)
    tooltip.show(x, y, renderTooltip ? renderTooltip(row, column, value) : null)
  }

  const hideCell = () => {
    setHoverKey(null)
    tooltip.hide()
  }

  return (
    <div className="chart-host chart-heatmap" ref={containerRef}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={height}>
        {rows.map((row, r) =>
          row.map((value, c) => {
            const key = `${r}-${c}`
            const isHover = hoverKey === key
            return (
              <rect
                key={key}
                x={padL + c * (cellW + gap)}
                y={padT + r * (cellH + gap)}
                width={cellW}
                height={cellH}
                rx={4}
                fill={colorScale(value / denom)}
                stroke={isHover ? '#ffffff' : 'rgba(102,240,192,0.18)'}
                strokeWidth={isHover ? 1.4 : 0.5}
                onMouseEnter={(e) => showCell(r, c, value, e)}
                onMouseMove={(e) => showCell(r, c, value, e)}
                onMouseLeave={hideCell}
                style={{ cursor: 'pointer', transition: 'stroke 90ms ease' }}
              />
            )
          }),
        )}
        {rowLabels?.map((label, i) => (
          <text
            key={label}
            x={padL - 6}
            y={padT + i * (cellH + gap) + cellH / 2 + 3}
            fill={chartColors.text}
            fontSize={9}
            textAnchor="end"
          >
            {label}
          </text>
        ))}
        {columnLabels?.map((label, i) => {
          if (i % Math.max(1, Math.floor(colCount / 8)) !== 0 && i !== colCount - 1) return null
          return (
            <text
              key={label}
              x={padL + i * (cellW + gap) + cellW / 2}
              y={H - 4}
              fill={chartColors.text}
              fontSize={9}
              textAnchor="middle"
            >
              {label}
            </text>
          )
        })}
      </svg>
      <ChartTooltipLayer
        state={tooltip.state}
        containerRef={containerRef}
        estimatedWidth={210}
        estimatedHeight={120}
      />
    </div>
  )
}
