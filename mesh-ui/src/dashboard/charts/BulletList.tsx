import { useRef, useState, type ReactNode } from 'react'
import { ChartTooltipLayer, useChartTooltip } from './Tooltip'
import { containerCoordsFromEvent } from './chartHelpers'
import { chartColors, formatChartNumber } from './chartTokens'

export interface BulletRow {
  id: string
  label: string
  reservedCap: number
  settled: number
  released: number
  outstanding: number
  targetMarker?: number
  liveMarker?: number
}

export interface BulletListProps {
  rows: BulletRow[]
  rowHeight?: number
  renderTooltip?: (row: BulletRow) => ReactNode
  onSelect?: (row: BulletRow) => void
  selectedId?: string
  labelWidth?: number
}

/**
 * Per-row bullet chart showing a reservation cap with settled/released/outstanding
 * segments, plus optional target and live spent markers. Row hover shows a tooltip.
 */
export function BulletList({
  rows,
  rowHeight = 44,
  renderTooltip,
  onSelect,
  selectedId,
  labelWidth = 112,
}: BulletListProps) {
  const W = 620
  const PAD = { l: labelWidth, r: 22, t: 14, b: 26 }
  const H = Math.max(rows.length * rowHeight + PAD.t + PAD.b, rowHeight + PAD.t + PAD.b)
  const maxCap = rows.reduce((m, r) => Math.max(m, r.reservedCap, r.targetMarker ?? 0), 0) * 1.08 || 1
  const x = (v: number) => PAD.l + (v / maxCap) * (W - PAD.l - PAD.r)

  const containerRef = useRef<HTMLDivElement>(null)
  const tooltip = useChartTooltip()
  const [hoverId, setHoverId] = useState<string | null>(null)

  const showRow = (row: BulletRow, e: React.MouseEvent) => {
    const cont = containerRef.current
    if (!cont) return
    const { x: cxx, y: cyy } = containerCoordsFromEvent(cont, e)
    setHoverId(row.id)
    tooltip.show(cxx, cyy, renderTooltip ? renderTooltip(row) : null)
  }

  const hideRow = () => {
    setHoverId(null)
    tooltip.hide()
  }

  if (rows.length === 0) {
    return <div className="chart-empty">No jobs to display yet.</div>
  }

  return (
    <div className="chart-host chart-bullet" ref={containerRef}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
        <defs>
          <pattern id="bullet-hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="rgba(255,189,89,0.14)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,189,89,0.55)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const xv = PAD.l + f * (W - PAD.l - PAD.r)
          return (
            <g key={f}>
              <line x1={xv} x2={xv} y1={PAD.t} y2={H - PAD.b + 4} stroke={chartColors.grid} />
              <text x={xv} y={H - 8} fill={chartColors.text} fontSize={9} textAnchor="middle">
                {formatChartNumber(maxCap * f)}
              </text>
            </g>
          )
        })}
        {rows.map((row, i) => {
          const y = PAD.t + i * rowHeight
          const barY = y + 14
          const barH = 12
          const isHover = hoverId === row.id
          const isSelected = selectedId === row.id
          const settledEnd = x(row.settled)
          const releasedEnd = x(row.settled + row.released)
          const outstandingEnd = x(row.settled + row.released + row.outstanding)
          return (
            <g
              key={row.id}
              onMouseEnter={(e) => showRow(row, e)}
              onMouseMove={(e) => showRow(row, e)}
              onMouseLeave={hideRow}
              onClick={() => onSelect?.(row)}
              style={{ cursor: onSelect ? 'pointer' : 'default' }}
            >
              <rect x={0} y={y} width={W} height={rowHeight} fill="transparent" />
              {isHover || isSelected ? (
                <rect
                  x={4}
                  y={y + 2}
                  width={W - 8}
                  height={rowHeight - 4}
                  rx={8}
                  fill={isSelected ? 'rgba(102,240,192,0.07)' : 'rgba(255,255,255,0.025)'}
                />
              ) : null}
              <text
                x={8}
                y={y + 26}
                fill={isHover || isSelected ? chartColors.textStrong : chartColors.text}
                fontSize={11}
                fontFamily="ui-monospace, Menlo, monospace"
              >
                {row.label}
              </text>
              <rect
                x={PAD.l}
                y={barY}
                width={x(row.reservedCap) - PAD.l}
                height={barH}
                rx={3}
                fill="rgba(106,169,255,0.12)"
                stroke="rgba(106,169,255,0.36)"
                strokeWidth={0.8}
              />
              {row.settled > 0 ? (
                <rect x={PAD.l} y={barY} width={settledEnd - PAD.l} height={barH} rx={3} fill={chartColors.settled} />
              ) : null}
              {row.released > 0 ? (
                <rect x={settledEnd} y={barY} width={releasedEnd - settledEnd} height={barH} fill={chartColors.released} />
              ) : null}
              {row.outstanding > 0 ? (
                <rect
                  x={releasedEnd}
                  y={barY}
                  width={outstandingEnd - releasedEnd}
                  height={barH}
                  fill="url(#bullet-hatch)"
                  opacity={0.9}
                />
              ) : null}
              {row.targetMarker !== undefined ? (
                <g>
                  <line
                    x1={x(row.targetMarker)}
                    x2={x(row.targetMarker)}
                    y1={barY - 4}
                    y2={barY + barH + 4}
                    stroke="#ffffff"
                    strokeWidth={1.6}
                  />
                  <circle cx={x(row.targetMarker)} cy={barY - 5} r={2} fill="#ffffff" />
                </g>
              ) : null}
              {row.liveMarker !== undefined ? (
                <line
                  x1={x(row.liveMarker)}
                  x2={x(row.liveMarker)}
                  y1={barY + 1}
                  y2={barY + barH - 1}
                  stroke={chartColors.outstanding}
                  strokeWidth={2}
                />
              ) : null}
              <text
                x={x(row.reservedCap) + 6}
                y={y + 26}
                fill={chartColors.text}
                fontSize={10}
                fontFamily="ui-monospace, Menlo, monospace"
              >
                {formatChartNumber(row.reservedCap)}
              </text>
            </g>
          )
        })}
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
