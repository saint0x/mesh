import { useEffect, useState, type ReactNode, type RefObject } from 'react'
import type { TooltipState } from './chartHelpers'

// eslint-disable-next-line react-refresh/only-export-components
export function useChartTooltip() {
  const [state, setState] = useState<TooltipState>({ visible: false, x: 0, y: 0, content: null })
  const show = (x: number, y: number, content: ReactNode) =>
    setState({ visible: true, x, y, content })
  const move = (x: number, y: number) =>
    setState((current) => (current.visible ? { ...current, x, y } : current))
  const hide = () => setState((current) => ({ ...current, visible: false }))
  return { state, show, move, hide }
}

export function ChartTooltipLayer({
  state,
  containerRef,
  estimatedWidth = 200,
  estimatedHeight = 120,
}: {
  state: TooltipState
  containerRef: RefObject<HTMLDivElement | null>
  estimatedWidth?: number
  estimatedHeight?: number
}) {
  const [size, setSize] = useState({ width: 600, height: 400 })

  useEffect(() => {
    const node = containerRef.current
    if (!node) return
    const update = () => setSize({ width: node.clientWidth, height: node.clientHeight })
    update()
    if (typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(update)
    observer.observe(node)
    return () => observer.disconnect()
  }, [containerRef])

  let tx = state.x + 14
  let ty = state.y + 14
  if (tx + estimatedWidth > size.width - 6) tx = state.x - estimatedWidth - 14
  if (ty + estimatedHeight > size.height - 6) ty = state.y - estimatedHeight - 14
  if (tx < 4) tx = 4
  if (ty < 4) ty = 4
  return (
    <div
      className="chart-tooltip"
      style={{
        opacity: state.visible ? 1 : 0,
        transform: `translate(${tx}px, ${ty}px)`,
      }}
      aria-hidden={!state.visible}
    >
      {state.content}
    </div>
  )
}

export function ChartTooltipHeader({ children }: { children: ReactNode }) {
  return <div className="chart-tooltip-header">{children}</div>
}

export function ChartTooltipRow({
  color,
  label,
  value,
  outlined,
}: {
  color: string
  label: string
  value: ReactNode
  outlined?: boolean
}) {
  return (
    <div className="chart-tooltip-row">
      <span
        className="chart-tooltip-dot"
        style={
          outlined
            ? { background: 'transparent', boxShadow: `inset 0 0 0 1.5px ${color}` }
            : { background: color }
        }
      />
      <span className="chart-tooltip-label">{label}</span>
      <span className="chart-tooltip-value">{value}</span>
    </div>
  )
}

export function ChartTooltipFoot({ children }: { children: ReactNode }) {
  return <div className="chart-tooltip-foot">{children}</div>
}

