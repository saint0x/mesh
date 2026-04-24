// Pure helpers extracted from Tooltip.tsx so the tooltip module exports only
// React components/hooks and stays compatible with react-refresh.

import type { ReactNode } from 'react'

export interface TooltipState {
  visible: boolean
  x: number
  y: number
  content: ReactNode
}

export function svgCoordsFromEvent(
  svg: SVGSVGElement,
  e: { clientX: number; clientY: number },
  viewBoxW: number,
  viewBoxH: number,
): { svgX: number; svgY: number } {
  const rect = svg.getBoundingClientRect()
  const svgX = ((e.clientX - rect.left) / rect.width) * viewBoxW
  const svgY = ((e.clientY - rect.top) / rect.height) * viewBoxH
  return { svgX, svgY }
}

export function containerCoordsFromEvent(
  container: HTMLDivElement,
  e: { clientX: number; clientY: number },
): { x: number; y: number } {
  const rect = container.getBoundingClientRect()
  return { x: e.clientX - rect.left, y: e.clientY - rect.top }
}
