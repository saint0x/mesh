import type { ReactNode } from 'react'
import { chartColors } from './chartTokens'

export interface ProgressBarProps {
  current: number
  total: number
  label?: ReactNode
  caption?: ReactNode
  color?: string
  background?: string
  height?: number
  indeterminate?: boolean
}

/**
 * Linear progress meter — used for token accounting, resource lock amounts,
 * and any "N of M" live metric.
 */
export function ProgressBar({
  current,
  total,
  label,
  caption,
  color = chartColors.settled,
  background = 'rgba(255,255,255,0.06)',
  height = 8,
  indeterminate,
}: ProgressBarProps) {
  const pct = total > 0 ? Math.max(0, Math.min(100, Math.round((current / total) * 100))) : 0
  return (
    <div className="chart-progress">
      {label ? <div className="chart-progress-label">{label}</div> : null}
      <div
        className="chart-progress-track"
        style={{ background, height }}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={total || 100}
        aria-valuenow={current}
      >
        <span
          className={indeterminate ? 'chart-progress-fill indeterminate' : 'chart-progress-fill'}
          style={{ background: color, width: indeterminate ? '40%' : `${pct}%` }}
        />
      </div>
      {caption ? <div className="chart-progress-caption">{caption}</div> : null}
    </div>
  )
}
