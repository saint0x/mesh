import type { ReactNode } from 'react'

export type StatusVariant =
  | 'ok'
  | 'warn'
  | 'fail'
  | 'info'
  | 'neutral'
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'dispatched'
  | 'acknowledged'

const variantFor: Record<string, StatusVariant> = {
  ok: 'ok',
  healthy: 'ok',
  online: 'ok',
  ready: 'ok',
  active: 'ok',
  stable: 'ok',
  completed: 'completed',
  success: 'ok',
  running: 'running',
  acknowledged: 'acknowledged',
  dispatched: 'dispatched',
  pending: 'pending',
  syncing: 'warn',
  degraded: 'warn',
  warn: 'warn',
  warning: 'warn',
  stale: 'warn',
  failed: 'failed',
  fail: 'fail',
  error: 'fail',
  offline: 'fail',
  cancelled: 'cancelled',
  canceled: 'cancelled',
  unknown: 'neutral',
}

export interface StatusBadgeProps {
  status: string
  variant?: StatusVariant
  children?: ReactNode
  dot?: boolean
  size?: 'sm' | 'md'
}

export function StatusBadge({ status, variant, children, dot = true, size = 'sm' }: StatusBadgeProps) {
  const normalized = status.toLowerCase()
  const resolved: StatusVariant = variant ?? variantFor[normalized] ?? 'neutral'
  return (
    <span className={`ms-badge ms-badge-${resolved} ms-badge-${size}`}>
      {dot ? <span className="ms-badge-dot" /> : null}
      <span className="ms-badge-label">{children ?? status}</span>
    </span>
  )
}
