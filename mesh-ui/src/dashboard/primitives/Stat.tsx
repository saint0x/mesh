import type { ReactNode } from 'react'

export interface StatProps {
  label: ReactNode
  value: ReactNode
  delta?: ReactNode
  caption?: ReactNode
  trend?: 'up' | 'down' | 'flat'
  accent?: 'accent' | 'warm' | 'cool' | 'violet' | 'danger' | 'neutral'
  monospace?: boolean
  align?: 'left' | 'center'
}

/**
 * Single KPI / metric card. Used in the metric strip at the top of each page.
 */
export function Stat({
  label,
  value,
  delta,
  caption,
  trend,
  accent = 'neutral',
  monospace,
  align = 'left',
}: StatProps) {
  return (
    <article className={`ms-stat ms-stat-${accent} ms-stat-${align}`}>
      <div className="ms-stat-label">{label}</div>
      <div className={monospace ? 'ms-stat-value mono' : 'ms-stat-value'}>{value}</div>
      {delta !== undefined ? (
        <div className={`ms-stat-delta ms-stat-delta-${trend ?? 'flat'}`}>{delta}</div>
      ) : null}
      {caption !== undefined ? <div className="ms-stat-caption">{caption}</div> : null}
    </article>
  )
}

/**
 * Container for a row of Stat cards.
 */
export function StatRow({ children }: { children: ReactNode }) {
  return <section className="ms-stat-row">{children}</section>
}
