import type { ReactNode } from 'react'

export interface EmptyStateProps {
  title: ReactNode
  hint?: ReactNode
  action?: ReactNode
}

export function EmptyState({ title, hint, action }: EmptyStateProps) {
  return (
    <div className="ms-empty">
      <strong>{title}</strong>
      {hint ? <p>{hint}</p> : null}
      {action ? <div className="ms-empty-action">{action}</div> : null}
    </div>
  )
}
