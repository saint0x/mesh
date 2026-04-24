import type { ReactNode } from 'react'

export interface SectionHeaderProps {
  eyebrow?: ReactNode
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
}

export function SectionHeader({ eyebrow, title, subtitle, actions }: SectionHeaderProps) {
  return (
    <div className="ms-section-head">
      <div className="ms-section-head-copy">
        {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
        <h3>{title}</h3>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {actions ? <div className="ms-section-head-actions">{actions}</div> : null}
    </div>
  )
}
