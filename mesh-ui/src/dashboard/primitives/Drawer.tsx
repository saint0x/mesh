import { useEffect, type ReactNode } from 'react'

export interface DrawerProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  eyebrow?: ReactNode
  actions?: ReactNode
  width?: number
  children: ReactNode
}

/**
 * Right-side sliding drawer. Closes on Escape and backdrop click.
 */
export function Drawer({ open, onClose, title, eyebrow, actions, width = 520, children }: DrawerProps) {
  useEffect(() => {
    if (!open) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, open])

  return (
    <div className={open ? 'ms-drawer open' : 'ms-drawer'} aria-hidden={!open}>
      <div className="ms-drawer-backdrop" onClick={onClose} />
      <aside
        className="ms-drawer-panel"
        style={{ width }}
        role="dialog"
        aria-modal={open ? 'true' : 'false'}
      >
        <header className="ms-drawer-head">
          <div>
            {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
            {title ? <h3>{title}</h3> : null}
          </div>
          <div className="ms-drawer-actions">
            {actions}
            <button className="ms-drawer-close" onClick={onClose} aria-label="Close drawer">
              ×
            </button>
          </div>
        </header>
        <div className="ms-drawer-body">{children}</div>
      </aside>
    </div>
  )
}
