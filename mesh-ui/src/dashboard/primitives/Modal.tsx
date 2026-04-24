import { useEffect, type ReactNode } from 'react'

export interface ModalProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  eyebrow?: ReactNode
  children: ReactNode
  footer?: ReactNode
  width?: number
}

/**
 * Centered modal dialog for focused actions (Create pool, Join pool, etc.).
 */
export function Modal({ open, onClose, title, eyebrow, children, footer, width = 480 }: ModalProps) {
  useEffect(() => {
    if (!open) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, open])

  if (!open) return null
  return (
    <div className="ms-modal" role="dialog" aria-modal="true">
      <div className="ms-modal-backdrop" onClick={onClose} />
      <div className="ms-modal-panel" style={{ width }}>
        <header className="ms-modal-head">
          <div>
            {eyebrow ? <div className="eyebrow">{eyebrow}</div> : null}
            {title ? <h3>{title}</h3> : null}
          </div>
          <button className="ms-drawer-close" onClick={onClose} aria-label="Close dialog">
            ×
          </button>
        </header>
        <div className="ms-modal-body">{children}</div>
        {footer ? <footer className="ms-modal-foot">{footer}</footer> : null}
      </div>
    </div>
  )
}
