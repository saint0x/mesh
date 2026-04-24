import type { ReactNode } from 'react'
import type { MutationState } from '../../domain/dashboard'

export interface MutationButtonProps {
  children: ReactNode
  onClick: () => void
  state?: MutationState
  variant?: 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  pendingLabel?: string
  type?: 'button' | 'submit'
  size?: 'md' | 'sm'
}

/**
 * Button that reflects a mutation lifecycle (pending/success/error) coming from
 * the controller's mutationState map. No state of its own.
 */
export function MutationButton({
  children,
  onClick,
  state = 'idle',
  variant = 'primary',
  disabled,
  pendingLabel = 'Working…',
  type = 'button',
  size = 'md',
}: MutationButtonProps) {
  const isPending = state === 'pending'
  return (
    <button
      type={type}
      className={`ms-button ms-button-${variant} ms-button-${size} ms-button-${state}`}
      onClick={onClick}
      disabled={disabled || isPending}
    >
      {isPending ? (
        <span className="ms-button-spinner" aria-hidden />
      ) : state === 'success' ? (
        <span className="ms-button-icon" aria-hidden>
          ✓
        </span>
      ) : state === 'error' ? (
        <span className="ms-button-icon" aria-hidden>
          !
        </span>
      ) : null}
      <span>{isPending ? pendingLabel : children}</span>
    </button>
  )
}
