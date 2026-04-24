import { useState } from 'react'
import { EmptyState, Modal, MutationButton, Stat, StatRow, StatusBadge } from '../primitives'
import { chartColors } from '../charts'
import { formatInteger } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function PoolsPage({ controller }: DashboardPageProps) {
  const [createOpen, setCreateOpen] = useState(false)
  const [joinOpen, setJoinOpen] = useState(false)
  const [createName, setCreateName] = useState('')
  const [joinPoolId, setJoinPoolId] = useState('')
  const [joinPubkey, setJoinPubkey] = useState('')
  const [joinName, setJoinName] = useState('')

  const pools = controller.state.pools
  const adminCount = pools.filter((p) => p.role === 'admin').length
  const memberCount = pools.length - adminCount
  const totalPeers = pools.reduce((sum, pool) => sum + pool.peerCount, 0)

  const createState = controller.mutationState['pool:create']?.state ?? 'idle'
  const joinState = controller.mutationState['pool:join']?.state ?? 'idle'

  const handleCreate = async () => {
    if (!createName.trim()) return
    await controller.createPool(createName.trim())
    setCreateName('')
    setCreateOpen(false)
  }

  const handleJoin = async () => {
    if (!joinPoolId.trim() || !joinPubkey.trim()) return
    await controller.joinPool({
      poolId: joinPoolId.trim(),
      poolRootPubkey: joinPubkey.trim(),
      ...(joinName.trim() ? { name: joinName.trim() } : {}),
    })
    setJoinPoolId('')
    setJoinPubkey('')
    setJoinName('')
    setJoinOpen(false)
  }

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat label="Pools" value={pools.length} accent="cool" caption="Configured on this device." />
        <Stat label="Admin" value={adminCount} accent="accent" caption="Pools you administer." />
        <Stat label="Member" value={memberCount} accent="violet" caption="Joined pools." />
        <Stat label="Cached peers" value={totalPeers} accent="warm" caption="Across all pools." />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">LAN pools</div>
            <h3>Membership and peer discovery</h3>
            <p>Pools provide encrypted LAN discovery so devices can find each other without a central directory.</p>
          </div>
          <div className="ms-section-head-actions">
            <MutationButton variant="ghost" size="sm" onClick={() => setJoinOpen(true)}>
              Join pool
            </MutationButton>
            <MutationButton variant="primary" size="sm" onClick={() => setCreateOpen(true)}>
              Create pool
            </MutationButton>
          </div>
        </div>

        {pools.length === 0 ? (
          <EmptyState
            title="No pools configured"
            hint="Create a pool to invite devices on the same LAN, or join an existing one with its credentials."
          />
        ) : (
          <div className="ms-assignments">
            <div className="ms-assignments-head">
              <span>Name</span>
              <span>Role</span>
              <span className="num">Peers</span>
              <span>Cert</span>
              <span>Root key</span>
            </div>
            {pools.map((pool) => (
              <button
                key={pool.id}
                type="button"
                className="ms-assignments-row"
                style={{ background: 'transparent', border: 'none', textAlign: 'left', width: '100%' }}
                onClick={() => controller.setSelectedPoolId(pool.id)}
              >
                <span>
                  <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                    {pool.name}
                  </strong>
                  <small className="mono" style={{ color: chartColors.textMuted, fontSize: 10.5 }}>
                    {pool.id.slice(0, 14)}
                  </small>
                </span>
                <span>
                  <StatusBadge status={pool.role === 'admin' ? 'ok' : 'info'} dot={false}>
                    {pool.role}
                  </StatusBadge>
                </span>
                <span className="num">{formatInteger(pool.peerCount)}</span>
                <span>
                  <StatusBadge status={pool.validCert ? 'ok' : 'fail'} dot={false}>
                    {pool.validCert ? 'valid' : 'invalid'}
                  </StatusBadge>
                </span>
                <span className="mono" style={{ fontSize: 11 }}>
                  {pool.rootPubkeyHex.slice(0, 16)}…
                </span>
              </button>
            ))}
          </div>
        )}
      </section>

      {controller.selectedPool ? (
        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Selected pool</div>
              <h3>{controller.selectedPool.name}</h3>
              <p>
                {formatInteger(controller.selectedPool.peerCount)} cached peer
                {controller.selectedPool.peerCount === 1 ? '' : 's'} ·{' '}
                {controller.selectedPool.daysUntilExpiry != null
                  ? `expires in ${controller.selectedPool.daysUntilExpiry} day(s)`
                  : 'no expiry recorded'}
              </p>
            </div>
          </div>
          <div className="ms-drawer-lifecycle">
            <div className="cell">
              <span>Pool ID</span>
              <strong className="mono" style={{ fontSize: 12 }}>
                {controller.selectedPool.id.slice(0, 18)}
              </strong>
            </div>
            <div className="cell">
              <span>Role</span>
              <strong>
                <StatusBadge status={controller.selectedPool.role === 'admin' ? 'ok' : 'info'} dot={false}>
                  {controller.selectedPool.role}
                </StatusBadge>
              </strong>
            </div>
            <div className="cell">
              <span>Created</span>
              <strong style={{ fontSize: 12 }}>{controller.selectedPool.createdAt}</strong>
            </div>
            <div className="cell">
              <span>Cert</span>
              <strong>
                <StatusBadge status={controller.selectedPool.validCert ? 'ok' : 'fail'}>
                  {controller.selectedPool.validCert ? 'valid' : 'invalid'}
                </StatusBadge>
              </strong>
            </div>
          </div>
        </section>
      ) : null}

      <Modal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        eyebrow="Create"
        title="Create a new pool"
        footer={
          <>
            <MutationButton variant="ghost" size="sm" onClick={() => setCreateOpen(false)}>
              Cancel
            </MutationButton>
            <MutationButton
              variant="primary"
              state={createState}
              disabled={!createName.trim()}
              onClick={() => void handleCreate()}
              pendingLabel="Creating…"
            >
              Create pool
            </MutationButton>
          </>
        }
      >
        <p style={{ color: chartColors.text, fontSize: 12.5 }}>
          A new root keypair will be generated and stored locally. You'll be the admin of this pool.
        </p>
        <label className="ms-field">
          <span>Pool name</span>
          <input
            value={createName}
            onChange={(event) => setCreateName(event.target.value)}
            placeholder="e.g. office-mesh"
            autoFocus
          />
        </label>
      </Modal>

      <Modal
        open={joinOpen}
        onClose={() => setJoinOpen(false)}
        eyebrow="Join"
        title="Join an existing pool"
        footer={
          <>
            <MutationButton variant="ghost" size="sm" onClick={() => setJoinOpen(false)}>
              Cancel
            </MutationButton>
            <MutationButton
              variant="primary"
              state={joinState}
              disabled={!joinPoolId.trim() || !joinPubkey.trim()}
              onClick={() => void handleJoin()}
              pendingLabel="Joining…"
            >
              Join pool
            </MutationButton>
          </>
        }
      >
        <p style={{ color: chartColors.text, fontSize: 12.5 }}>
          Paste the credentials shared by the pool admin. The pool root pubkey verifies that all members
          are speaking to the right network.
        </p>
        <label className="ms-field">
          <span>Pool ID (hex)</span>
          <input value={joinPoolId} onChange={(event) => setJoinPoolId(event.target.value)} placeholder="hex pool id" />
        </label>
        <label className="ms-field">
          <span>Pool root pubkey (hex)</span>
          <input value={joinPubkey} onChange={(event) => setJoinPubkey(event.target.value)} placeholder="hex pubkey" />
        </label>
        <label className="ms-field">
          <span>Local alias (optional)</span>
          <input value={joinName} onChange={(event) => setJoinName(event.target.value)} placeholder="optional alias" />
        </label>
      </Modal>
    </div>
  )
}
