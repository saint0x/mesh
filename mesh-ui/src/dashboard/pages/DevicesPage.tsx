import { EmptyState, MutationButton, Stat, StatRow, StatusBadge } from '../primitives'
import { ProgressBar, chartColors } from '../charts'
import { formatBytes } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function DevicesPage({ controller }: DashboardPageProps) {
  const selectedDevice = controller.selectedDevice
  const localDevice = controller.networkDevices.find((device) => device.localDevice) ?? selectedDevice
  const deviceStatus = controller.deviceStatus
  const lock = controller.resourceLock

  const startState = controller.mutationState['device:start']?.state ?? 'idle'
  const stopState = controller.mutationState['device:stop']?.state ?? 'idle'
  const lockState = controller.mutationState['resource:lock']?.state ?? 'idle'
  const unlockState = controller.mutationState['resource:unlock']?.state ?? 'idle'
  const joinState = controller.mutationState['ring:join']?.state ?? 'idle'
  const leaveState = controller.mutationState['ring:leave']?.state ?? 'idle'

  const totalDevices = controller.networkDevices.length
  const healthyDevices = controller.networkDevices.filter((device) => device.health === 'healthy').length
  const ringMembers = controller.networkDevices.filter((device) => device.ringPosition != null).length

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Devices"
          value={totalDevices}
          accent="cool"
          caption={`${healthyDevices} healthy / ${totalDevices - healthyDevices} other`}
        />
        <Stat
          label="Ring members"
          value={ringMembers}
          accent="accent"
          caption="Devices currently holding shard assignments."
        />
        <Stat
          label="Daemon"
          value={
            <StatusBadge
              status={deviceStatus?.daemonRunning ? 'ok' : 'warn'}
              size="md"
              dot
            >
              {deviceStatus?.daemonRunning ? 'running' : 'stopped'}
            </StatusBadge>
          }
          accent={deviceStatus?.daemonRunning ? 'accent' : 'warm'}
          caption={
            deviceStatus?.deviceId ? `Device ${deviceStatus.deviceId.slice(0, 12)}…` : 'No local device configured.'
          }
        />
        <Stat
          label="Resource lock"
          value={lock ? formatBytes(lock.lockedMemoryBytes) : '—'}
          accent={lock?.status === 'locked' ? 'warm' : 'neutral'}
          caption={lock ? `Status: ${lock.status}` : 'No memory committed.'}
        />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Local operator</div>
            <h3>{deviceStatus?.name ?? localDevice?.name ?? 'This device'}</h3>
            <p>Daemon, resource lock, and ring membership control for this machine.</p>
          </div>
          <div className="ms-section-head-actions">
            <MutationButton
              variant={deviceStatus?.daemonRunning ? 'ghost' : 'primary'}
              size="sm"
              state={startState}
              onClick={() => void controller.startDevice()}
              pendingLabel="Starting…"
            >
              {deviceStatus?.daemonRunning ? 'Restart daemon' : 'Start daemon'}
            </MutationButton>
            {deviceStatus?.daemonRunning ? (
              <MutationButton
                variant="ghost"
                size="sm"
                state={stopState}
                onClick={() => void controller.stopDevice()}
                pendingLabel="Stopping…"
              >
                Stop daemon
              </MutationButton>
            ) : null}
          </div>
        </div>

        <div className="ms-drawer-lifecycle">
          <div className="cell">
            <span>Device ID</span>
            <strong className="mono" style={{ fontSize: 12 }}>
              {(deviceStatus?.deviceId ?? localDevice?.id ?? '—').slice(0, 14)}
            </strong>
          </div>
          <div className="cell">
            <span>Control plane</span>
            <strong style={{ fontSize: 12 }}>
              {deviceStatus?.controlPlaneUrl ?? controller.state.settings.controlPlaneUrl ?? '—'}
            </strong>
          </div>
          <div className="cell">
            <span>Certificate</span>
            <strong>
              <StatusBadge status={deviceStatus?.hasCertificate ? 'ok' : 'fail'} dot={false}>
                {deviceStatus?.hasCertificate ? 'present' : 'missing'}
              </StatusBadge>
            </strong>
          </div>
          <div className="cell">
            <span>Direct candidates</span>
            <strong>{deviceStatus?.directCandidateCount ?? 0}</strong>
          </div>
        </div>

        <div className="ms-section-head" style={{ marginTop: 18 }}>
          <div className="ms-section-head-copy">
            <h4 style={{ margin: 0, fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', color: chartColors.accent }}>
              Resource &amp; ring controls
            </h4>
          </div>
        </div>
        <div className="ms-section-head-actions" style={{ marginBottom: 12 }}>
          <MutationButton
            variant="ghost"
            size="sm"
            state={lockState}
            onClick={() => void controller.lockResources('8GB')}
            pendingLabel="Locking…"
          >
            Lock 8GB
          </MutationButton>
          <MutationButton
            variant="ghost"
            size="sm"
            state={unlockState}
            onClick={() => void controller.unlockResources()}
            pendingLabel="Unlocking…"
          >
            Unlock
          </MutationButton>
          <MutationButton
            variant="ghost"
            size="sm"
            state={joinState}
            disabled={!controller.selectedModel?.id}
            onClick={() => void controller.joinRing(controller.selectedModel?.id ?? '')}
            pendingLabel="Joining…"
          >
            Join {controller.selectedModel?.id ? controller.selectedModel.id.slice(0, 16) : 'ring'}
          </MutationButton>
          <MutationButton
            variant="ghost"
            size="sm"
            state={leaveState}
            onClick={() => void controller.leaveRing()}
            pendingLabel="Leaving…"
          >
            Leave ring
          </MutationButton>
        </div>
        {lock ? (
          <ProgressBar
            current={lock.lockedMemoryBytes}
            total={Math.max(lock.totalMemoryBytes, lock.lockedMemoryBytes, 1)}
            color={lock.status === 'locked' ? chartColors.outstanding : chartColors.settled}
            label={
              <>
                <span>locked memory</span>
                <span>
                  {formatBytes(lock.lockedMemoryBytes)} / {formatBytes(lock.totalMemoryBytes)}
                </span>
              </>
            }
            caption={
              lock.unlockInSeconds != null
                ? `Cooldown ${lock.unlockInSeconds}s — ready to unlock: ${lock.readyToUnlock ? 'yes' : 'no'}`
                : `Status: ${lock.status}`
            }
          />
        ) : null}
      </section>

      <div className="ms-credit-grid">
        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Inventory</div>
              <h3>{controller.selectedNetwork?.name ?? 'Network'} devices</h3>
            </div>
          </div>
          {controller.networkDevices.length === 0 ? (
            <EmptyState title="No devices in this network" />
          ) : (
            <div className="ms-assignments">
              <div className="ms-assignments-head">
                <span>Device</span>
                <span>Health</span>
                <span>Provider</span>
                <span>Tier</span>
                <span>Memory</span>
              </div>
              {controller.networkDevices.map((device) => (
                <button
                  key={device.id}
                  type="button"
                  className="ms-assignments-row"
                  style={{ background: 'transparent', border: 'none', textAlign: 'left', width: '100%', cursor: 'pointer' }}
                  onClick={() => controller.setSelectedDeviceId(device.id)}
                >
                  <span>
                    <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                      {device.name}
                    </strong>
                    <small className="mono" style={{ color: chartColors.textMuted, fontSize: 10 }}>
                      {device.id.slice(0, 14)}
                    </small>
                  </span>
                  <span>
                    <StatusBadge status={device.health} />
                  </span>
                  <span>{device.capabilities.defaultExecutionProvider}</span>
                  <span>{device.capabilities.tier}</span>
                  <span>
                    {device.contributedMemoryBytes ? formatBytes(device.contributedMemoryBytes) : '—'}
                  </span>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="panel dashboard-panel">
          {selectedDevice ? (
            <>
              <div className="ms-section-head">
                <div className="ms-section-head-copy">
                  <div className="eyebrow">Selected device</div>
                  <h3>{selectedDevice.name}</h3>
                  <p className="mono" style={{ fontSize: 11 }}>{selectedDevice.id}</p>
                </div>
              </div>
              <div className="ms-drawer-lifecycle">
                <div className="cell">
                  <span>Peer ID</span>
                  <strong className="mono" style={{ fontSize: 12 }}>
                    {selectedDevice.peerId ? selectedDevice.peerId.slice(0, 14) : '—'}
                  </strong>
                </div>
                <div className="cell">
                  <span>CPU / RAM</span>
                  <strong style={{ fontSize: 13 }}>
                    {selectedDevice.capabilities.cpuCores}c · {Math.round(selectedDevice.capabilities.ramMb / 1024)}gb
                  </strong>
                </div>
                <div className="cell">
                  <span>GPU</span>
                  <strong style={{ fontSize: 13 }}>
                    {selectedDevice.capabilities.gpuPresent
                      ? selectedDevice.capabilities.gpuVramMb
                        ? `${Math.round(selectedDevice.capabilities.gpuVramMb / 1024)}gb VRAM`
                        : 'present'
                      : 'no'}
                  </strong>
                </div>
                <div className="cell">
                  <span>Last seen</span>
                  <strong style={{ fontSize: 12 }}>{selectedDevice.lastSeen ?? '—'}</strong>
                </div>
                <div className="cell">
                  <span>Ring position</span>
                  <strong>{selectedDevice.ringPosition ?? '—'}</strong>
                </div>
                <div className="cell">
                  <span>Shard model</span>
                  <strong style={{ fontSize: 12 }}>{selectedDevice.shardModelId ?? '—'}</strong>
                </div>
                <div className="cell">
                  <span>Shard range</span>
                  <strong>
                    {selectedDevice.shardColumnStart != null
                      ? `${selectedDevice.shardColumnStart}–${selectedDevice.shardColumnEnd ?? '?'}`
                      : '—'}
                  </strong>
                </div>
                <div className="cell">
                  <span>Certificate</span>
                  <strong>
                    <StatusBadge status={selectedDevice.certificateStatus} />
                  </strong>
                </div>
              </div>
            </>
          ) : (
            <EmptyState title="No device selected" hint="Click a device on the left to inspect it." />
          )}
        </section>
      </div>
    </div>
  )
}
