import type { DeviceRecord } from '../../domain/dashboard'
import { formatBytes } from '../lib/format'
import type { DashboardPageProps } from '../lib/pageProps'

export function DevicesPage({ controller }: DashboardPageProps) {
  const selectedNetwork = controller.selectedNetwork
  const selectedDevice = controller.selectedDevice
  if (!selectedNetwork) return null

  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Device inventory</div>
            <h3>{selectedNetwork.name}</h3>
          </div>
        </div>
        <div className="dashboard-list">
          {controller.networkDevices.map((device: DeviceRecord) => (
            <button
              key={device.id}
              className={device.id === selectedDevice?.id ? 'dashboard-list-item active' : 'dashboard-list-item'}
              onClick={() => controller.setSelectedDeviceId(device.id)}
            >
              <strong>{device.name}</strong>
              <span>{device.capabilities.defaultExecutionProvider} / {device.health} / {device.connectivityState?.activePath ?? device.status}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel dashboard-panel">
        {selectedDevice ? (
          <>
            <div className="dashboard-panel-head">
              <div>
                <div className="eyebrow">Selected device</div>
                <h3>{selectedDevice.name}</h3>
              </div>
            </div>
            <div className="dashboard-detail-grid">
              <article><span>Peer ID</span><strong>{selectedDevice.peerId ?? 'n/a'}</strong></article>
              <article><span>Default provider</span><strong>{selectedDevice.capabilities.defaultExecutionProvider}</strong></article>
              <article><span>Contributed memory</span><strong>{selectedDevice.contributedMemoryBytes ? formatBytes(selectedDevice.contributedMemoryBytes) : 'n/a'}</strong></article>
              <article><span>Last seen</span><strong>{selectedDevice.lastSeen ?? 'n/a'}</strong></article>
              <article><span>Ring position</span><strong>{selectedDevice.ringPosition ?? 'n/a'}</strong></article>
              <article><span>Certificate</span><strong>{selectedDevice.certificateStatus}</strong></article>
            </div>

            <div className="snapshot-grid">
              <article>
                <span>CPU / RAM</span>
                <strong>{selectedDevice.capabilities.cpuCores} cores / {Math.round(selectedDevice.capabilities.ramMb / 1024)} GB</strong>
                <small>{selectedDevice.capabilities.os} / {selectedDevice.capabilities.arch} / {selectedDevice.capabilities.tier}</small>
              </article>
              <article>
                <span>GPU</span>
                <strong>{selectedDevice.capabilities.gpuPresent ? 'present' : 'not detected'}</strong>
                <small>{selectedDevice.capabilities.gpuVramMb ? `${Math.round(selectedDevice.capabilities.gpuVramMb / 1024)} GB VRAM` : 'No VRAM detail reported'}</small>
              </article>
              <article>
                <span>Connectivity</span>
                <strong>{selectedDevice.connectivityState?.status ?? selectedDevice.status}</strong>
                <small>{selectedDevice.connectivityState?.activeEndpoint ?? 'No active endpoint reported'}</small>
              </article>
              <article>
                <span>Shard</span>
                <strong>{selectedDevice.shardModelId ?? 'no active shard'}</strong>
                <small>
                  {selectedDevice.shardColumnStart !== null && selectedDevice.shardColumnStart !== undefined
                    ? `${selectedDevice.shardColumnStart} - ${selectedDevice.shardColumnEnd ?? 'n/a'}`
                    : 'No shard range reported'}
                </small>
              </article>
            </div>

            <div className="dashboard-data-table">
              <div className="dashboard-data-head">
                <span>Execution provider</span>
                <span>Available</span>
                <span>Reason</span>
              </div>
              {selectedDevice.capabilities.executionProviders.map((provider) => (
                <div key={provider.kind} className="dashboard-data-row">
                  <span className="row-primary">{provider.kind}</span>
                  <span>{provider.available ? 'yes' : 'no'}</span>
                  <span>{provider.reason ?? 'available'}</span>
                </div>
              ))}
            </div>

            <div className="dashboard-data-table">
              <div className="dashboard-data-head">
                <span>Tensor/listen endpoint</span>
                <span>Type</span>
              </div>
              {[...selectedDevice.tensorPlaneEndpoints, ...selectedDevice.listenAddrs].map((endpoint) => (
                <div key={endpoint} className="dashboard-data-row">
                  <span className="row-primary">{endpoint}</span>
                  <span>{endpoint.startsWith('dataplane://') ? 'tensor-plane' : 'listen'}</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="dashboard-empty">No device is selected for this network.</div>
        )}
      </section>
    </div>
  )
}
