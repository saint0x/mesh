import type { DashboardPageProps } from '../lib/pageProps'

function renderJsonBlock(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

export function SettingsPage({ controller }: DashboardPageProps) {
  return (
    <div className="dashboard-grid">
      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Local runtime</div>
            <h3>Device and control-plane configuration</h3>
          </div>
        </div>
        <div className="dashboard-detail-grid">
          <article><span>Local device</span><strong>{controller.state.settings.localDeviceName ?? 'n/a'}</strong></article>
          <article><span>Control plane</span><strong>{controller.state.settings.controlPlaneUrl ?? 'n/a'}</strong></article>
          <article><span>Preferred provider</span><strong>{controller.state.settings.preferredProvider ?? 'auto'}</strong></article>
          <article><span>Snapshot generated</span><strong>{controller.state.generatedAt || 'n/a'}</strong></article>
        </div>
        <div className="snapshot-grid">
          <article>
            <span>Device config</span>
            <strong>{controller.state.settings.configPaths.deviceConfig}</strong>
            <small>Local device TOML read by Mesh UI.</small>
          </article>
          <article>
            <span>Device certificate</span>
            <strong>{controller.state.settings.configPaths.deviceCertificate}</strong>
            <small>Local identity certificate path.</small>
          </article>
          <article>
            <span>Relay config</span>
            <strong>{controller.state.settings.configPaths.relayConfig}</strong>
            <small>Relay TOML if present on this machine.</small>
          </article>
          <article>
            <span>Control-plane DB</span>
            <strong>{controller.state.settings.configPaths.controlPlaneDb}</strong>
            <small>Authoritative local SQLite snapshot.</small>
          </article>
        </div>
      </section>

      <section className="panel dashboard-panel">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Governance</div>
            <h3>Local runtime limits</h3>
          </div>
        </div>
        <pre className="dashboard-empty">{renderJsonBlock(controller.state.settings.governance)}</pre>
      </section>

      <section className="panel dashboard-panel dashboard-grid-span-2">
        <div className="dashboard-panel-head">
          <div>
            <div className="eyebrow">Relay</div>
            <h3>Relay service configuration</h3>
          </div>
        </div>
        <pre className="dashboard-empty">{renderJsonBlock(controller.state.settings.relay)}</pre>
      </section>
    </div>
  )
}
