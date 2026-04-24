import { Stat, StatRow, StatusBadge } from '../primitives'
import { chartColors } from '../charts'
import type { DashboardPageProps, DashboardShellProps } from '../lib/pageProps'

interface SettingsPageProps extends DashboardPageProps {
  shellProps?: Pick<DashboardShellProps, 'theme' | 'onToggleTheme'>
}

function renderJsonBlock(value: unknown): string {
  if (value == null) return '—'
  return JSON.stringify(value, null, 2)
}

export function SettingsPage({ controller }: SettingsPageProps) {
  const settings = controller.state.settings

  return (
    <div className="dashboard-stack">
      <StatRow>
        <Stat
          label="Local device"
          value={settings.localDeviceName ?? '—'}
          accent="cool"
          caption="Name set during mesh device init."
        />
        <Stat
          label="Control plane"
          value={settings.controlPlaneUrl ?? '—'}
          accent="accent"
          caption="HTTP endpoint used by the agent."
        />
        <Stat
          label="Preferred provider"
          value={
            <StatusBadge status="info" dot={false}>
              {settings.preferredProvider ?? 'auto'}
            </StatusBadge>
          }
          accent="violet"
        />
        <Stat
          label="Snapshot generated"
          value={controller.state.generatedAt || '—'}
          accent="warm"
          caption={`Mesh home: ${controller.state.meshHome || '—'}`}
        />
      </StatRow>

      <section className="panel dashboard-panel">
        <div className="ms-section-head">
          <div className="ms-section-head-copy">
            <div className="eyebrow">Config paths</div>
            <h3>Local files this Mesh UI reads</h3>
            <p>All paths resolve relative to ~/.meshnet on this machine.</p>
          </div>
        </div>
        <div className="ms-assignments">
          <div className="ms-assignments-head">
            <span>Purpose</span>
            <span>Path</span>
            <span></span>
            <span></span>
            <span></span>
          </div>
          {[
            { label: 'Device config', path: settings.configPaths.deviceConfig, hint: 'TOML for the local agent identity.' },
            { label: 'Device certificate', path: settings.configPaths.deviceCertificate, hint: 'Ed25519 cert binding to control plane.' },
            { label: 'Relay config', path: settings.configPaths.relayConfig, hint: 'Local relay-server configuration if present.' },
            { label: 'Control-plane DB', path: settings.configPaths.controlPlaneDb, hint: 'Authoritative SQLite snapshot.' },
            { label: 'Shard registry', path: settings.configPaths.shardRegistry, hint: 'Locally loaded model shard manifest.' },
          ].map((row) => (
            <div
              key={row.label}
              className="ms-assignments-row"
              style={{ gridTemplateColumns: 'minmax(0, 0.9fr) minmax(0, 2.4fr) 0 0 0' }}
            >
              <span>
                <strong style={{ color: chartColors.textStrong, display: 'block', fontSize: 12 }}>
                  {row.label}
                </strong>
                <small style={{ color: chartColors.textMuted, fontSize: 10.5 }}>{row.hint}</small>
              </span>
              <span className="mono" style={{ fontSize: 11, wordBreak: 'break-all' }}>
                {row.path || '—'}
              </span>
              <span></span>
              <span></span>
              <span></span>
            </div>
          ))}
        </div>
      </section>

      <div className="ms-credit-grid">
        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Governance</div>
              <h3>Local runtime limits</h3>
            </div>
          </div>
          <pre
            style={{
              margin: 0,
              padding: 14,
              borderRadius: 10,
              border: '1px solid var(--line)',
              background: 'rgba(255,255,255,0.02)',
              fontFamily: 'JetBrains Mono, ui-monospace, Menlo, monospace',
              fontSize: 11,
              color: chartColors.text,
              maxHeight: 320,
              overflow: 'auto',
            }}
          >
            {renderJsonBlock(settings.governance)}
          </pre>
        </section>

        <section className="panel dashboard-panel">
          <div className="ms-section-head">
            <div className="ms-section-head-copy">
              <div className="eyebrow">Relay</div>
              <h3>Local relay configuration</h3>
            </div>
          </div>
          <pre
            style={{
              margin: 0,
              padding: 14,
              borderRadius: 10,
              border: '1px solid var(--line)',
              background: 'rgba(255,255,255,0.02)',
              fontFamily: 'JetBrains Mono, ui-monospace, Menlo, monospace',
              fontSize: 11,
              color: chartColors.text,
              maxHeight: 320,
              overflow: 'auto',
            }}
          >
            {renderJsonBlock(settings.relay)}
          </pre>
        </section>
      </div>
    </div>
  )
}
