import { startTransition } from 'react'
import { formatSchedulingPolicy, navGroups, type DashboardSection, type ThemeMode } from '../../domain/dashboard'
import type { DashboardController } from '../lib/useDashboardState'

interface DashboardSidebarProps {
  controller: DashboardController
  currentSection: DashboardSection
  onExit: () => void
  onNavigateSection: (section: DashboardSection) => void
  onToggleTheme: () => void
  theme: ThemeMode
}

export function DashboardSidebar({
  controller,
  currentSection,
  onExit,
  onNavigateSection,
  onToggleTheme,
  theme,
}: DashboardSidebarProps) {
  return (
    <aside className="dashboard-sidebar">
      <button
        className="brand dashboard-brand"
        onClick={() => startTransition(() => onNavigateSection('overview'))}
      >
        <span className="brand-mark" aria-hidden="true">
          <span className="brand-mark-core">M</span>
        </span>
        <span className="dashboard-brand-copy">
          <strong>Mesh UI</strong>
          <small>local operator console</small>
        </span>
      </button>

      <div className="network-picker">
        <span className="sidebar-label">Active network</span>
        <label className="network-picker-field">
          <span className="network-picker-meta">
            <strong>{controller.selectedNetwork?.name ?? 'No network selected'}</strong>
            <small>
              {controller.selectedNetwork ? formatSchedulingPolicy(controller.selectedNetwork.schedulingPolicy) : 'Select a network'}
            </small>
          </span>
          <select
            aria-label="Select active network"
            value={controller.selectedNetwork?.id ?? ''}
            onChange={(event) => controller.setSelectedNetworkId(event.target.value)}
          >
            {controller.state.networks.map((network) => (
              <option key={network.id} value={network.id}>
                {network.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <nav className="dashboard-nav">
        {navGroups.map((group) => (
          <div key={group.id} className="sidebar-group">
            <button className="sidebar-group-toggle" onClick={() => controller.toggleGroup(group.id)}>
              <span>{group.label}</span>
              <small>{controller.expandedGroups[group.id] ? '−' : '+'}</small>
            </button>
            {controller.expandedGroups[group.id] ? (
              <div className="sidebar-subnav">
                {group.items.map((item) => (
                  <button
                    key={item.id}
                    className={currentSection === item.id ? 'sidebar-subitem active' : 'sidebar-subitem'}
                    onClick={() => startTransition(() => onNavigateSection(item.id))}
                  >
                    <span>{item.label}</span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="ghost-button" onClick={onToggleTheme}>
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
        <button className="ghost-button" onClick={onExit}>
          Back to site
        </button>
        <button className="primary-button" onClick={() => void controller.refresh()}>
          Refresh snapshot
        </button>
      </div>
    </aside>
  )
}
