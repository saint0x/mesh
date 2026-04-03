import { startTransition } from 'react'
import { formatSchedulingPolicy, pageMeta, type DashboardSection, type ThemeMode } from '../../domain/dashboard'
import type { DashboardController } from '../lib/useDashboardState'

interface DashboardTopbarProps {
  controller: DashboardController
  currentSection: DashboardSection
  onNavigateSection: (section: DashboardSection) => void
  onToggleTheme: () => void
  theme: ThemeMode
}

export function DashboardTopbar({
  controller,
  currentSection,
  onNavigateSection,
  onToggleTheme,
  theme,
}: DashboardTopbarProps) {
  const currentMeta = pageMeta[currentSection]

  return (
    <>
      <div className="dashboard-utilitybar">
        <div className="dashboard-breadcrumbs">
          <span>Mesh UI</span>
          <span>/</span>
          <strong>Dashboard</strong>
          <span>/</span>
          <span>{controller.selectedNetwork?.name ?? 'No network'}</span>
        </div>
        <div className="dashboard-utility-actions">
          <div className="dashboard-toolbar-search">
            <input
              className="dashboard-search"
              placeholder="Search jobs, devices, models"
              value={controller.jobSearch}
              onChange={(event) => controller.setJobSearch(event.target.value)}
            />
          </div>
          <button className="ghost-button" onClick={onToggleTheme}>
            {theme === 'dark' ? 'Light mode' : 'Dark mode'}
          </button>
          <button className="ghost-button" onClick={controller.exportState}>
            Export JSON
          </button>
          <button className="ghost-button" onClick={() => void controller.refresh()}>
            Refresh
          </button>
          <button className="ghost-button" onClick={() => startTransition(() => onNavigateSection('settings'))}>
            Open settings
          </button>
        </div>
      </div>

      <header className="panel dashboard-header compact">
        <div>
          <div className="eyebrow">Operator console</div>
          <h1>{currentMeta.title}</h1>
          <p className="dashboard-header-copy">{currentMeta.subtitle}</p>
        </div>
        <div className="dashboard-header-meta">
          <article>
            <span>Network</span>
            <strong>{controller.selectedNetwork?.name ?? 'No network'}</strong>
          </article>
          <article>
            <span>Devices</span>
            <strong>{controller.networkDevices.length}</strong>
          </article>
          <article>
            <span>Models</span>
            <strong>{controller.networkModels.length}</strong>
          </article>
        </div>
      </header>

      <section className="dashboard-summary-bar">
        <div>
          <span>Health</span>
          <strong>{controller.healthScore}%</strong>
        </div>
        <div>
          <span>Policy</span>
          <strong>
            {controller.selectedNetwork ? formatSchedulingPolicy(controller.selectedNetwork.schedulingPolicy) : 'n/a'}
          </strong>
        </div>
        <div>
          <span>Generated</span>
          <strong>{controller.state.generatedAt || 'n/a'}</strong>
        </div>
      </section>
    </>
  )
}
