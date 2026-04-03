import { lazy, Suspense } from 'react'
import { DashboardSidebar } from './dashboard/components/DashboardSidebar'
import { DashboardTopbar } from './dashboard/components/DashboardTopbar'
import type { DashboardShellProps } from './dashboard/lib/pageProps'
import { useDashboardState } from './dashboard/lib/useDashboardState'

const OverviewPage = lazy(async () => import('./dashboard/pages/OverviewPage').then((module) => ({ default: module.OverviewPage })))
const NetworksPage = lazy(async () => import('./dashboard/pages/NetworksPage').then((module) => ({ default: module.NetworksPage })))
const DevicesPage = lazy(async () => import('./dashboard/pages/DevicesPage').then((module) => ({ default: module.DevicesPage })))
const TopologyPage = lazy(async () => import('./dashboard/pages/TopologyPage').then((module) => ({ default: module.TopologyPage })))
const ModelsPage = lazy(async () => import('./dashboard/pages/ModelsPage').then((module) => ({ default: module.ModelsPage })))
const JobsPage = lazy(async () => import('./dashboard/pages/JobsPage').then((module) => ({ default: module.JobsPage })))
const LedgerPage = lazy(async () => import('./dashboard/pages/LedgerPage').then((module) => ({ default: module.LedgerPage })))
const CreditsPage = lazy(async () => import('./dashboard/pages/CreditsPage').then((module) => ({ default: module.CreditsPage })))
const SettingsPage = lazy(async () => import('./dashboard/pages/SettingsPage').then((module) => ({ default: module.SettingsPage })))

export function Dashboard(props: DashboardShellProps) {
  const controller = useDashboardState()

  return (
    <div className="dashboard-shell">
      <DashboardSidebar
        controller={controller}
        currentSection={props.currentSection}
        onExit={props.onExit}
        onNavigateSection={props.onNavigateSection}
        onToggleTheme={props.onToggleTheme}
        theme={props.theme}
      />

      <main className="dashboard-main">
        <DashboardTopbar
          controller={controller}
          currentSection={props.currentSection}
          onNavigateSection={props.onNavigateSection}
          onToggleTheme={props.onToggleTheme}
          theme={props.theme}
        />
        {controller.error ? (
          <section className="panel dashboard-panel">
            <div className="dashboard-empty">
              Failed to load the local mesh snapshot: {controller.error}
            </div>
          </section>
        ) : null}
        <Suspense fallback={<DashboardPageFallback />}>
          {renderPage(props.currentSection, controller, props.onNavigateSection)}
        </Suspense>
      </main>
    </div>
  )
}

function renderPage(
  section: DashboardShellProps['currentSection'],
  controller: ReturnType<typeof useDashboardState>,
  onNavigateSection: DashboardShellProps['onNavigateSection'],
) {
  const pageProps = { controller, onNavigateSection }

  switch (section) {
    case 'overview':
      return <OverviewPage {...pageProps} />
    case 'networks':
      return <NetworksPage {...pageProps} />
    case 'devices':
      return <DevicesPage {...pageProps} />
    case 'topology':
      return <TopologyPage {...pageProps} />
    case 'models':
      return <ModelsPage {...pageProps} />
    case 'jobs':
      return <JobsPage {...pageProps} />
    case 'ledger':
      return <LedgerPage {...pageProps} />
    case 'credits':
      return <CreditsPage {...pageProps} />
    case 'settings':
      return <SettingsPage {...pageProps} />
  }
}

function DashboardPageFallback() {
  return (
    <section className="panel dashboard-panel dashboard-page-fallback">
      <div className="dashboard-panel-head">
        <div>
          <div className="eyebrow">Loading section</div>
          <h3>Preparing dashboard surface</h3>
        </div>
      </div>
      <div className="dashboard-empty">
        Pulling the next control-plane view into the workspace.
      </div>
    </section>
  )
}
