import type { DashboardSection, ThemeMode } from '../../domain/dashboard'
import type { DashboardController } from './useDashboardState'

export interface DashboardShellProps {
  currentSection: DashboardSection
  onExit: () => void
  onNavigateSection: (section: DashboardSection) => void
  theme: ThemeMode
  onToggleTheme: () => void
}

export interface DashboardPageProps {
  controller: DashboardController
  onNavigateSection: (section: DashboardSection) => void
}
