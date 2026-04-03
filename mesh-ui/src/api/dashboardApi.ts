import type { DashboardState } from '../domain/dashboard'

const apiBase =
  typeof import.meta.env['VITE_MESH_UI_API_BASE'] === 'string'
    ? import.meta.env['VITE_MESH_UI_API_BASE']
    : ''

export async function loadDashboardState(): Promise<DashboardState> {
  const response = await fetch(`${apiBase}/api/local/dashboard`)
  if (!response.ok) {
    throw new Error(`Failed to load dashboard snapshot: ${response.status}`)
  }

  const payload: unknown = await response.json()
  return payload as DashboardState
}
