import type { ApiErrorShape, DashboardState } from '../domain/dashboard'

type ApiEnvelope<T> = { ok: true; data: T } | { ok: false; error: ApiErrorShape }

const apiBase =
  typeof import.meta.env['VITE_MESH_UI_API_BASE'] === 'string'
    ? import.meta.env['VITE_MESH_UI_API_BASE']
    : ''

function buildUrl(path: string, search?: URLSearchParams): string {
  const query = search && Array.from(search.keys()).length > 0 ? `?${search.toString()}` : ''
  return `${apiBase}${path}${query}`
}

export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  search?: URLSearchParams,
): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  const response = await fetch(buildUrl(path, search), {
    ...init,
    headers,
  })

  const text = await response.text()
  let payload: ApiEnvelope<T> | undefined
  try {
    payload = JSON.parse(text) as ApiEnvelope<T>
  } catch {
    const snippet = text.slice(0, 100).replace(/\s+/g, ' ').trim()
    throw new Error(
      `API returned non-JSON for ${path}${snippet ? `: ${snippet}` : ''}. If this is the Vite dev server, make sure the local Mesh UI API is running.`,
    )
  }

  if (!response.ok || !payload.ok) {
    const message = 'error' in payload ? payload.error.message : `Request failed with ${response.status}`
    throw new Error(message)
  }

  return payload.data
}

export async function loadDashboardState(include: string[] = []): Promise<DashboardState> {
  const search = new URLSearchParams()
  if (include.length > 0) {
    search.set('include', include.join(','))
  }
  return apiRequest<DashboardState>('/api/local/dashboard', undefined, search)
}
