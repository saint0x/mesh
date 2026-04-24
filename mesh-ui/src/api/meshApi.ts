import { apiRequest } from './dashboardApi'
import type { DeviceStatus, DoctorReport, PoolPeer, PoolSummary, QuoteResponse, ResourceLockStatus } from '../domain/dashboard'

export function getDeviceStatus() {
  return apiRequest<DeviceStatus>('/api/local/device/status')
}

export function startDevice(logLevel = 'info') {
  return apiRequest<{ running: boolean }>('/api/local/device/start', {
    method: 'POST',
    body: JSON.stringify({ logLevel }),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function stopDevice() {
  return apiRequest<{ running: boolean }>('/api/local/device/stop', {
    method: 'POST',
    body: JSON.stringify({}),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function lockResources(memory: string) {
  return apiRequest<{ locked: boolean }>('/api/local/resource/lock', {
    method: 'POST',
    body: JSON.stringify({ memory }),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function unlockResources() {
  return apiRequest<{ locked: boolean }>('/api/local/resource/unlock', {
    method: 'POST',
    body: JSON.stringify({}),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function getResourceStatus() {
  return apiRequest<ResourceLockStatus>('/api/local/resource/status')
}

export function joinRing(modelId: string, memory?: string) {
  return apiRequest<{ joined: boolean }>('/api/local/ring/join', {
    method: 'POST',
    body: JSON.stringify({ modelId, memory }),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function leaveRing() {
  return apiRequest<{ joined: boolean }>('/api/local/ring/leave', {
    method: 'POST',
    body: JSON.stringify({}),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function runJob(input: {
  prompt: string
  modelId: string
  maxTokens: number
  temperature: number
  topP: number
}) {
  return apiRequest<{ success: boolean; job_id?: string; jobId?: string }>('/api/local/jobs', {
    method: 'POST',
    body: JSON.stringify(input),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function cancelJob(jobId: string) {
  return apiRequest<{ success: boolean; status: string }>(`/api/local/jobs/${jobId}`, {
    method: 'DELETE',
  })
}

export function runDoctor() {
  return apiRequest<DoctorReport>('/api/local/doctor', {
    method: 'POST',
    body: JSON.stringify({}),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function createPool(name: string) {
  return apiRequest<PoolSummary[]>('/api/local/pools', {
    method: 'POST',
    body: JSON.stringify({ name }),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function joinPool(input: { poolId: string; poolRootPubkey: string; name?: string }) {
  return apiRequest<PoolSummary[]>('/api/local/pools/join', {
    method: 'POST',
    body: JSON.stringify(input),
    headers: { 'Content-Type': 'application/json' },
  })
}

export function getPoolPeers(poolId: string) {
  return apiRequest<PoolPeer[]>(`/api/local/pools/${poolId}/peers`)
}

export function getModelQuote(modelId: string, promptTokens: number, maxTokens: number, networkId?: string) {
  const search = new URLSearchParams({
    promptTokens: String(promptTokens),
    maxTokens: String(maxTokens),
  })
  if (networkId) search.set('networkId', networkId)
  return apiRequest<QuoteResponse>(`/api/local/models/${modelId}/quote`, undefined, search)
}
