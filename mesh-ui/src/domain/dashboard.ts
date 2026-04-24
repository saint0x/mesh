export type DashboardSection =
  | 'overview'
  | 'networks'
  | 'devices'
  | 'topology'
  | 'models'
  | 'jobs'
  | 'ledger'
  | 'credits'
  | 'pools'
  | 'doctor'
  | 'settings'

export type ThemeMode = 'dark' | 'light'
export type OverviewTab = 'overview' | 'operations' | 'runtime'
export type TimeRange = '24h' | '7d' | '30d'
export type MutationState = 'idle' | 'pending' | 'success' | 'error'

export interface ApiErrorShape {
  code: string
  message: string
  hint?: string | null
}

export interface AttachmentRecord {
  kind: string
  endpoint: string
  priority: number
}

export interface TierCapacityUnits {
  tier0: number
  tier1: number
  tier2: number
  tier3: number
  tier4: number
}

export interface SchedulingPolicy {
  submitterActiveJobSoftCap: number
  modelActiveJobSoftCapDivisor: number
  capacityUnitSoftCapDivisor: number
  tierCapacityUnits: TierCapacityUnits
}

export interface NetworkRecord {
  id: string
  name: string
  owner: string
  createdAt: string
  preferredPath: string
  attachments: AttachmentRecord[]
  schedulingPolicy: SchedulingPolicy
}

export interface ExecutionProviderRecord {
  kind: string
  available: boolean
  reason?: string | null
}

export interface CapabilityRecord {
  tier: string
  cpuCores: number
  ramMb: number
  gpuPresent: boolean
  gpuVramMb?: number | null
  os: string
  arch: string
  executionProviders: ExecutionProviderRecord[]
  defaultExecutionProvider: string
}

export interface ConnectivityStateRecord {
  activePath: string
  activeEndpoint?: string | null
  status: string
}

export interface DirectCandidateRecord {
  endpoint: string
  transport: string
  scope: string
  source: string
  priority: number
  lastUpdatedMs: number
}

export interface DeviceRecord {
  id: string
  networkId: string
  name: string
  peerId?: string | null
  status: string
  health: string
  lastSeen?: string | null
  ringPosition?: number | null
  leftNeighborId?: string | null
  rightNeighborId?: string | null
  shardModelId?: string | null
  shardColumnStart?: number | null
  shardColumnEnd?: number | null
  contributedMemoryBytes?: number | null
  connectivityState?: ConnectivityStateRecord | null
  listenAddrs: string[]
  tensorPlaneEndpoints: string[]
  directCandidates: DirectCandidateRecord[]
  capabilities: CapabilityRecord
  certificateStatus: string
  identityStatus: string
  localDevice: boolean
}

export interface ShardRangeRecord {
  start: number
  end: number
}

export interface ModelRecord {
  id: string
  networkIds: string[]
  totalModelBytes?: number | null
  tensorParallelismDim?: number | null
  artifactReady: boolean
  tokenizerReady: boolean
  manifestCount: number
  weightsCount: number
  participantCount: number
  loadedLocalShard: boolean
  localShardRange?: ShardRangeRecord | null
  localMemoryBytes?: number | null
  shardStatus?: string | null
  providerCompatibility: string[]
}

export interface AssignmentCreditBreakdownRecord {
  deviceId: string
  credits: number
  computeShare: number
  throughputMultiplier: number
  resourcePressureMultiplier: number
  normalizedContributionShare: number
  measuredServiceRate: number
  referenceServiceRate: number
  memoryPressure: number
}

export interface CreditPolicyRecord {
  jobCreditBudget: number
  assignments: AssignmentCreditBreakdownRecord[]
}

export interface AssignmentRecord {
  assignmentId: string
  deviceId: string
  deviceName: string
  ringPosition: number
  status: string
  leaseExpiresAt?: string | null
  assignedAt: string
  acknowledgedAt?: string | null
  completedAt?: string | null
  failureReason?: string | null
  executionTimeMs: number
  shardColumnStart?: number | null
  shardColumnEnd?: number | null
  assignedCapacityUnits: number
  executionProvider?: string | null
  reportedCompletionTokens: number
  creditsEarned?: number | null
  throughputMultiplier?: number | null
  resourcePressureMultiplier?: number | null
  normalizedContributionShare?: number | null
  availableMemoryBytes?: number | null
}

export interface JobRecord {
  id: string
  networkId: string
  modelId: string
  status: string
  submittedByDeviceId: string
  submittedByName: string
  ringWorkerCount: number
  createdAt: string
  startedAt?: string | null
  completedAt?: string | null
  completionTokens: number
  promptTokens?: number | null
  executionTimeMs: number
  reservedCredits: number
  settledCredits: number
  releasedCredits: number
  availableCompletionTokens: number
  modelSizeFactor: number
  accountedCompletionTokens: number
  promptCreditsAccounted: boolean
  error?: string | null
  creditPolicy?: CreditPolicyRecord | null
  assignments: AssignmentRecord[]
}

export interface LedgerRecord {
  id: string
  networkId: string
  eventType: string
  jobId?: string | null
  deviceId?: string | null
  creditsAmount?: number | null
  detail: string
  metadata: Record<string, unknown> | null
  createdAt: string
}

export interface TopologyWorkerRecord {
  deviceId: string
  deviceName: string
  peerId?: string | null
  position?: number | null
  status: string
  contributedMemoryBytes?: number | null
  shardColumnStart?: number | null
  shardColumnEnd?: number | null
  leftNeighborId?: string | null
  rightNeighborId?: string | null
  activePath?: string | null
  activeEndpoint?: string | null
  tensorPlaneEndpoints: string[]
}

export interface PunchPlanRecord {
  sourceDeviceId: string
  targetDeviceId: string
  targetPeerId: string
  reason: string
  strategy: string
  relayRendezvousRequired: boolean
  attemptWindowMs: number
  issuedAtMs: number
  targetCandidates: DirectCandidateRecord[]
}

export interface TopologyRecord {
  networkId: string
  source: string
  ringStable: boolean
  workers: TopologyWorkerRecord[]
  punchPlans: PunchPlanRecord[]
}

export interface ResourceLockStatus {
  status: string
  totalMemoryBytes: number
  userAllocatedBytes: number
  lockedMemoryBytes: number
  lockTimestampMs?: number | null
  readyToUnlock: boolean
  unlockInSeconds?: number | null
}

export interface PoolSummary {
  id: string
  name: string
  role: string
  createdAt: string
  expiresAt: number
  daysUntilExpiry?: number | null
  peerCount: number
  rootPubkeyHex: string
  validCert: boolean
}

export interface PoolPeer {
  nodeId: string
  lanAddr: string
  discoveryMethod: string
  lastSeen: number
}

export interface DoctorCheck {
  id: string
  label: string
  status: 'ok' | 'warn' | 'fail'
  detail: string
  hint?: string | null
  durationMs: number
}

export interface DoctorReport {
  generatedAt: string
  overall: 'ok' | 'warn' | 'fail'
  checks: DoctorCheck[]
}

export interface DeviceStatus {
  configured: boolean
  deviceId?: string | null
  networkId?: string | null
  name?: string | null
  controlPlaneUrl?: string | null
  preferredProvider?: string | null
  hasCertificate: boolean
  daemonRunning: boolean
  listenAddrs: string[]
  observedAddrs: string[]
  directCandidateCount: number
}

export interface QuoteResponse {
  modelId: string
  networkId: string
  modelSizeFactor: number
  promptTokens: number
  maxTokens: number
  promptCredits: number
  completionCreditsCap: number
  totalCreditsCap: number
  availableCompletionTokens: number
  deviceAvailableCredits: number
  feasible: boolean
  reason?: string | null
}

export interface SettingsRecord {
  controlPlaneUrl?: string | null
  localDeviceName?: string | null
  preferredProvider?: string | null
  governance: Record<string, unknown> | null
  relay: Record<string, unknown> | null
  configPaths: {
    deviceConfig: string
    deviceCertificate: string
    relayConfig: string
    controlPlaneDb: string
    shardRegistry: string
  }
}

export interface DashboardState {
  generatedAt: string
  meshHome: string
  localDeviceId?: string | null
  networks: NetworkRecord[]
  devices: DeviceRecord[]
  models: ModelRecord[]
  jobs: JobRecord[]
  ledgerEvents: LedgerRecord[]
  topologies: TopologyRecord[]
  runtimeStats?: Record<string, unknown> | null
  resourceLock?: ResourceLockStatus | null
  pools: PoolSummary[]
  doctor?: DoctorReport | null
  settings: SettingsRecord
}

export type SectionDefinition = {
  id: DashboardSection
  label: string
  summary: string
}

export type SidebarGroup = {
  id: string
  label: string
  items: Array<{ id: DashboardSection; label: string }>
}

export const sections: SectionDefinition[] = [
  { id: 'overview', label: 'Overview', summary: 'Live local control-plane snapshot' },
  { id: 'networks', label: 'Networks', summary: 'Scheduling, relay attachments, and mesh policy' },
  { id: 'devices', label: 'Devices', summary: 'Provider, capability, runtime, and identity truth' },
  { id: 'topology', label: 'Topology', summary: 'Ring workers, neighbors, tensor endpoints, and punch plans' },
  { id: 'models', label: 'Models', summary: 'Artifact readiness, shard manifests, and provider compatibility' },
  { id: 'jobs', label: 'Jobs', summary: 'Reservation lifecycle and assignment-level execution detail' },
  { id: 'ledger', label: 'Ledger', summary: 'Authoritative ledger events and audit trail' },
  { id: 'credits', label: 'Credits', summary: 'Participation accounting and reservation flow' },
  { id: 'pools', label: 'Pools', summary: 'LAN pools, peers, membership, and invitations' },
  { id: 'doctor', label: 'Doctor', summary: 'Local setup and control-plane reachability checks' },
  { id: 'settings', label: 'Settings', summary: 'Local device, relay, and runtime configuration' },
]

export const navGroups: SidebarGroup[] = [
  {
    id: 'workspace',
    label: 'Workspace',
    items: [
      { id: 'overview', label: 'Overview' },
      { id: 'jobs', label: 'Jobs' },
      { id: 'ledger', label: 'Ledger' },
      { id: 'credits', label: 'Credits' },
    ],
  },
  {
    id: 'infrastructure',
    label: 'Infrastructure',
    items: [
      { id: 'networks', label: 'Networks' },
      { id: 'devices', label: 'Devices' },
      { id: 'topology', label: 'Topology' },
      { id: 'pools', label: 'Pools' },
      { id: 'doctor', label: 'Doctor' },
    ],
  },
  {
    id: 'catalog',
    label: 'Catalog',
    items: [
      { id: 'models', label: 'Models' },
      { id: 'settings', label: 'Settings' },
    ],
  },
]

export const pageMeta: Record<DashboardSection, { title: string; subtitle: string }> = {
  overview: {
    title: 'Overview',
    subtitle: 'A local snapshot of real MeshNet state sourced from ~/.meshnet and the control-plane database.',
  },
  networks: {
    title: 'Networks',
    subtitle: 'Inspect scheduling controls, preferred path policy, and relay attachments per network.',
  },
  devices: {
    title: 'Devices',
    subtitle: 'See execution providers, hardware capabilities, ring state, resources, and certificate posture.',
  },
  topology: {
    title: 'Topology',
    subtitle: 'Review ring order, neighbor assignments, tensor endpoints, and live punch plans.',
  },
  models: {
    title: 'Models',
    subtitle: 'Track local artifact readiness, shard manifest coverage, and provider compatibility by model.',
  },
  jobs: {
    title: 'Jobs',
    subtitle: 'Submit work, watch reservation settlement, and inspect assignment-level execution detail.',
  },
  ledger: {
    title: 'Ledger',
    subtitle: 'Read the authoritative append-only ledger summary and recent network events.',
  },
  credits: {
    title: 'Credits',
    subtitle: 'View participation accounting and reserved-to-settled-to-released credit flow.',
  },
  pools: {
    title: 'Pools',
    subtitle: 'Manage local pool membership, peer discovery, and shareable pool credentials.',
  },
  doctor: {
    title: 'Doctor',
    subtitle: 'Run local health checks across device config, certificate posture, and control-plane reachability.',
  },
  settings: {
    title: 'Settings',
    subtitle: 'Review local runtime, control-plane, relay, and config-path settings for this machine.',
  },
}

export const dashboardPathBySection: Record<DashboardSection, string> = {
  overview: '/dashboard',
  networks: '/dashboard/networks',
  devices: '/dashboard/devices',
  topology: '/dashboard/topology',
  models: '/dashboard/models',
  jobs: '/dashboard/jobs',
  ledger: '/dashboard/ledger',
  credits: '/dashboard/credits',
  pools: '/dashboard/pools',
  doctor: '/dashboard/doctor',
  settings: '/dashboard/settings',
}

export function getDashboardSectionFromPath(pathname: string): DashboardSection {
  const normalizedPath = pathname.replace(/\/+$/, '') || '/'
  const sectionEntry = Object.entries(dashboardPathBySection).find(([, path]) => path === normalizedPath)
  return (sectionEntry?.[0] as DashboardSection | undefined) ?? 'overview'
}

export function formatSchedulingPolicy(policy: SchedulingPolicy): string {
  return [
    `submitter ${policy.submitterActiveJobSoftCap}`,
    `model /${policy.modelActiveJobSoftCapDivisor}`,
    `capacity /${policy.capacityUnitSoftCapDivisor}`,
  ].join(' • ')
}
