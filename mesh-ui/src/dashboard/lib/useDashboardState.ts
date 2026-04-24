import { useDeferredValue, useEffect, useMemo, useState } from 'react'
import { loadDashboardState } from '../../api/dashboardApi'
import {
  cancelJob,
  createPool,
  getDeviceStatus,
  getModelQuote,
  joinPool,
  joinRing,
  leaveRing,
  lockResources,
  runDoctor,
  runJob,
  startDevice,
  stopDevice,
  unlockResources,
} from '../../api/meshApi'
import type {
  DashboardState,
  DeviceRecord,
  DeviceStatus,
  DoctorReport,
  JobRecord,
  ModelRecord,
  MutationState,
  NetworkRecord,
  OverviewTab,
  PoolSummary,
  QuoteResponse,
  ResourceLockStatus,
  TimeRange,
  TopologyRecord,
} from '../../domain/dashboard'

const emptyDashboardState: DashboardState = {
  generatedAt: '',
  meshHome: '',
  localDeviceId: null,
  networks: [],
  devices: [],
  models: [],
  jobs: [],
  ledgerEvents: [],
  topologies: [],
  runtimeStats: null,
  resourceLock: null,
  pools: [],
  doctor: null,
  settings: {
    controlPlaneUrl: null,
    localDeviceName: null,
    preferredProvider: null,
    governance: null,
    relay: null,
    configPaths: {
      deviceConfig: '',
      deviceCertificate: '',
      relayConfig: '',
      controlPlaneDb: '',
      shardRegistry: '',
    },
  },
}

type MutationMap = Record<string, { state: MutationState; message?: string }>

const defaultIncludes = ['runtimeStats', 'resourceLock', 'pools']

export interface DashboardController {
  expandedGroups: Record<string, boolean>
  state: DashboardState
  isLoading: boolean
  error: string | null
  deviceStatus: DeviceStatus | null
  doctorReport: DoctorReport | null
  resourceLock: ResourceLockStatus | null
  selectedNetwork: NetworkRecord | undefined
  selectedDevice: DeviceRecord | undefined
  selectedJob: JobRecord | undefined
  selectedModel: ModelRecord | undefined
  selectedPool: PoolSummary | undefined
  selectedTopology: TopologyRecord | undefined
  networkDevices: DeviceRecord[]
  networkModels: ModelRecord[]
  networkJobs: JobRecord[]
  networkLedger: DashboardState['ledgerEvents']
  summary: {
    healthyDevices: number
    activeModels: number
    runningJobs: number
    netCredits: number
  }
  healthScore: number
  jobSearch: string
  overviewTab: OverviewTab
  timeRange: TimeRange
  mutationState: MutationMap
  jobDraft: {
    prompt: string
    modelId: string
    maxTokens: number
    temperature: number
    topP: number
  }
  quote: QuoteResponse | null
  setJobSearch: (value: string) => void
  setOverviewTab: (tab: OverviewTab) => void
  cycleTimeRange: () => void
  setSelectedNetworkId: (id: string) => void
  setSelectedDeviceId: (id: string) => void
  setSelectedJobId: (id: string) => void
  setSelectedModelId: (id: string) => void
  setSelectedPoolId: (id: string) => void
  setJobDraft: (draft: Partial<DashboardController['jobDraft']>) => void
  toggleGroup: (groupId: string) => void
  exportState: () => void
  refresh: (include?: string[]) => Promise<void>
  refreshDoctor: () => Promise<void>
  startDevice: () => Promise<void>
  stopDevice: () => Promise<void>
  lockResources: (memory: string) => Promise<void>
  unlockResources: () => Promise<void>
  joinRing: (modelId: string, memory?: string) => Promise<void>
  leaveRing: () => Promise<void>
  runJob: () => Promise<void>
  cancelJob: (jobId: string) => Promise<void>
  createPool: (name: string) => Promise<void>
  joinPool: (input: { poolId: string; poolRootPubkey: string; name?: string }) => Promise<void>
}

export function useDashboardState(): DashboardController {
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({
    workspace: true,
    infrastructure: true,
    catalog: true,
  })
  const [state, setState] = useState(emptyDashboardState)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNetworkId, setSelectedNetworkIdState] = useState('')
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [selectedJobId, setSelectedJobId] = useState('')
  const [selectedModelId, setSelectedModelId] = useState('')
  const [selectedPoolId, setSelectedPoolId] = useState('')
  const [jobSearch, setJobSearch] = useState('')
  const [overviewTab, setOverviewTab] = useState<OverviewTab>('overview')
  const [timeRange, setTimeRange] = useState<TimeRange>('7d')
  const [mutationState, setMutationState] = useState<MutationMap>({})
  const [deviceStatus, setDeviceStatus] = useState<DeviceStatus | null>(null)
  const [doctorReport, setDoctorReport] = useState<DoctorReport | null>(null)
  const [quote, setQuote] = useState<QuoteResponse | null>(null)
  const [jobDraft, setJobDraftState] = useState({
    prompt: '',
    modelId: '',
    maxTokens: 128,
    temperature: 1,
    topP: 0.9,
  })
  const deferredJobSearch = useDeferredValue(jobSearch)

  const setMutation = (key: string, state: MutationState, message?: string) => {
    setMutationState((current) => ({
      ...current,
      [key]: message === undefined ? { state } : { state, message },
    }))
  }

  const syncLocalStatus = async () => {
    try {
      const nextStatus = await getDeviceStatus()
      setDeviceStatus(nextStatus)
      if (state.doctor) setDoctorReport(state.doctor)
    } catch {
      setDeviceStatus(null)
    }
  }

  const refresh = async (include: string[] = defaultIncludes) => {
    setIsLoading(true)
    setError(null)
    try {
      const nextState = await loadDashboardState(include)
      setState(nextState)
      setDoctorReport(nextState.doctor ?? null)
      const nextNetworkId = selectedNetworkId || nextState.networks[0]?.id || ''
      setSelectedNetworkIdState(nextNetworkId)
      setSelectedDeviceId((current) => current || nextState.devices.find((device) => device.networkId === nextNetworkId)?.id || '')
      setSelectedJobId((current) => current || nextState.jobs.find((job) => job.networkId === nextNetworkId)?.id || '')
      setSelectedModelId((current) => current || nextState.models.find((model) => model.networkIds.includes(nextNetworkId))?.id || '')
      setSelectedPoolId((current) => current || nextState.pools[0]?.id || '')
      setJobDraftState((current) => ({
        ...current,
        modelId: current.modelId || nextState.models.find((model) => model.networkIds.includes(nextNetworkId))?.id || '',
      }))
      await syncLocalStatus()
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load dashboard snapshot')
      setState(emptyDashboardState)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
    // Initial load only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const hasActiveJob = state.jobs.some((job) => ['dispatched', 'running', 'acknowledged'].includes(job.status))
    const intervalMs = hasActiveJob ? 2_000 : 10_000
    const interval = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh()
      }
    }, intervalMs)
    return () => window.clearInterval(interval)
    // Poll cadence is driven by live job state; refresh intentionally stays out of deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.jobs])

  useEffect(() => {
    const estimatePromptTokens = Math.max(1, Math.ceil(jobDraft.prompt.trim().length / 4))
    if (!jobDraft.modelId) {
      setQuote(null)
      return
    }
    const timeout = window.setTimeout(() => {
      void getModelQuote(jobDraft.modelId, estimatePromptTokens, jobDraft.maxTokens, selectedNetworkId || undefined)
        .then(setQuote)
        .catch(() => setQuote(null))
    }, 250)
    return () => window.clearTimeout(timeout)
  }, [jobDraft.maxTokens, jobDraft.modelId, jobDraft.prompt, selectedNetworkId])

  const selectedNetwork = useMemo(
    () => state.networks.find((network) => network.id === selectedNetworkId) ?? state.networks[0],
    [selectedNetworkId, state.networks],
  )

  const networkDevices = useMemo(
    () => state.devices.filter((device) => device.networkId === selectedNetwork?.id),
    [selectedNetwork?.id, state.devices],
  )

  const networkModels = useMemo(
    () => state.models.filter((model) => (selectedNetwork ? model.networkIds.includes(selectedNetwork.id) : false)),
    [selectedNetwork, state.models],
  )

  const networkJobs = useMemo(() => {
    const search = deferredJobSearch.trim().toLowerCase()
    return state.jobs.filter((job) => {
      const matchesNetwork = job.networkId === selectedNetwork?.id
      if (!matchesNetwork) return false
      if (!search) return true
      return `${job.id} ${job.status} ${job.submittedByName} ${job.modelId}`.toLowerCase().includes(search)
    })
  }, [deferredJobSearch, selectedNetwork?.id, state.jobs])

  const networkLedger = useMemo(
    () => state.ledgerEvents.filter((event) => event.networkId === selectedNetwork?.id),
    [selectedNetwork?.id, state.ledgerEvents],
  )

  const selectedTopology = useMemo(
    () => state.topologies.find((topology) => topology.networkId === selectedNetwork?.id),
    [selectedNetwork?.id, state.topologies],
  )

  const selectedDevice = useMemo(
    () => networkDevices.find((device) => device.id === selectedDeviceId) ?? networkDevices[0],
    [networkDevices, selectedDeviceId],
  )

  const selectedJob = useMemo(
    () => networkJobs.find((job) => job.id === selectedJobId) ?? networkJobs[0],
    [networkJobs, selectedJobId],
  )

  const selectedModel = useMemo(
    () => networkModels.find((model) => model.id === selectedModelId) ?? networkModels[0],
    [networkModels, selectedModelId],
  )

  const selectedPool = useMemo(
    () => state.pools.find((pool) => pool.id === selectedPoolId) ?? state.pools[0],
    [selectedPoolId, state.pools],
  )

  const summary = useMemo(() => {
    const healthyDevices = networkDevices.filter((device) => device.health === 'healthy').length
    const activeModels = networkModels.filter((model) => model.artifactReady).length
    const runningJobs = networkJobs.filter((job) => ['running', 'acknowledged', 'dispatched'].includes(job.status)).length
    const netCredits = networkLedger.reduce((total, event) => total + (event.creditsAmount ?? 0), 0)
    return { healthyDevices, activeModels, runningJobs, netCredits }
  }, [networkDevices, networkJobs, networkLedger, networkModels])

  const healthScore = useMemo(() => {
    const denominator = Math.max(networkDevices.length + networkModels.length + Math.max(networkJobs.length, 1), 1)
    return Math.round(((summary.healthyDevices + summary.activeModels + Math.max(summary.runningJobs, 1)) / denominator) * 100)
  }, [networkDevices.length, networkJobs.length, networkModels.length, summary])

  const setSelectedNetworkId = (id: string) => {
    setSelectedNetworkIdState(id)
    setSelectedDeviceId(state.devices.find((device) => device.networkId === id)?.id || '')
    setSelectedJobId(state.jobs.find((job) => job.networkId === id)?.id || '')
    setSelectedModelId(state.models.find((model) => model.networkIds.includes(id))?.id || '')
  }

  const toggleGroup = (groupId: string) => {
    setExpandedGroups((current) => ({ ...current, [groupId]: !current[groupId] }))
  }

  const exportState = () => {
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `mesh-ui-dashboard-${selectedNetwork?.id ?? 'state'}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const cycleTimeRange = () => {
    setTimeRange((current) => (current === '24h' ? '7d' : current === '7d' ? '30d' : '24h'))
  }

  const runMutation = async (key: string, action: () => Promise<unknown>, include?: string[]) => {
    setMutation(key, 'pending')
    try {
      await action()
      setMutation(key, 'success')
      await refresh(include)
    } catch (mutationError) {
      setMutation(key, 'error', mutationError instanceof Error ? mutationError.message : 'Action failed')
    }
  }

  return {
    expandedGroups,
    state,
    isLoading,
    error,
    deviceStatus,
    doctorReport,
    resourceLock: state.resourceLock ?? null,
    selectedNetwork,
    selectedDevice,
    selectedJob,
    selectedModel,
    selectedPool,
    selectedTopology,
    networkDevices,
    networkModels,
    networkJobs,
    networkLedger,
    summary,
    healthScore,
    jobSearch,
    overviewTab,
    timeRange,
    mutationState,
    jobDraft,
    quote,
    setJobSearch,
    setOverviewTab,
    cycleTimeRange,
    setSelectedNetworkId,
    setSelectedDeviceId,
    setSelectedJobId,
    setSelectedModelId,
    setSelectedPoolId,
    setJobDraft: (draft) => setJobDraftState((current) => ({ ...current, ...draft })),
    toggleGroup,
    exportState,
    refresh,
    refreshDoctor: async () => {
      await runMutation('doctor', async () => {
        const report = await runDoctor()
        setDoctorReport(report)
      }, [...defaultIncludes, 'doctor'])
    },
    startDevice: async () => {
      await runMutation('device:start', async () => {
        await startDevice()
      })
      await syncLocalStatus()
    },
    stopDevice: async () => {
      await runMutation('device:stop', async () => {
        await stopDevice()
      })
      await syncLocalStatus()
    },
    lockResources: async (memory: string) => {
      await runMutation('resource:lock', async () => {
        await lockResources(memory)
      })
    },
    unlockResources: async () => {
      await runMutation('resource:unlock', async () => {
        await unlockResources()
      })
    },
    joinRing: async (modelId: string, memory?: string) => {
      await runMutation('ring:join', async () => {
        await joinRing(modelId, memory)
      })
    },
    leaveRing: async () => {
      await runMutation('ring:leave', async () => {
        await leaveRing()
      })
    },
    runJob: async () => {
      await runMutation('job:run', async () => {
        const response = await runJob(jobDraft)
        const createdJobId = response.jobId ?? response.job_id
        if (createdJobId) setSelectedJobId(createdJobId)
      })
    },
    cancelJob: async (jobId: string) => {
      await runMutation(`job:cancel:${jobId}`, async () => {
        await cancelJob(jobId)
      })
    },
    createPool: async (name: string) => {
      await runMutation('pool:create', async () => {
        await createPool(name)
      })
    },
    joinPool: async (input) => {
      await runMutation('pool:join', async () => {
        await joinPool(input)
      })
    },
  }
}
