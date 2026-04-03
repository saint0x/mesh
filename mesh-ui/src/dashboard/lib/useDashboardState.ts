import { useDeferredValue, useEffect, useEffectEvent, useMemo, useState } from 'react'
import { loadDashboardState } from '../../api/dashboardApi'
import type {
  DashboardState,
  DeviceRecord,
  JobRecord,
  ModelRecord,
  NetworkRecord,
  OverviewTab,
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

export interface DashboardController {
  expandedGroups: Record<string, boolean>
  state: DashboardState
  isLoading: boolean
  error: string | null
  selectedNetwork: NetworkRecord | undefined
  selectedDevice: DeviceRecord | undefined
  selectedJob: JobRecord | undefined
  selectedModel: ModelRecord | undefined
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
  setJobSearch: (value: string) => void
  setOverviewTab: (tab: OverviewTab) => void
  cycleTimeRange: () => void
  setSelectedNetworkId: (id: string) => void
  setSelectedDeviceId: (id: string) => void
  setSelectedJobId: (id: string) => void
  setSelectedModelId: (id: string) => void
  toggleGroup: (groupId: string) => void
  exportState: () => void
  refresh: () => Promise<void>
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
  const [jobSearch, setJobSearch] = useState('')
  const [overviewTab, setOverviewTab] = useState<OverviewTab>('overview')
  const [timeRange, setTimeRange] = useState<TimeRange>('7d')
  const deferredJobSearch = useDeferredValue(jobSearch)

  const initialize = useEffectEvent(async () => {
    await refresh()
  })

  const refresh = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const nextState = await loadDashboardState()
      setState(nextState)
      const nextNetworkId = selectedNetworkId || nextState.networks[0]?.id || ''
      setSelectedNetworkIdState(nextNetworkId)
      setSelectedDeviceId((current) => current || nextState.devices.find((device) => device.networkId === nextNetworkId)?.id || '')
      setSelectedJobId((current) => current || nextState.jobs.find((job) => job.networkId === nextNetworkId)?.id || '')
      setSelectedModelId((current) => current || nextState.models.find((model) => model.networkIds.includes(nextNetworkId))?.id || '')
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load dashboard snapshot')
      setState(emptyDashboardState)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void initialize()
  }, [])

  const selectedNetwork = useMemo(
    () => state.networks.find((network) => network.id === selectedNetworkId) ?? state.networks[0],
    [selectedNetworkId, state.networks],
  )

  const networkDevices = useMemo(
    () => state.devices.filter((device) => device.networkId === selectedNetwork?.id),
    [selectedNetwork?.id, state.devices],
  )

  const networkModels = useMemo(
    () => state.models.filter((model) => selectedNetwork ? model.networkIds.includes(selectedNetwork.id) : false),
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

  const summary = useMemo(() => {
    const healthyDevices = networkDevices.filter((device) => device.health === 'healthy').length
    const activeModels = networkModels.filter((model) => model.artifactReady).length
    const runningJobs = networkJobs.filter((job) => job.status === 'running' || job.status === 'acknowledged').length
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

  return {
    expandedGroups,
    state,
    isLoading,
    error,
    selectedNetwork,
    selectedDevice,
    selectedJob,
    selectedModel,
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
    setJobSearch,
    setOverviewTab,
    cycleTimeRange,
    setSelectedNetworkId,
    setSelectedDeviceId,
    setSelectedJobId,
    setSelectedModelId,
    toggleGroup,
    exportState,
    refresh,
  }
}
