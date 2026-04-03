import { useEffect, useState } from 'react'
import './App.css'
import { Dashboard } from './dashboard'
import {
  dashboardPathBySection,
  getDashboardSectionFromPath,
  type DashboardSection,
  type ThemeMode,
} from './domain/dashboard'

type MarketingSection = 'overview' | 'runtime' | 'control' | 'pricing'
type AppRoute = { kind: 'marketing' } | { kind: 'dashboard'; section: DashboardSection }

type MetricKey = 'throughput' | 'latency' | 'coverage'

type Scenario = {
  id: string
  label: string
  eyebrow: string
  title: string
  description: string
  rollout: string
  tokensPerSecond: string
  latency: string
  efficiency: string
  nodes: Array<{
    id: string
    name: string
    type: string
    load: string
    status: string
    x: string
    y: string
  }>
}

type Feature = {
  id: string
  title: string
  summary: string
  bullets: string[]
}

type Command = {
  id: string
  label: string
  command: string
  description: string
  output: string[]
}

const navigation: Array<{ id: MarketingSection; label: string; anchor: string }> = [
  { id: 'overview', label: 'Overview', anchor: 'overview' },
  { id: 'runtime', label: 'Runtime', anchor: 'runtime' },
  { id: 'control', label: 'Control Plane', anchor: 'control' },
  { id: 'pricing', label: 'Launch', anchor: 'launch' },
]

const metrics: Record<
  MetricKey,
  { label: string; headline: string; detail: string; accent: string }
> = {
  throughput: {
    label: 'Distributed throughput',
    headline: '10.4 tok/s on mixed edge nodes',
    detail: 'Sustained across laptops, workstations, and detached GPU hosts.',
    accent: 'Throughput',
  },
  latency: {
    label: 'First-token latency',
    headline: 'Sub-900ms startup path',
    detail: 'Warm routing, pinned models, and policy-aware prefetch keep the mesh responsive.',
    accent: 'Latency',
  },
  coverage: {
    label: 'Hardware coverage',
    headline: 'CPU, GPU, and NPU in one runtime',
    detail: 'One operator workflow across sovereign edge, air-gapped racks, and cloud spillover.',
    accent: 'Coverage',
  },
}

const scenarios: Scenario[] = [
  {
    id: 'sovereign',
    label: 'Sovereign Edge',
    eyebrow: 'Disconnected autonomy',
    title: 'Run high-reasoning inference where the data already lives',
    description:
      'Coordinate ruggedized CPUs, workstations, and portable GPU nodes without shipping prompts or telemetry to the cloud.',
    rollout: '12 forward sites / policy-locked routing',
    tokensPerSecond: '9.7 tok/s',
    latency: '780 ms',
    efficiency: '62% idle compute reclaimed',
    nodes: [
      { id: 'n1', name: 'Forward GPU', type: 'RTX 6000 Ada', load: '92%', status: 'Serving', x: '18%', y: '26%' },
      { id: 'n2', name: 'Ops Rack', type: 'EPYC CPU', load: '71%', status: 'Embedding', x: '50%', y: '56%' },
      { id: 'n3', name: 'Vehicle NPU', type: 'Edge TPU', load: '48%', status: 'Vision', x: '80%', y: '22%' },
      { id: 'n4', name: 'Analyst Laptop', type: 'M-series', load: '37%', status: 'Caching', x: '72%', y: '78%' },
    ],
  },
  {
    id: 'industrial',
    label: 'Industrial Mesh',
    eyebrow: 'IT/OT intelligence bridge',
    title: 'Fuse plant-floor compute into a resilient inference fabric',
    description:
      'Harvest dead compute from factory PCs and remote sensor gateways to power always-on copilots, anomaly triage, and local automation.',
    rollout: '43 facilities / offline-safe failover',
    tokensPerSecond: '7.2 tok/s',
    latency: '840 ms',
    efficiency: '58% cost reduction',
    nodes: [
      { id: 'n5', name: 'Line A Vision', type: 'Jetson Orin', load: '86%', status: 'Inspecting', x: '20%', y: '24%' },
      { id: 'n6', name: 'SCADA Bridge', type: 'Xeon CPU', load: '54%', status: 'Coordinating', x: '46%', y: '48%' },
      { id: 'n7', name: 'QA Workstation', type: 'A5000 GPU', load: '88%', status: 'Ranking', x: '78%', y: '30%' },
      { id: 'n8', name: 'Remote Gateway', type: 'ARM CPU', load: '42%', status: 'Fallback', x: '68%', y: '76%' },
    ],
  },
  {
    id: 'workspace',
    label: 'Private Workspace',
    eyebrow: 'The private agentic mesh',
    title: 'Turn office machines into a collaborative inference plane',
    description:
      'Large private models run across AI PCs, idle desktops, and shared GPU towers with enterprise routing, governance, and audit trails.',
    rollout: '3 offices / zero data egress',
    tokensPerSecond: '11.1 tok/s',
    latency: '690 ms',
    efficiency: '4.3x lower spend',
    nodes: [
      { id: 'n9', name: 'Design Tower', type: 'Dual GPU', load: '95%', status: 'Primary', x: '17%', y: '31%' },
      { id: 'n10', name: 'Sales AI PC', type: 'NPU', load: '43%', status: 'Drafting', x: '45%', y: '58%' },
      { id: 'n11', name: 'Legal Desktop', type: 'CPU', load: '35%', status: 'Retrieval', x: '82%', y: '25%' },
      { id: 'n12', name: 'Conference Hub', type: 'Mac Studio', load: '57%', status: 'Voice', x: '74%', y: '79%' },
    ],
  },
]

const features: Feature[] = [
  {
    id: 'scheduler',
    title: 'Policy-aware scheduler',
    summary: 'Direct workloads to the right silicon with guardrails for sovereignty, power, and trust zones.',
    bullets: ['Pin models to trust domains', 'Prefer low-latency local paths', 'Spill gracefully when capacity changes'],
  },
  {
    id: 'observability',
    title: 'Operator-grade observability',
    summary: 'Watch token flow, memory pressure, and failover decisions from one control plane.',
    bullets: ['Live topology health', 'Traceable routing events', 'Deterministic replay for incidents'],
  },
  {
    id: 'deployment',
    title: 'Simple deployment surface',
    summary: 'Roll out agents onto edge devices, racks, or cloud workers through the same lifecycle primitives.',
    bullets: ['One bundle for mixed hardware', 'Click-through rollout simulator', 'Offline updates and rollbacks'],
  },
]

const commands: Command[] = [
  {
    id: 'deploy',
    label: 'Deploy Runtime',
    command: 'cog runtime deploy --mesh sovereign-edge --policy pinned-local',
    description: 'Stages the mesh runtime, validates node capabilities, and applies the active routing policy.',
    output: [
      'Mesh sovereign-edge discovered 12 nodes',
      'Pinned-local policy validated across 3 trust domains',
      'Rollout complete: 12/12 nodes healthy',
    ],
  },
  {
    id: 'benchmark',
    label: 'Run Benchmark',
    command: 'cog benchmark run --scenario mixed-topology --profile latency-first',
    description: 'Executes the current benchmark profile and compares token flow against fallback inference stacks.',
    output: [
      'Profile latency-first loaded',
      'Median first token: 0.78s',
      'Throughput uplift: +3.1x vs baseline',
    ],
  },
  {
    id: 'trace',
    label: 'Inspect Trace',
    command: 'cog trace inspect run-0482 --events route,cache,failover',
    description: 'Surfaces a deterministic replay of routing, cache hits, and rescheduling across the mesh.',
    output: [
      'Trace run-0482 loaded',
      'Route shift detected: office-npu -> dual-gpu-tower',
      'Replay verified with zero dropped requests',
    ],
  },
]

const faqs = [
  {
    id: 'copy',
    question: 'How close is this recreation to the reference product?',
    answer:
      'The layout, motion language, and information density are intentionally modeled after a premium distributed inference product experience, but the implementation here is clean-room code inside this new Vite app.',
  },
  {
    id: 'clicks',
    question: 'Do all the clicks do something?',
    answer:
      'Yes. Navigation pills scroll, scenario cards update the topology, metrics switch the benchmark story, command buttons populate the operator console, feature cards swap detail content, and the CTA opens a launch panel.',
  },
  {
    id: 'next',
    question: 'Can we wire this to a real backend later?',
    answer:
      'Yes. The components are organized around simple local state right now, which makes it straightforward to replace the mocked datasets with live API calls, websockets, and persisted routing state later.',
  },
]

const initialScenarioId = scenarios.at(0)?.id ?? ''
const initialFeatureId = features.at(0)?.id ?? ''
const initialCommandId = commands.at(0)?.id ?? ''
const initialFaqId = faqs.at(0)?.id ?? ''

function parseRoute(pathname: string): AppRoute {
  if (pathname.startsWith('/dashboard')) {
    return { kind: 'dashboard', section: getDashboardSectionFromPath(pathname) }
  }

  return { kind: 'marketing' }
}

function getPathnameForRoute(route: AppRoute): string {
  if (route.kind === 'dashboard') {
    return dashboardPathBySection[route.section]
  }

  return '/'
}

function App() {
  const [route, setRoute] = useState<AppRoute>(() => parseRoute(window.location.pathname))
  const [theme, setTheme] = useState<ThemeMode>(() => {
    const stored = window.localStorage.getItem('mesh-ui-theme')
    return stored === 'dark' ? 'dark' : 'light'
  })
  const [activeNav, setActiveNav] = useState<MarketingSection>('overview')
  const [activeMetric, setActiveMetric] = useState<MetricKey>('throughput')
  const [activeScenarioId, setActiveScenarioId] = useState(initialScenarioId)
  const [activeFeatureId, setActiveFeatureId] = useState(initialFeatureId)
  const [activeCommandId, setActiveCommandId] = useState(initialCommandId)
  const [activeFaqId, setActiveFaqId] = useState(initialFaqId)
  const [isLaunchOpen, setIsLaunchOpen] = useState(false)

  const navigate = (nextRoute: AppRoute) => {
    const nextPath = getPathnameForRoute(nextRoute)
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, '', nextPath)
    }
    setRoute(nextRoute)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  useEffect(() => {
    const onPopState = () => setRoute(parseRoute(window.location.pathname))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  useEffect(() => {
    document.documentElement.dataset['theme'] = theme
    window.localStorage.setItem('mesh-ui-theme', theme)
  }, [theme])

  const activeScenario = scenarios.find((scenario) => scenario.id === activeScenarioId) ?? scenarios.at(0)
  const activeFeature = features.find((feature) => feature.id === activeFeatureId) ?? features.at(0)
  const activeCommand = commands.find((command) => command.id === activeCommandId) ?? commands.at(0)
  const activeMetricCard = metrics[activeMetric]
  const toggleTheme = () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
  const nextCommandIdForNode = (nodeId: string) =>
    commands.at((Number(nodeId.replace(/\D/g, '')) + 1) % commands.length)?.id ?? initialCommandId

  const scrollToSection = (anchor: string, nav: MarketingSection) => {
    setActiveNav(nav)
    document.getElementById(anchor)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  if (route.kind === 'dashboard') {
    return (
      <Dashboard
        onExit={() => navigate({ kind: 'marketing' })}
        onNavigateSection={(section) => navigate({ kind: 'dashboard', section })}
        currentSection={route.section}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
    )
  }

  if (!activeScenario || !activeFeature || !activeCommand) {
    return null
  }

  return (
    <>
      <div className="shell">
        <header className="topbar">
          <button className="brand" onClick={() => scrollToSection('overview', 'overview')}>
            <span className="brand-mark">C</span>
            <span>
              <strong>Mesh UI</strong>
              <small>distributed inference, reimagined</small>
            </span>
          </button>

          <nav className="nav">
            {navigation.map((item) => (
              <button
                key={item.id}
                className={item.id === activeNav ? 'nav-pill active' : 'nav-pill'}
                onClick={() => scrollToSection(item.anchor, item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <div className="topbar-actions">
            <button className="ghost-button" onClick={toggleTheme}>
              {theme === 'dark' ? 'Light mode' : 'Dark mode'}
            </button>
            <button className="ghost-button" onClick={() => navigate({ kind: 'dashboard', section: 'overview' })}>
              Dashboard
            </button>
            <button className="ghost-button" onClick={() => scrollToSection('control', 'control')}>
              View Runtime
            </button>
            <button className="primary-button" onClick={() => setIsLaunchOpen(true)}>
              Launch Replica
            </button>
          </div>
        </header>

        <main className="page">
          <section className="hero-section panel" id="overview">
            <div className="hero-copy">
              <div className="eyebrow">Run AI anywhere</div>
              <h1>Recreated UI for a modern distributed mesh inference product.</h1>
              <p className="lead">
                Built as a standalone Vite app in <code>mesh-ui</code> with a premium control-plane
                feel, dense product storytelling, and working click paths throughout the interface.
              </p>

              <div className="hero-actions">
                <button
                  className="ghost-button"
                  onClick={() => navigate({ kind: 'dashboard', section: 'overview' })}
                >
                  Open Dashboard
                </button>
                <button className="primary-button" onClick={() => scrollToSection('runtime', 'runtime')}>
                  Explore Topology
                </button>
                <button className="ghost-button" onClick={() => setIsLaunchOpen(true)}>
                  Open Launch Sheet
                </button>
              </div>

              <div className="hero-stats">
                <article>
                  <span>5 silicon classes</span>
                  <strong>One runtime surface</strong>
                </article>
                <article>
                  <span>60% idle compute</span>
                  <strong>Recovered into the mesh</strong>
                </article>
                <article>
                  <span>Policy-first routing</span>
                  <strong>Built for sovereign workloads</strong>
                </article>
              </div>
            </div>

            <div className="hero-visual">
              <div className="signal-ring ring-one" />
              <div className="signal-ring ring-two" />
              <div className="signal-ring ring-three" />

              <div className="hero-orbit orbit-a">
                <span>GPU</span>
              </div>
              <div className="hero-orbit orbit-b">
                <span>CPU</span>
              </div>
              <div className="hero-orbit orbit-c">
                <span>NPU</span>
              </div>

              <div className="hero-core">
                <small>Cog Runtime</small>
                <strong>Inference Mesh</strong>
                <span>routing • caching • failover</span>
              </div>

              <div className="floating-card top-left">
                <span>Mixed topology</span>
                <strong>2 PCs + 3 accelerators</strong>
              </div>
              <div className="floating-card bottom-right">
                <span>Always-on agents</span>
                <strong>Failover in 180ms</strong>
              </div>
            </div>
          </section>

          <section className="metrics-section" aria-label="Benchmark switcher">
            {Object.entries(metrics).map(([key, value]) => (
              <button
                key={key}
                className={activeMetric === key ? 'metric-card active' : 'metric-card'}
                onClick={() => setActiveMetric(key as MetricKey)}
              >
                <span>{value.label}</span>
                <strong>{value.accent}</strong>
              </button>
            ))}
          </section>

          <section className="benchmark panel">
            <div>
              <div className="eyebrow">Benchmark lens</div>
              <h2>{activeMetricCard.headline}</h2>
              <p>{activeMetricCard.detail}</p>
            </div>
            <div className="benchmark-chart" aria-hidden="true">
              <div className="chart-bars">
                <div className="bar bar-primary" />
                <div className="bar bar-secondary" />
                <div className="bar bar-tertiary" />
              </div>
              <div className="chart-labels">
                <span>Cog Mesh</span>
                <span>Single host</span>
                <span>Fallback stack</span>
              </div>
            </div>
          </section>

          <section className="runtime-grid" id="runtime">
            <div className="scenario-list panel">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Runtime scenarios</div>
                  <h2>Choose a deployment shape</h2>
                </div>
                <button className="ghost-button small" onClick={() => setIsLaunchOpen(true)}>
                  Simulate rollout
                </button>
              </div>

              <div className="scenario-cards">
                {scenarios.map((scenario) => (
                  <button
                    key={scenario.id}
                    className={scenario.id === activeScenarioId ? 'scenario-card active' : 'scenario-card'}
                    onClick={() => setActiveScenarioId(scenario.id)}
                  >
                    <span>{scenario.eyebrow}</span>
                    <strong>{scenario.label}</strong>
                    <p>{scenario.rollout}</p>
                  </button>
                ))}
              </div>

              <div className="scenario-summary">
                <div className="summary-copy">
                  <div className="eyebrow">{activeScenario.eyebrow}</div>
                  <h3>{activeScenario.title}</h3>
                  <p>{activeScenario.description}</p>
                </div>
                <div className="summary-stats">
                  <article>
                    <span>Throughput</span>
                    <strong>{activeScenario.tokensPerSecond}</strong>
                  </article>
                  <article>
                    <span>Latency</span>
                    <strong>{activeScenario.latency}</strong>
                  </article>
                  <article>
                    <span>Efficiency</span>
                    <strong>{activeScenario.efficiency}</strong>
                  </article>
                </div>
              </div>
            </div>

            <div className="mesh-panel panel">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Topology view</div>
                  <h2>Live mesh composition</h2>
                </div>
                <div className="mesh-status">
                  <span className="status-dot" />
                  <span>Healthy mesh</span>
                </div>
              </div>

              <div className="mesh-canvas">
                <svg className="mesh-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
                  <path d="M18 26 C 34 35, 38 44, 50 56" />
                  <path d="M50 56 C 61 40, 66 34, 80 22" />
                  <path d="M50 56 C 57 69, 63 73, 72 78" />
                  <path d="M18 26 C 38 24, 58 24, 80 22" />
                </svg>

                {activeScenario.nodes.map((node) => (
                  <button
                    key={node.id}
                    className="mesh-node"
                    style={{ left: node.x, top: node.y }}
                    onClick={() => setActiveCommandId(nextCommandIdForNode(node.id))}
                  >
                    <span>{node.name}</span>
                    <strong>{node.type}</strong>
                    <small>{node.load}</small>
                  </button>
                ))}
              </div>

              <div className="mesh-footer">
                {activeScenario.nodes.map((node) => (
                  <article key={node.id}>
                    <span>{node.name}</span>
                    <strong>{node.status}</strong>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="control-grid" id="control">
            <div className="feature-panel panel">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Control plane</div>
                  <h2>Everything operators need</h2>
                </div>
              </div>

              <div className="feature-tabs">
                {features.map((feature) => (
                  <button
                    key={feature.id}
                    className={feature.id === activeFeatureId ? 'feature-tab active' : 'feature-tab'}
                    onClick={() => setActiveFeatureId(feature.id)}
                  >
                    <strong>{feature.title}</strong>
                    <span>{feature.summary}</span>
                  </button>
                ))}
              </div>

              <div className="feature-detail">
                <h3>{activeFeature.title}</h3>
                <p>{activeFeature.summary}</p>
                <ul>
                  {activeFeature.bullets.map((bullet) => (
                    <li key={bullet}>{bullet}</li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="console-panel panel">
              <div className="section-heading">
                <div>
                  <div className="eyebrow">Operator console</div>
                  <h2>Replay the product clicks</h2>
                </div>
              </div>

              <div className="command-tabs">
                {commands.map((command) => (
                  <button
                    key={command.id}
                    className={command.id === activeCommandId ? 'command-tab active' : 'command-tab'}
                    onClick={() => setActiveCommandId(command.id)}
                  >
                    {command.label}
                  </button>
                ))}
              </div>

              <div className="console-output">
                <div className="console-header">
                  <span className="console-dot red" />
                  <span className="console-dot amber" />
                  <span className="console-dot green" />
                  <strong>mesh-control</strong>
                </div>

                <code className="console-command">{activeCommand.command}</code>
                <p>{activeCommand.description}</p>

                <div className="console-lines">
                  {activeCommand.output.map((line) => (
                    <div key={line}>{line}</div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="faq-grid panel" id="launch">
            <div>
              <div className="eyebrow">Polish pass</div>
              <h2>Built clean, ready to extend</h2>
              <p>
                This recreation is intentionally structured so we can keep refining the fidelity or wire
                it into real APIs without ripping apart the view layer.
              </p>
            </div>

            <div className="faq-list">
              {faqs.map((faq) => (
                <button
                  key={faq.id}
                  className={faq.id === activeFaqId ? 'faq-item active' : 'faq-item'}
                  onClick={() => setActiveFaqId(faq.id)}
                >
                  <strong>{faq.question}</strong>
                  {faq.id === activeFaqId ? <p>{faq.answer}</p> : null}
                </button>
              ))}
            </div>
          </section>
        </main>
      </div>

      {isLaunchOpen ? (
        <div className="launch-sheet" role="dialog" aria-modal="true" aria-labelledby="launch-title">
          <div className="launch-backdrop" onClick={() => setIsLaunchOpen(false)} />
          <div className="launch-panel">
            <button className="close-button" onClick={() => setIsLaunchOpen(false)} aria-label="Close launch panel">
              ×
            </button>
            <div className="eyebrow">Launch replica</div>
            <h2 id="launch-title">The local `mesh-ui` experience is ready.</h2>
            <p>
              Use the running Vite app as the baseline for deeper fidelity work, backend wiring, or
              route expansion. Every major surface in this first pass is already interactive.
            </p>

            <div className="launch-grid">
              <article>
                <span>What works now</span>
                <strong>Navigation, scenario switching, topology clicks, command console, and launch sheet.</strong>
              </article>
              <article>
                <span>Best next step</span>
                <strong>Swap mock data for API streams or add route-level pages for settings and models.</strong>
              </article>
            </div>

            <div className="launch-actions">
              <button className="primary-button" onClick={() => scrollToSection('runtime', 'runtime')}>
                Jump to mesh
              </button>
              <button className="ghost-button" onClick={() => setIsLaunchOpen(false)}>
                Keep exploring
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}

export default App
