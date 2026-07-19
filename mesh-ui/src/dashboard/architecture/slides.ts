export type ArchitectureNodeId =
  | 'client'
  | 'ui'
  | 'control'
  | 'planner'
  | 'db'
  | 'scheduler'
  | 'relay'
  | 'tensor'
  | 'agentA'
  | 'agentB'
  | 'agentC'
  | 'zip'
  | 'kv'
  | 'ledger'

export type RouteTone = 'accent' | 'warm' | 'cool'

export interface ArchitectureRoute {
  from: ArchitectureNodeId
  to: ArchitectureNodeId
  label: string
  tone: RouteTone
}

export interface ArchitectureMetric {
  label: string
  value: string
}

export interface ArchitectureSlide {
  id: string
  eyebrow: string
  title: string
  summary: string
  platformDetail: string
  zipDetail: string
  operatorLens: string
  metrics: ArchitectureMetric[]
  codeRefs: string[]
  focusNodes: ArchitectureNodeId[]
  routes: ArchitectureRoute[]
}

export const architectureSlides: ArchitectureSlide[] = [
  {
    id: 'foundation',
    eyebrow: 'Step 1',
    title: 'The mesh is already provisioned before a prompt exists',
    summary:
      'Devices have registered, joined a ring, advertised providers, and exposed tensor-plane endpoints. The control plane already knows who can serve, how they connect, and whether relay rendezvous may be needed.',
    platformDetail:
      'MeshNet owns device identity, ring membership, topology, capabilities, direct candidates, relay attachments, and scheduling policy.',
    zipDetail:
      'ZIP is dormant here, but the agent has already validated the exact shard, provider contract, and runtime readiness needed to admit work.',
    operatorLens:
      'This is the state you inspect on Devices, Topology, Models, Pools, and Settings before submitting anything.',
    metrics: [
      { label: 'Control stance', value: 'identity + topology + policy' },
      { label: 'Data plane bias', value: 'direct first, relay fallback' },
      { label: 'Readiness gate', value: 'exact shard + provider match' },
    ],
    codeRefs: [
      'agent/src/main.rs',
      'control-plane/src/api/routes.rs',
      'control-plane/src/services/ring_manager.rs',
      'relay-server/src/main.rs',
    ],
    focusNodes: ['control', 'db', 'relay', 'agentA', 'agentB', 'agentC', 'tensor'],
    routes: [
      { from: 'agentA', to: 'control', label: 'register + heartbeat', tone: 'cool' },
      { from: 'agentB', to: 'control', label: 'capabilities', tone: 'cool' },
      { from: 'agentC', to: 'control', label: 'direct candidates', tone: 'cool' },
      { from: 'control', to: 'db', label: 'persist topology', tone: 'accent' },
      { from: 'relay', to: 'agentB', label: 'fallback path ready', tone: 'warm' },
    ],
  },
  {
    id: 'submission',
    eyebrow: 'Step 2',
    title: 'A request enters through the Mesh surface, not ZIP directly',
    summary:
      'A user submits a prompt through `mesh job run` or the dashboard. The control plane receives a `SubmitInferenceRequest` carrying the prompt, model, limits, and request ID.',
    platformDetail:
      'MeshNet owns the public request contract, request dedupe, job creation, network selection, and API ergonomics.',
    zipDetail:
      'ZIP still has not started execution. It only receives work after planning and leasing decide who should run what.',
    operatorLens:
      'The Jobs page is the operator-facing intake surface, but the real source of truth becomes the persisted control-plane records.',
    metrics: [
      { label: 'Entry point', value: 'CLI or dashboard' },
      { label: 'Payload', value: 'prompt + model + max tokens' },
      { label: 'Idempotency', value: 'request_id required' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'agent/src/main.rs',
      'mesh-ui/src/dashboard/pages/JobsPage.tsx',
    ],
    focusNodes: ['client', 'ui', 'control'],
    routes: [
      { from: 'client', to: 'ui', label: 'submit intent', tone: 'cool' },
      { from: 'ui', to: 'control', label: 'SubmitInferenceRequest', tone: 'accent' },
    ],
  },
  {
    id: 'admission',
    eyebrow: 'Step 3',
    title: 'The control plane validates topology, prompt, and credits',
    summary:
      'Before planning, the control plane checks that the ring exists and is stable, tokenizes the prompt with the model assets, loads the manifest, and verifies the submitter has enough credits for the reservation.',
    platformDetail:
      'This is classic MeshNet shell logic: topology stability, consumption quoting, scheduling policy lookup, and admission control.',
    zipDetail:
      'ZIP constraints still matter indirectly because model manifests and provider contracts influence whether the plan can be legal.',
    operatorLens:
      'When a request fails early, it is usually because the ring is unstable, workers are absent, artifacts are missing, or credits are insufficient.',
    metrics: [
      { label: 'Topology gate', value: 'ring must be stable' },
      { label: 'Token basis', value: 'model tokenizer on prompt' },
      { label: 'Credit check', value: 'reservation before dispatch' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'control-plane/src/consumption_policy.rs',
      'control-plane/src/credit_policy.rs',
    ],
    focusNodes: ['control', 'planner', 'db', 'ledger'],
    routes: [
      { from: 'control', to: 'planner', label: 'tokenize + validate', tone: 'accent' },
      { from: 'planner', to: 'db', label: 'read manifests + policy', tone: 'cool' },
      { from: 'control', to: 'ledger', label: 'quote reservation', tone: 'warm' },
    ],
  },
  {
    id: 'planning',
    eyebrow: 'Step 4',
    title: 'MeshNet authors an explicit two-phase execution plan',
    summary:
      'The planner emits a durable `InferenceExecutionPlan` with explicit prefill and decode segments, execution groups, support groups, participant lists, KV ownership, and backend compatibility classes.',
    platformDetail:
      'MeshNet chooses who participates first, who can later decode, and which support roles exist for KV, checkpoints, recovery, or overflow.',
    zipDetail:
      'ZIP depends on this plan as the authoritative contract for segment identity, runtime mode, participants, and the legal shape of serving.',
    operatorLens:
      'This is the architectural seam: MeshNet plans the work, ZIP executes the work.',
    metrics: [
      { label: 'Core phases', value: 'prefill then decode' },
      { label: 'Group model', value: 'execution + support groups' },
      { label: 'Authority', value: 'plan persisted at submit time' },
    ],
    codeRefs: [
      'control-plane/src/services/planner.rs',
      'control-plane/src/services/scheduler.rs',
      'ENGINE.md',
    ],
    focusNodes: ['planner', 'scheduler', 'control', 'zip'],
    routes: [
      { from: 'control', to: 'planner', label: 'prepare submission', tone: 'accent' },
      { from: 'planner', to: 'scheduler', label: 'phase-aware groups', tone: 'cool' },
      { from: 'planner', to: 'zip', label: 'execution contract', tone: 'warm' },
    ],
  },
  {
    id: 'persistence',
    eyebrow: 'Step 5',
    title: 'The request becomes durable control-plane state in one transaction',
    summary:
      'The control plane inserts the job, session, replicas, serving groups, job assignments, ledger events, and a decode-queue row that starts in `blocked_on_prefill`.',
    platformDetail:
      'MeshNet is the authoritative persistence layer. It turns intent into recoverable state before any worker begins execution.',
    zipDetail:
      'ZIP has no durable control database of its own here; it relies on MeshNet to represent session authority and queue state.',
    operatorLens:
      'This is why the dashboard can reconstruct the lifecycle after crashes, regroups, or restarts.',
    metrics: [
      { label: 'Tables touched', value: 'jobs + sessions + queue + ledger' },
      { label: 'Initial session state', value: 'prefill_pending' },
      { label: 'Decode queue state', value: 'blocked_on_prefill' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'control-plane/migrations/018_create_inference_sessions.sql',
      'control-plane/migrations/022_create_inference_decode_queue.sql',
    ],
    focusNodes: ['db', 'ledger', 'control', 'scheduler'],
    routes: [
      { from: 'control', to: 'db', label: 'transaction commit', tone: 'accent' },
      { from: 'control', to: 'ledger', label: 'credits_reserved', tone: 'warm' },
      { from: 'scheduler', to: 'db', label: 'seed decode queue', tone: 'cool' },
    ],
  },
  {
    id: 'leasing',
    eyebrow: 'Step 6',
    title: 'Workers poll and the scheduler leases the next legal assignment',
    summary:
      'Agents continuously claim work. The scheduler ranks runnable candidates, reconciles stale leases, prefers legal prefill work first, and can reason about decode cohorts once prefill is complete.',
    platformDetail:
      'MeshNet owns claim-time arbitration and fairness. This is where queue age, topology, policy mode, and lease state become concrete scheduling decisions.',
    zipDetail:
      'ZIP receives a leased segment and session context only after MeshNet decides the worker should run it.',
    operatorLens:
      'If a worker is idle while jobs exist, the answer is usually in the scheduler snapshot: blocked reason, lease owner, transfer debt, or cohort readiness.',
    metrics: [
      { label: 'Claim cadence', value: 'polling agents' },
      { label: 'Policies', value: 'fit / throughput / latency / resilient' },
      { label: 'Blocked reasons', value: 'prefill, transfer, lease, eligibility' },
    ],
    codeRefs: [
      'control-plane/src/services/scheduler.rs',
      'control-plane/src/api/inference.rs',
      'agent/src/main.rs',
    ],
    focusNodes: ['scheduler', 'db', 'agentA', 'agentB', 'agentC'],
    routes: [
      { from: 'agentA', to: 'scheduler', label: 'claim assignment', tone: 'accent' },
      { from: 'scheduler', to: 'db', label: 'lease selection', tone: 'cool' },
      { from: 'scheduler', to: 'agentA', label: 'execution lease', tone: 'warm' },
    ],
  },
  {
    id: 'runtime-ready',
    eyebrow: 'Step 7',
    title: 'The chosen agent proves it can really run the exact shard',
    summary:
      'Before serving, the agent validates the selected provider, checks tokenizer and safetensor assets, materializes the shard into resident memory, and joins or refreshes its ring position.',
    platformDetail:
      'MeshNet still owns the process lifecycle, device config, ring metadata, and local capability reporting.',
    zipDetail:
      'ZIP now becomes active: shard loading, model residency caching, and runtime state are prepared inside the agent boundary exposed through `agent/src/zip.rs`.',
    operatorLens:
      'This prevents fake readiness. A worker is not eligible just because it registered; it must prove the exact shard-provider pair can materialize.',
    metrics: [
      { label: 'Provider rule', value: 'no silent fallback' },
      { label: 'Artifact scope', value: 'exact shard package' },
      { label: 'Residency model', value: 'shared model cache' },
    ],
    codeRefs: [
      'agent/src/main.rs',
      'agent/src/zip.rs',
      'agent/src/inference/artifact_loader.rs',
      'agent/src/inference/coordinator.rs',
    ],
    focusNodes: ['agentA', 'zip', 'kv'],
    routes: [
      { from: 'agentA', to: 'zip', label: 'load shard + backend', tone: 'accent' },
      { from: 'zip', to: 'kv', label: 'reserve runtime memory', tone: 'cool' },
    ],
  },
  {
    id: 'transport',
    eyebrow: 'Step 8',
    title: 'The request gets a real transport path across the mesh',
    summary:
      'Workers prefer direct tensor-plane connectivity on the local network. If that cannot be established, Mesh can rendezvous through a libp2p relay while still trying to upgrade peers onto a direct path.',
    platformDetail:
      'MeshNet owns connectivity posture, peer punch plans, relayed attachments, and topology-aware path planning.',
    zipDetail:
      'ZIP depends on the resulting tensor path because its ring execution pushes tensors through neighbors on the hot path.',
    operatorLens:
      'A relayed path is a survivability tool, not the intended fast path. The Topology page exposes where relay rendezvous is required.',
    metrics: [
      { label: 'Hot path', value: 'direct tensor transport' },
      { label: 'Fallback', value: 'relay rendezvous + upgrade' },
      { label: 'Coordination', value: 'explicit punch plans' },
    ],
    codeRefs: [
      'control-plane/src/services/ring_manager.rs',
      'agent/src/network/tensor_plane.rs',
      'relay-server/src/relay.rs',
    ],
    focusNodes: ['agentA', 'agentB', 'agentC', 'tensor', 'relay'],
    routes: [
      { from: 'agentA', to: 'tensor', label: 'direct lane', tone: 'accent' },
      { from: 'relay', to: 'agentB', label: 'rendezvous fallback', tone: 'warm' },
      { from: 'tensor', to: 'agentC', label: 'neighbor path', tone: 'cool' },
    ],
  },
  {
    id: 'prefill',
    eyebrow: 'Step 9',
    title: 'ZIP runs prefill once to seed the session and first logits',
    summary:
      'The agent acknowledges the lease and executes the prefill segment through the tensor-parallel ring. Prompt tokens traverse shard-local forward passes and collective exchange until the first logits and live KV state exist.',
    platformDetail:
      'MeshNet tracks the phase transition and the assignment acknowledgment, but it does not perform the math itself.',
    zipDetail:
      'ZIP owns the actual forward pass, the ring all-reduce behavior, the prompt prefill semantics, and initial KV materialization.',
    operatorLens:
      'This is the moment where the session graduates from planned work to live model state.',
    metrics: [
      { label: 'Phase intent', value: 'single-session prompt fill' },
      { label: 'Data product', value: 'first logits + live KV' },
      { label: 'Runtime boundary', value: 'agent-embedded ZIP engine' },
    ],
    codeRefs: [
      'agent/src/inference/forward_pass.rs',
      'agent/src/inference/backend.rs',
      'agent/src/executor/ring_allreduce.rs',
    ],
    focusNodes: ['zip', 'agentA', 'agentB', 'agentC', 'tensor', 'kv'],
    routes: [
      { from: 'agentA', to: 'zip', label: 'ack + start prefill', tone: 'accent' },
      { from: 'zip', to: 'tensor', label: 'tensor-parallel exchange', tone: 'cool' },
      { from: 'zip', to: 'kv', label: 'seed live KV', tone: 'warm' },
    ],
  },
  {
    id: 'decode-queue',
    eyebrow: 'Step 10',
    title: 'Prefill completion opens a pooled decode cohort',
    summary:
      'Once prefill finishes, MeshNet refreshes decode placement against live topology, computes pooled batch targets, and promotes sessions from blocked to ready, leased, or active decode states.',
    platformDetail:
      'This is a MeshNet specialty: the scheduler reasons about cohort density, transfer debt, readiness, owned groups, and policy mode to decide how decode should open.',
    zipDetail:
      'ZIP only sees the sessions that are activated for decode. The cohorting logic itself lives outside the engine in the control plane.',
    operatorLens:
      'This is where continuous batching becomes operationally visible rather than just a backend optimization.',
    metrics: [
      { label: 'Queue states', value: 'ready / leased / active / blocked' },
      { label: 'Pooling key', value: 'shared serving participants' },
      { label: 'Lease target', value: 'session count == batch size' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'control-plane/src/services/scheduler.rs',
      'ENGINE.md',
    ],
    focusNodes: ['scheduler', 'db', 'zip', 'agentA', 'agentB'],
    routes: [
      { from: 'zip', to: 'scheduler', label: 'prefill complete', tone: 'accent' },
      { from: 'scheduler', to: 'db', label: 'refresh decode cohort', tone: 'cool' },
      { from: 'scheduler', to: 'agentB', label: 'lease siblings together', tone: 'warm' },
    ],
  },
  {
    id: 'decode-fast-path',
    eyebrow: 'Step 11',
    title: 'ZIP executes decode as a bounded microbatch fast path',
    summary:
      'Active sessions are merged into a decode microbatch when provider contracts align and KV budgets allow it. The runtime enforces max active sessions, total KV tokens, and memory ceilings while choosing the fastest safe path.',
    platformDetail:
      'MeshNet tells ZIP which sessions belong together through lease targets and serving-group ownership.',
    zipDetail:
      'ZIP owns microbatch formation, fast-path bucket planning, workspace reuse, backend specialization, and the per-token decode loop.',
    operatorLens:
      'This is the highest-leverage performance surface in the whole stack because it converts many small interactive sessions into one efficient hardware step.',
    metrics: [
      { label: 'Batch control', value: 'size + KV-token guardrails' },
      { label: 'Optimization', value: 'provider fast path buckets' },
      { label: 'Fairness', value: 'defer instead of overload' },
    ],
    codeRefs: [
      'agent/src/inference/coordinator.rs',
      'agent/src/inference/backend.rs',
      'agent/src/inference/fast_path.rs',
    ],
    focusNodes: ['zip', 'kv', 'agentA', 'agentB', 'agentC'],
    routes: [
      { from: 'scheduler', to: 'zip', label: 'decode cohort target', tone: 'accent' },
      { from: 'zip', to: 'kv', label: 'budget live KV', tone: 'warm' },
      { from: 'zip', to: 'agentC', label: 'batched token step', tone: 'cool' },
    ],
  },
  {
    id: 'kv-handoff',
    eyebrow: 'Step 12',
    title: 'KV handoff and checkpoint payloads keep locality warm',
    summary:
      'When decode placement changes or support members need state, MeshNet records KV residency and transfer rows while ZIP can export checkpoint-backed session state. Payloads are uploaded, remotely referenced, and replayed without forcing full restart semantics.',
    platformDetail:
      'MeshNet persists residency, transfer, remote-access URI, and prompt-cache metadata so the session can move deliberately instead of blindly.',
    zipDetail:
      'ZIP provides the exportable checkpoint and the live KV snapshot semantics needed for real recovery and continuation.',
    operatorLens:
      'This is how the platform avoids recomputing the whole prompt every time placement changes.',
    metrics: [
      { label: 'Residency view', value: 'owner + replica + shard range' },
      { label: 'Transfer kind', value: 'checkpoint handoff' },
      { label: 'Recovery mode', value: 'remote reference or import' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'agent/src/inference/coordinator.rs',
      'agent/src/checkpoint/manager.rs',
    ],
    focusNodes: ['kv', 'db', 'zip', 'agentB', 'agentC'],
    routes: [
      { from: 'zip', to: 'kv', label: 'export checkpoint', tone: 'accent' },
      { from: 'kv', to: 'db', label: 'residency + transfer metadata', tone: 'cool' },
      { from: 'db', to: 'agentC', label: 'remote payload access', tone: 'warm' },
    ],
  },
  {
    id: 'failover',
    eyebrow: 'Step 13',
    title: 'Regroup and failover preserve the session when the mesh changes',
    summary:
      'If a participant disappears, the control plane can select replacements, shrink a decode group when necessary, and keep session authority coherent while the agent pauses, checkpoints, resumes, or recovers from the latest safe point.',
    platformDetail:
      'MeshNet owns regroup policy, event history, replacement legality, and the tradeoff between failover cost and restart cost.',
    zipDetail:
      'ZIP owns pause/resume semantics, checkpoint recovery, and restoring session-local runtime state without corrupting the decode stream.',
    operatorLens:
      'The point is not perfect invisibility. The point is bounded disruption with a truthful control story.',
    metrics: [
      { label: 'Protection mode', value: 'pause before corrupting' },
      { label: 'Fallback shape', value: 'replace or shrink' },
      { label: 'Recovery budget', value: 'governed checkpoint loads' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'control-plane/migrations/024_create_inference_regroup_events.sql',
      'agent/src/inference/coordinator.rs',
    ],
    focusNodes: ['scheduler', 'db', 'zip', 'agentA', 'agentC'],
    routes: [
      { from: 'agentA', to: 'scheduler', label: 'participant loss', tone: 'warm' },
      { from: 'scheduler', to: 'db', label: 'regroup event', tone: 'accent' },
      { from: 'db', to: 'zip', label: 'resume with new cohort', tone: 'cool' },
    ],
  },
  {
    id: 'settlement',
    eyebrow: 'Step 14',
    title: 'Results, credits, and dashboard surfaces close the loop',
    summary:
      'Agents stream progress, report final inference results, and release decode leases. The control plane settles credits, releases unused reservations, marks KV state stale when appropriate, and exposes the full trail back to the dashboard.',
    platformDetail:
      'MeshNet finishes the business transaction: result persistence, ledger updates, queue cleanup, and observability surfaces.',
    zipDetail:
      'ZIP is done once it returns the generated tokens and final runtime stats, but those outputs remain legible because MeshNet wraps them in durable status and operator UX.',
    operatorLens:
      'Jobs, Ledger, Credits, Topology, and Overview are all different windows onto this same authoritative lifecycle.',
    metrics: [
      { label: 'Reports', value: 'progress + final result + release' },
      { label: 'Accounting', value: 'reserved -> settled -> released' },
      { label: 'Visibility', value: 'dashboard mirrors control-plane truth' },
    ],
    codeRefs: [
      'control-plane/src/api/inference.rs',
      'mesh-ui/src/dashboard/pages/OverviewPage.tsx',
      'mesh-ui/src/dashboard/pages/LedgerPage.tsx',
      'mesh-ui/src/dashboard/pages/JobsPage.tsx',
    ],
    focusNodes: ['zip', 'control', 'db', 'ledger', 'ui'],
    routes: [
      { from: 'zip', to: 'control', label: 'report progress + result', tone: 'accent' },
      { from: 'control', to: 'ledger', label: 'settle credits', tone: 'warm' },
      { from: 'db', to: 'ui', label: 'dashboard snapshot', tone: 'cool' },
    ],
  },
]
