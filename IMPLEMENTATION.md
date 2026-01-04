# MeshAI Net: Implementation Roadmap

> **Note on External Research:** This implementation plan incorporates architectural patterns inspired by open-source P2P networking projects (particularly mesh VPN implementations using libp2p). All code will be implemented with MeshAI-specific naming and adaptations for our job execution use case.

## Production Difficulty Assessment

**Overall Complexity: 8.5/10**
**Realistic Timeline: 8-11 months**
**Team Size: 5-7 people (~7 FTE)**

### Complexity by Component

| Component | Difficulty | Timeline | Critical Risks |
|-----------|-----------|----------|----------------|
| Control Plane API | 6.5/10 | 2-3 weeks | Auth integration, ledger consistency |
| Relay Gateway | 7.5/10 | 2-3 weeks | Mobile connection churn, bandwidth costs |
| Desktop Agent | 6.0/10 | 2-3 weeks | Cross-platform services, GPU detection |
| **Mobile Agents** | **9.0/10** | **4-6 weeks each** | **iOS background limits (SHOWSTOPPER RISK)** |
| Workload Runtime | 7.0/10 | 3-4 weeks | Model optimization, OOM crashes |
| Credit System | 7.0/10 | 2-3 weeks | Gaming detection, validation |
| mTLS Management | 6.5/10 | 1-2 weeks | Rotation, revocation propagation |

---

## Phase 0: Minimal Viable Prototype (6-8 weeks)

**Goal:** Prove core concept with desktop-only network, relay-based embeddings execution

### Week 1-2: Critical Validation Prototypes

**MUST validate these assumptions BEFORE building the full system**

#### ✅ Checkpoint 1.1: iOS Background Compute Test (2-3 days)

**Objective:** Determine if iOS can sustain AI inference in background

**Tasks:**
- [ ] Create minimal iOS app (Swift, Xcode)
- [ ] Integrate Whisper.cpp tiny model
- [ ] Implement background task handler
- [ ] Test 30-second background task execution
- [ ] Test extended background task (request additional time)
- [ ] Monitor thermal state and battery drain
- [ ] Test under various conditions (charging, WiFi, cellular)

**Success Criteria:**
- ✅ Completes 30s+ inference without being killed by iOS
- ✅ Can request extended background time successfully
- ✅ Battery drain is acceptable (<10%/hour)

**Failure Scenario (40% probability):**
- ❌ iOS kills task after 30 seconds consistently
- **PIVOT:** Change to foreground-only or "charging + screen on" mode
- **Impact:** Redesign mobile UX, adjust marketing messaging

**Deliverable:** Decision document on iOS background strategy

---

#### ✅ Checkpoint 1.2: Relay Throughput Benchmark (3-5 days)

**Objective:** Validate relay can handle concurrent job streams

**Tasks:**
- [ ] Implement simple WebSocket relay server (Rust + tokio-tungstenite)
- [ ] Create mock device clients (2-100 concurrent connections)
- [ ] Simulate 1MB job payloads forwarded through relay
- [ ] Load test: 100 concurrent jobs
- [ ] Monitor memory usage, connection stability
- [ ] Measure p50/p99 latency

**Success Criteria:**
- ✅ p99 latency <100ms for job forwarding
- ✅ Stable memory usage under load (<500MB for 100 connections)
- ✅ No connection leaks over 30-minute test

**Failure Scenario:**
- ❌ High latency (>200ms p99) or memory leaks
- **PIVOT:** Investigate QUIC instead of WebSocket, or optimize buffer management

**Deliverable:** Relay performance report with benchmarks

---

#### ✅ Checkpoint 1.3: mTLS on Mobile (1-2 days)

**Objective:** Confirm mobile platforms support client certificates

**Tasks:**
- [ ] Generate test CA + client certificates (openssl)
- [ ] Create test HTTPS server requiring client cert
- [ ] Build minimal iOS app connecting with mTLS
- [ ] Build minimal Android app connecting with mTLS
- [ ] Test on physical devices (not just simulator)

**Success Criteria:**
- ✅ Both iOS and Android can establish mTLS connection
- ✅ Certificate validation works correctly

**Failure Scenario:**
- ❌ Platform restrictions block client certificates
- **PIVOT:** Use bearer token authentication instead of mTLS

**Deliverable:** mTLS compatibility report

---

### Week 3-4: Core Networking Layer

**Objective:** Build libp2p-based relay connectivity with NAT traversal

**Reference Architecture:** Inspired by mesh VPN patterns using libp2p relay + DCUTR (direct connection upgrade through relay)

#### ✅ Module 3.1: Network Swarm Setup (3 days)

**File:** `src/network/mesh_swarm.rs`

**Tasks:**
- [ ] Set up Cargo.toml dependencies
  ```toml
  [dependencies]
  libp2p = { version = "0.56", features = ["tcp", "quic", "noise", "yamux", "relay", "dcutr", "identify"] }
  tokio = { version = "1", features = ["full"] }
  ```
- [ ] Create `MeshSwarm` struct composing libp2p behaviors
  ```rust
  pub struct MeshSwarm {
      pub identify: libp2p::identify::Behaviour,
      pub relay_client: libp2p::relay::client::Behaviour,
      pub dcutr: libp2p::dcutr::Behaviour,
      pub job_protocol: JobProtocol,
  }
  ```
- [ ] Implement `SwarmBuilder` initialization
  - Configure TCP transport with Noise encryption + Yamux multiplexing
  - Configure QUIC transport
  - Set up relay client behavior
  - Register custom job protocol handler
- [ ] Add connection event handlers
- [ ] Test: Two nodes connect through relay

**Success Criteria:**
- ✅ Desktop device connects to relay server
- ✅ Mobile device (behind NAT) connects to relay
- ✅ Relay forwards connection between devices

**Deliverable:** Working libp2p swarm with relay connectivity

---

#### ✅ Module 3.2: Job Protocol Handler (4 days)

**File:** `src/network/job_protocol.rs`

**Tasks:**
- [ ] Define `JobEnvelope` message type
  ```rust
  #[derive(serde::Serialize, serde::Deserialize)]
  pub struct JobEnvelope {
      pub job_id: uuid::Uuid,
      pub network_id: String,
      pub workload_id: String,
      pub payload: Vec<u8>,
      pub timeout_ms: u64,
      pub auth_signature: Vec<u8>,
  }
  ```
- [ ] Implement CBOR serialization (using `ciborium` crate)
  - `JobEnvelope::read_from(stream)` - async read with length-prefix (u32)
  - `JobEnvelope::write_to(stream)` - async write with length-prefix
- [ ] Create async state machine for connection handler
  ```rust
  enum ConnectionState {
      Reading(BoxFuture<'static, io::Result<(Stream, JobEnvelope)>>),
      Writing(BoxFuture<'static, io::Result<Stream>>),
      Executing(Uuid, BoxFuture<'static, io::Result<JobResult>>),
      Idle(Option<Stream>),
  }
  ```
- [ ] Implement `NetworkBehaviour` trait for `JobProtocol`
  - Handle incoming connections with policy checks
  - Handle outgoing job submissions
  - Poll job queue and dispatch to devices
- [ ] Add connection lifecycle management
  - Graceful disconnect handling
  - Reconnection logic with exponential backoff

**Success Criteria:**
- ✅ Device A sends JobEnvelope to Device B via relay
- ✅ Device B receives, parses, and acknowledges envelope
- ✅ Handles device disconnect during job transmission

**Deliverable:** Working job protocol with async state machine

---

#### ✅ Module 3.3: Relay Server Deployment (2 days)

**File:** `relay-server/src/main.rs`

**Tasks:**
- [ ] Set up relay server using `libp2p::relay::Behaviour`
- [ ] Configure listen addresses (TCP + QUIC on 0.0.0.0:4001)
- [ ] Implement circuit relay v2 protocol
- [ ] Add relay reservation limits (max 100 reservations)
- [ ] Set up basic logging (tracing crate)
- [ ] Deploy to cloud VM (DigitalOcean/AWS/Fly.io)
- [ ] Configure firewall rules (allow TCP 4001, UDP 4001)
- [ ] Test NAT traversal: Device behind NAT → Relay → Device on public IP

**Success Criteria:**
- ✅ Relay server runs stably for 24 hours
- ✅ Mobile device behind NAT successfully connects
- ✅ Jobs forwarded through relay with <50ms added latency

**Deliverable:** Production relay server + deployment docs

---

### Week 5-6: Device Identity & Job Execution

#### ✅ Module 5.1: Device Configuration & Keys (2 days)

**File:** `src/device/device_config.rs`

**Tasks:**
- [ ] Create `DeviceConfig` struct
  ```rust
  pub struct DeviceConfig {
      pub device_id: Uuid,
      pub name: String,
      pub keypair: ed25519::Keypair,
      pub network_id: String,
      pub control_plane_url: String,
      pub capabilities: DeviceCapabilities,
  }
  ```
- [ ] Implement Ed25519 keypair generation
- [ ] Add custom serde for keypair (multibase encoding)
  ```rust
  mod keypair_serde {
      pub fn serialize(keypair: &ed25519::Keypair, s: S) -> Result<S::Ok, S::Error> {
          let encoded = multibase::encode(multibase::Base::Base58Btc, keypair.to_bytes());
          s.serialize_str(&encoded)
      }
      pub fn deserialize(d: D) -> Result<ed25519::Keypair, D::Error> {
          let string = String::deserialize(d)?;
          let (_, decoded) = multibase::decode(&string)?;
          ed25519::Keypair::try_from_bytes(&decoded)
      }
  }
  ```
- [ ] Implement TOML config file serialization
- [ ] Add `DeviceCapabilities` detection
  ```rust
  pub struct DeviceCapabilities {
      pub tier: Tier,  // tier0-tier4
      pub cpu_cores: usize,
      pub ram_mb: usize,
      pub gpu_present: bool,
      pub gpu_vram_mb: Option<usize>,
      pub os: String,
  }
  ```
- [ ] Create `DeviceConfig::generate()` factory method
- [ ] Test: Save config → Load config → Verify keypair unchanged

**Success Criteria:**
- ✅ Device generates unique Ed25519 keypair
- ✅ Config saves to `~/.meshnet/device.toml`
- ✅ Config loads correctly on restart

**Deliverable:** Device identity management system

---

#### ✅ Module 5.2: Control Plane Registration (3 days)

**Files:**
- Backend: `control-plane/src/routes/devices.ts`
- Agent: `src/api/registration.rs`

**Tasks:**
- [ ] **Backend:** Create Postgres schema
  ```sql
  CREATE TABLE devices (
      device_id UUID PRIMARY KEY,
      network_id VARCHAR NOT NULL,
      name VARCHAR NOT NULL,
      public_key BYTEA NOT NULL,
      capabilities JSONB NOT NULL,
      certificate BYTEA,
      created_at TIMESTAMPTZ NOT NULL,
      last_seen TIMESTAMPTZ,
      status VARCHAR NOT NULL  -- 'online', 'offline', 'revoked'
  );
  ```
- [ ] **Backend:** Implement `POST /api/devices/register` endpoint
  - Validate network membership
  - Generate device certificate (signed by control plane CA)
  - Store device in database
  - Return certificate + relay addresses
- [ ] **Agent:** Implement registration client
  - Send device_id, public key, capabilities
  - Receive and store certificate
  - Fetch relay addresses from control plane
- [ ] Add device heartbeat endpoint `POST /api/devices/:id/heartbeat`
- [ ] Implement soft-state presence (offline after 20s without heartbeat)

**Success Criteria:**
- ✅ Device registers successfully and receives certificate
- ✅ Device appears in control plane database
- ✅ Heartbeat updates `last_seen` timestamp

**Deliverable:** Device registration flow

---

#### ✅ Module 5.3: Embeddings Workload Executor (4 days)

**File:** `src/executor/embeddings.rs`

**Tasks:**
- [ ] Choose embedding library (llama.cpp, sentence-transformers, or ONNX Runtime)
- [ ] Download test model (all-MiniLM-L6-v2, ~90MB)
- [ ] Implement `EmbeddingsExecutor` trait
  ```rust
  pub struct EmbeddingsExecutor {
      model: Model,
      cache_dir: PathBuf,
  }

  impl EmbeddingsExecutor {
      pub async fn execute(&self, input: &str) -> Result<Vec<f32>> {
          // Load model if not cached
          // Run inference
          // Return embedding vector
      }
  }
  ```
- [ ] Add model caching (download once, reuse)
- [ ] Implement timeout handling (kill inference after job timeout_ms)
- [ ] Add resource limits (max memory, CPU affinity)
- [ ] Test inference on sample text
- [ ] Measure latency (should be <500ms for short text on CPU)

**Success Criteria:**
- ✅ Embeddings executor returns 384-dim vector for "Hello world"
- ✅ Model caches correctly (no re-download)
- ✅ Timeout kills long-running jobs

**Deliverable:** Working embeddings executor

---

#### ✅ Module 5.4: Job Execution Loop (3 days)

**File:** `src/executor/job_runner.rs`

**Tasks:**
- [ ] Create job execution event loop
  ```rust
  pub async fn run_executor(
      job_queue: JobConsumer,
      executor: EmbeddingsExecutor,
  ) -> Result<()> {
      loop {
          if let Some(job) = job_queue.try_pop() {
              let result = executor.execute(&job.input).await?;
              send_result_to_relay(job.job_id, result).await?;
          }
          tokio::time::sleep(Duration::from_millis(100)).await;
      }
  }
  ```
- [ ] Integrate with job protocol handler
- [ ] Add execution state tracking (in-progress jobs)
- [ ] Implement graceful shutdown (finish current job before exit)
- [ ] Add structured logging (job_id, duration, result size)

**Success Criteria:**
- ✅ Receives job from relay → executes → sends result back
- ✅ Logs show job lifecycle (received, started, completed)

**Deliverable:** End-to-end job execution

---

### Week 7-8: Integration, Testing & Polish

#### ✅ Module 7.1: Job Queue with Bounded Buffers (2 days)

**File:** `src/scheduler/job_queue.rs`

**Reference:** Lock-free ring buffer pattern for async job queuing

**Tasks:**
- [ ] Add dependency: `ringbuf = "0.4"`
- [ ] Create bounded job queue (capacity: 64 jobs per device)
  ```rust
  const JOB_BUFFER_SIZE: usize = 64;

  type JobProducer = ringbuf::CachingProd<
      Arc<ringbuf::StaticRb<JobEnvelope, JOB_BUFFER_SIZE>>
  >;
  type JobConsumer = ringbuf::CachingCons<
      Arc<ringbuf::StaticRb<JobEnvelope, JOB_BUFFER_SIZE>>
  >;

  pub struct DeviceJobQueue {
      producer: JobProducer,
      consumer: JobConsumer,
      overflow_policy: OverflowPolicy,
  }

  pub enum OverflowPolicy {
      DropOldest,
      RejectNew,
  }
  ```
- [ ] Implement push/pop operations
- [ ] Add overflow handling (drop oldest job when full)
- [ ] Add timeout tracking (remove jobs older than timeout_ms)
- [ ] Test: Fill queue to capacity → verify oldest job dropped

**Success Criteria:**
- ✅ Queue handles 64 concurrent jobs per device
- ✅ Overflow drops oldest job without blocking
- ✅ No memory leaks under sustained load

**Deliverable:** Production-ready job queue

---

#### ✅ Module 7.2: Error Handling Framework (1 day)

**File:** `src/errors/mod.rs`

**Reference:** Error chaining pattern with context propagation

**Tasks:**
- [ ] Create error chain type
  ```rust
  pub struct ErrorChain {
      message: String,
      context: Vec<String>,
      source: Option<Box<dyn std::error::Error>>,
  }

  impl ErrorChain {
      pub fn new(msg: impl Into<String>) -> Self { ... }
      pub fn context(mut self, ctx: impl Into<String>) -> Self { ... }
  }
  ```
- [ ] Implement `ResultExt` trait for adding context
  ```rust
  pub trait ResultExt<T> {
      fn context(self, msg: &str) -> Result<T, ErrorChain>;
  }

  impl<T, E: std::error::Error> ResultExt<T> for Result<T, E> {
      fn context(self, msg: &str) -> Result<T, ErrorChain> {
          self.map_err(|e| ErrorChain::new(msg).source(e))
      }
  }
  ```
- [ ] Add pretty-printing for errors (use `owo-colors` for terminal output)
- [ ] Replace `anyhow` usage with `ErrorChain` crate-wide

**Success Criteria:**
- ✅ Errors show full context chain in logs
- ✅ Terminal output is readable with colors

**Deliverable:** Unified error handling

---

#### ✅ Module 7.3: Basic Ledger Event Logging (2 days)

**Files:**
- Backend: `control-plane/src/routes/ledger.ts`
- Agent: `src/telemetry/ledger.rs`

**Tasks:**
- [ ] **Backend:** Create ledger events table
  ```sql
  CREATE TABLE ledger_events (
      event_id UUID PRIMARY KEY,
      network_id VARCHAR NOT NULL,
      event_type VARCHAR NOT NULL,  -- 'job_started', 'job_completed', 'credits_burned'
      job_id UUID,
      device_id UUID,
      user_id VARCHAR,
      credits_amount DECIMAL,
      metadata JSONB,
      timestamp TIMESTAMPTZ NOT NULL
  );
  ```
- [ ] **Backend:** Implement `POST /api/ledger/events` endpoint
- [ ] **Agent:** Send events on job start/complete
  - Job started: log job_id, device_id, timestamp
  - Job completed: log duration, result size, credits burned
- [ ] Calculate credits burned (duration × tier rate)
- [ ] Test: Run 10 jobs → verify 10 "job_completed" events in DB

**Success Criteria:**
- ✅ All jobs create ledger events
- ✅ Credits calculation matches tier rates

**Deliverable:** Basic ledger tracking

---

#### ✅ Checkpoint 7.4: End-to-End Integration Test (3 days)

**Objective:** Validate complete flow from network creation to job execution

**Test Scenario:**
```
1. User creates account (magic link auth)
2. User creates network "test-network"
3. Device A (macOS laptop) joins network
   - Generates keypair
   - Registers with control plane
   - Connects to relay
4. Device B (macOS laptop #2) joins network
   - Same registration flow
5. User submits embeddings job from Device A CLI
   - Job routed to Device B via relay
   - Device B executes, returns result
   - Device A receives result
6. Verify ledger shows job completed + credits burned
7. Test device disconnect during job
   - Disconnect Device B mid-job
   - Verify job marked as failed
8. Test relay failover
   - Stop relay server
   - Verify devices reconnect after relay restart
```

**Tasks:**
- [ ] Set up test environment (2 VMs or Docker containers)
- [ ] Run integration test script
- [ ] Measure end-to-end latency (should be <5s for embeddings)
- [ ] Verify no memory leaks (run 100 jobs, check RSS)
- [ ] Document bugs found

**Success Criteria:**
- ✅ 100% of jobs complete successfully
- ✅ End-to-end latency <5s (p99)
- ✅ Ledger events 100% consistent with jobs
- ✅ Graceful handling of device disconnects

**Deliverable:** Integration test report + bug fixes

---

#### ✅ Checkpoint 7.5: Phase 0 Demo & Decision Point (1 day)

**Objective:** Demonstrate prototype to stakeholders, decide Phase 1 scope

**Demo Flow:**
1. Show network creation in CLI
2. Join 2 desktop devices
3. Run embeddings job, show result
4. Show ledger events in database
5. Demonstrate device disconnect handling

**Decision Points:**
- [ ] **iOS Background Decision:** Based on Week 1 validation, finalize mobile strategy
  - ✅ If iOS background works → Proceed with mobile agents in Phase 1
  - ❌ If iOS background fails → Pivot to foreground-only, adjust timeline
- [ ] **Relay Performance:** Confirm relay bandwidth costs are acceptable
  - Estimate: 500 devices × 10GB/month = 5TB/month
  - Cloudflare: $50/month ✅
  - AWS: $450/month ❌
  - **Decision:** Use Cloudflare or similar low-egress provider
- [ ] **Phase 1 Priorities:** Rank features by value
  - Must-have: Mobile agents (iOS + Android)
  - Must-have: One more workload (OCR recommended, easier than chat)
  - Nice-to-have: Streaming (defer to Phase 2)

**Deliverable:** Phase 1 scope document + stakeholder sign-off

---

## Phase 1: MVP (12-16 weeks after Phase 0)

**Goal:** Production-ready alpha with mobile agents, 2 workloads, web dashboard

### Week 9-12: Mobile Agents (iOS)

#### ✅ Module 9.1: iOS App Foundation (1 week)

**Tasks:**
- [ ] Create Xcode project (Swift, iOS 15+ target)
- [ ] Set up navigation (SwiftUI)
  - Network list screen
  - Device detail screen
  - Job history screen
- [ ] Integrate MeshNet SDK (Swift package wrapping Rust core via FFI)
- [ ] Implement authentication (OAuth or magic link)
- [ ] Add network selection/creation UI

**Deliverable:** iOS app shell with navigation

---

#### ✅ Module 9.2: iOS Background Execution (2 weeks)

**Tasks:**
- [ ] Implement background task handler
  ```swift
  func scheduleBackgroundTask() {
      let request = BGProcessingTaskRequest(identifier: "com.meshai.compute")
      request.requiresNetworkConnectivity = true
      request.requiresExternalPower = true  // CRITICAL for iOS

      try? BGTaskScheduler.shared.submit(request)
  }
  ```
- [ ] Configure background modes in Info.plist
  - `fetch`
  - `processing`
  - `remote-notification` (for job push notifications)
- [ ] Implement job execution in background
  - Pull job from queue
  - Run Core ML inference
  - Submit result
  - Request extended background time if needed
- [ ] Add thermal monitoring
  ```swift
  ProcessInfo.processInfo.thermalState // .nominal, .fair, .serious, .critical
  ```
- [ ] Implement battery state checks
  ```swift
  UIDevice.current.batteryState == .charging && UIDevice.current.batteryLevel > 0.2
  ```
- [ ] Test on physical device (NOT simulator)
  - Background task execution
  - Thermal throttling
  - Battery drain over 1 hour

**Critical Constraints:**
- iOS will kill background tasks after 30 seconds typically
- Extended background tasks are **not guaranteed**
- Must design for "charging + foreground" as primary mode
- Background refresh is opportunistic only

**Success Criteria:**
- ✅ Executes at least 1 job in background (charging + WiFi)
- ✅ Gracefully handles iOS background termination
- ✅ Battery drain <10%/hour during contribution

**Deliverable:** iOS background job execution

---

#### ✅ Module 9.3: iOS Core ML Integration (1 week)

**Tasks:**
- [ ] Convert embeddings model to Core ML format (.mlmodel)
- [ ] Integrate Core ML framework
  ```swift
  let model = try EmbeddingsModel(configuration: MLModelConfiguration())
  let input = EmbeddingsModelInput(text: inputText)
  let output = try model.prediction(input: input)
  ```
- [ ] Implement model download + caching
  - Store in app's Documents directory
  - Only download on WiFi
  - Show download progress
- [ ] Add model size limits (max 500MB for mobile)
- [ ] Test inference latency (should be <1s on iPhone 12+)

**Deliverable:** iOS embeddings executor

---

### Week 13-16: Mobile Agents (Android)

#### ✅ Module 13.1: Android App Foundation (1 week)

**Tasks:**
- [ ] Create Android Studio project (Kotlin, minSdk 26)
- [ ] Set up Jetpack Compose UI
  - Network list screen
  - Device detail screen
  - Job history screen
- [ ] Integrate MeshNet SDK (Kotlin bindings via JNI to Rust core)
- [ ] Implement authentication
- [ ] Add network selection/creation UI

**Deliverable:** Android app shell

---

#### ✅ Module 13.2: Android Background Execution (2 weeks)

**Tasks:**
- [ ] Implement foreground service (REQUIRED for sustained work)
  ```kotlin
  class ComputeService : Service() {
      override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
          val notification = createNotification("Contributing to network...")
          startForeground(NOTIFICATION_ID, notification)
          // Start job executor
      }
  }
  ```
- [ ] Create persistent notification
  - "MeshAI is running - tap to manage"
  - Show current job status
- [ ] Implement WorkManager for periodic tasks
  ```kotlin
  val workRequest = PeriodicWorkRequestBuilder<ComputeWorker>(15, TimeUnit.MINUTES)
      .setConstraints(
          Constraints.Builder()
              .setRequiresCharging(true)
              .setRequiredNetworkType(NetworkType.UNMETERED)  // WiFi only
              .build()
      )
      .build()
  ```
- [ ] Add battery optimization exclusion request
  ```kotlin
  val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
  intent.data = Uri.parse("package:$packageName")
  startActivity(intent)
  ```
- [ ] Implement thermal monitoring (if available on device)
- [ ] Test on diverse devices
  - Samsung (aggressive battery optimization)
  - Pixel (stock Android)
  - Xiaomi (very aggressive)

**Success Criteria:**
- ✅ Foreground service runs for 1+ hour while charging
- ✅ Executes jobs successfully
- ✅ Survives Doze mode transitions

**Deliverable:** Android background job execution

---

#### ✅ Module 13.3: Android ML Integration (1 week)

**Tasks:**
- [ ] Choose ML framework
  - TFLite (best compatibility)
  - NNAPI (hardware acceleration, fragmented)
  - ONNX Runtime (cross-platform)
- [ ] Convert embeddings model to TFLite format
- [ ] Implement model loading + caching
  ```kotlin
  val interpreter = Interpreter(modelFile)
  val input = Array(1) { FloatArray(512) }
  val output = Array(1) { FloatArray(384) }
  interpreter.run(input, output)
  ```
- [ ] Add GPU delegation (optional, if available)
  ```kotlin
  val options = Interpreter.Options()
  options.addDelegate(GpuDelegate())
  ```
- [ ] Test inference latency on low-end device (should be <2s)

**Deliverable:** Android embeddings executor

---

### Week 17-20: Additional Workloads

#### ✅ Module 17.1: OCR Workload (2 weeks)

**Priority:** HIGH (easier than chat, high user value)

**Tasks:**
- [ ] Choose OCR library
  - Tesseract (mature, large binary)
  - EasyOCR (Python, good accuracy)
  - PaddleOCR (lightweight, ONNX-compatible)
- [ ] Implement `OCRExecutor`
  ```rust
  pub struct OCRExecutor {
      model: OcrModel,
  }

  impl OCRExecutor {
      pub async fn execute(&self, image: &[u8]) -> Result<String> {
          // Preprocess image (resize, denoise, grayscale)
          // Run OCR
          // Return extracted text
      }
  }
  ```
- [ ] Add image preprocessing
  - Resize to max 2048×2048
  - Grayscale conversion
  - Contrast enhancement
- [ ] Support multiple image formats (PNG, JPEG, WebP)
- [ ] Test on sample documents (receipts, forms, screenshots)
- [ ] Measure accuracy (>90% on clean text)

**Deliverable:** OCR workload executor

---

#### ✅ Module 17.2: Small Chat Workload (2 weeks)

**Priority:** MEDIUM (complex, requires streaming)

**Tasks:**
- [ ] Choose small LLM (2-4B parameters)
  - Phi-3-mini (3.8B, good quality)
  - Gemma-2B (Google, fast)
  - Llama-3.2-3B (Meta, instruction-tuned)
- [ ] Integrate llama.cpp for inference
- [ ] Implement streaming
  ```rust
  pub struct ChatExecutor {
      model: LlamaModel,
  }

  impl ChatExecutor {
      pub async fn execute_stream(
          &self,
          messages: &[ChatMessage],
          tx: mpsc::Sender<String>
      ) -> Result<()> {
          for token in self.model.generate(messages) {
              tx.send(token).await?;
          }
      }
  }
  ```
- [ ] Add context window management (max 2048 tokens)
- [ ] Implement stop sequences (</s>, <|im_end|>)
- [ ] Test latency (first token <500ms, throughput >20 tok/s on CPU)
- [ ] Add model quantization (4-bit GGUF format)

**Deliverable:** Chat workload with streaming

---

### Week 21-24: Web Dashboard & Policies

#### ✅ Module 21.1: Web Dashboard Foundation (2 weeks)

**Stack:** Next.js 14 + TypeScript + Tailwind

**Tasks:**
- [ ] Set up Next.js project
- [ ] Implement authentication (NextAuth.js)
- [ ] Create layouts
  - Dashboard layout with sidebar
  - Network selector dropdown
- [ ] Build pages
  - `/networks` - List of user's networks
  - `/networks/[id]` - Network detail with tabs
  - `/networks/[id]/devices` - Device list
  - `/networks/[id]/jobs` - Job history
  - `/networks/[id]/ledger` - Ledger events + export
  - `/networks/[id]/settings` - Network settings
- [ ] Connect to control plane API
- [ ] Add real-time updates (WebSocket or polling)

**Deliverable:** Web dashboard UI

---

#### ✅ Module 21.2: Ledger UI & Export (1 week)

**Tasks:**
- [ ] Build ledger table component
  - Columns: timestamp, event_type, device, user, credits, job_id
  - Pagination (50 events per page)
  - Filters (date range, event type, device)
- [ ] Implement CSV export
  ```typescript
  export async function exportLedger(networkId: string, format: 'csv' | 'jsonl') {
      const events = await fetchAllLedgerEvents(networkId);
      if (format === 'csv') {
          return convertToCSV(events);
      } else {
          return events.map(e => JSON.stringify(e)).join('\n');
      }
  }
  ```
- [ ] Add credit balance summary
  - Total credits minted
  - Total credits burned
  - Per-member breakdown (pie chart)
  - Per-device breakdown (bar chart)
- [ ] Test with 10,000+ events (performance)

**Deliverable:** Ledger UI with export

---

#### ✅ Module 21.3: Roles & Policies (2 weeks)

**Tasks:**
- [ ] **Backend:** Update database schema
  ```sql
  CREATE TABLE network_members (
      id UUID PRIMARY KEY,
      network_id VARCHAR NOT NULL,
      user_id VARCHAR NOT NULL,
      role VARCHAR NOT NULL,  -- 'admin', 'member', 'guest'
      invited_by VARCHAR,
      joined_at TIMESTAMPTZ,
      UNIQUE(network_id, user_id)
  );

  CREATE TABLE network_policies (
      id UUID PRIMARY KEY,
      network_id VARCHAR NOT NULL,
      workload_id VARCHAR,
      min_role VARCHAR,  -- Minimum role required
      min_tier VARCHAR,  -- Minimum device tier
      data_sensitivity VARCHAR,  -- 'public', 'private', 'sensitive'
      created_at TIMESTAMPTZ
  );
  ```
- [ ] **Backend:** Implement policy engine
  ```typescript
  function canRunWorkload(
      user: User,
      device: Device,
      workload: Workload,
      policies: Policy[]
  ): boolean {
      // Check role
      if (user.role < policies.min_role) return false;
      // Check tier
      if (device.tier < policies.min_tier) return false;
      // Check data sensitivity
      if (workload.sensitivity === 'sensitive' && device.tier < 'tier2') return false;
      return true;
  }
  ```
- [ ] **Dashboard:** Add member management UI
  - Invite member by email
  - Set role (admin/member/guest)
  - Remove member
- [ ] **Dashboard:** Add policy configuration UI
  - Set per-workload policies
  - Set per-role restrictions
  - Set data sensitivity defaults
- [ ] **Agent:** Enforce policies before job execution
  - Check policy before accepting job
  - Decline job if policy violation

**Deliverable:** Role-based access control + policy engine

---

### Week 25-28: Job Retries & Production Hardening

#### ✅ Module 25.1: Job Retry Logic (1 week)

**Tasks:**
- [ ] Implement retry policy
  ```rust
  pub struct RetryPolicy {
      pub max_attempts: u32,
      pub backoff_ms: Vec<u64>,  // [250, 750, 2000]
      pub retry_on: Vec<FailureReason>,
  }

  pub enum FailureReason {
      DeviceDisconnect,
      Timeout,
      DeviceDeclined,
      ExecutionError,
  }
  ```
- [ ] Add retry tracking to job state
  ```rust
  pub struct JobState {
      pub job_id: Uuid,
      pub attempts: u32,
      pub last_attempt_at: Option<Instant>,
      pub failure_reasons: Vec<FailureReason>,
  }
  ```
- [ ] Implement exponential backoff
  ```rust
  async fn retry_job(job: &JobState, policy: &RetryPolicy) {
      if job.attempts >= policy.max_attempts {
          mark_job_failed(job).await;
          return;
      }

      let backoff = policy.backoff_ms.get(job.attempts as usize)
          .copied()
          .unwrap_or(2000);

      tokio::time::sleep(Duration::from_millis(backoff)).await;
      reschedule_job(job).await;
  }
  ```
- [ ] Test retry scenarios
  - Device disconnects mid-job → retry on another device
  - Device declines job → retry immediately on another device
  - Job times out → retry with increased timeout
  - 3 failures → mark job as permanently failed

**Deliverable:** Robust retry mechanism

---

#### ✅ Module 25.2: Metrics & Observability (2 weeks)

**Tasks:**
- [ ] Add Prometheus metrics
  ```rust
  lazy_static! {
      static ref JOB_LATENCY: Histogram = register_histogram!(
          "job_latency_ms",
          "Job execution latency in milliseconds",
          vec![10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
      ).unwrap();

      static ref JOB_SUCCESS_RATE: Counter = register_counter!(
          "job_success_total",
          "Total number of successful jobs"
      ).unwrap();

      static ref RELAY_BANDWIDTH_BYTES: Counter = register_counter!(
          "relay_bandwidth_bytes",
          "Total bytes forwarded through relay"
      ).unwrap();
  }
  ```
- [ ] Add distributed tracing (OpenTelemetry)
  ```rust
  use tracing::{info, instrument};

  #[instrument(skip(executor))]
  async fn execute_job(job: &JobEnvelope, executor: &Executor) -> Result<JobResult> {
      info!(job_id = %job.job_id, "Executing job");
      let result = executor.execute(&job.payload).await?;
      info!(job_id = %job.job_id, latency_ms = result.latency_ms, "Job completed");
      Ok(result)
  }
  ```
- [ ] Set up Grafana dashboards
  - Job latency (p50, p95, p99)
  - Job success rate by workload
  - Device online rate by network
  - Relay bandwidth usage
  - Credits minted vs burned
- [ ] Add Sentry for error tracking
- [ ] Create alerts
  - Job success rate <70%
  - Relay down
  - Certificate expiring in <3 days

**Deliverable:** Production observability stack

---

#### ✅ Module 25.3: Certificate Rotation & Revocation (1 week)

**Tasks:**
- [ ] Implement automated certificate rotation
  ```rust
  async fn check_certificate_expiry() {
      let cert = load_device_certificate().await?;
      let days_until_expiry = cert.expiry_date().signed_duration_since(Utc::now()).num_days();

      if days_until_expiry < 7 {
          renew_certificate().await?;
      }
  }
  ```
- [ ] Add grace period for rotation (accept both old + new cert for 24 hours)
- [ ] Implement device revocation
  - Backend: Add device to revocation list
  - Relay: Check revocation list on connection
  - Agent: Poll for revocation status every 5 minutes
- [ ] Test rotation scenarios
  - Certificate expires → auto-renew
  - Device revoked → connection rejected
  - Grace period → both certs accepted

**Deliverable:** Certificate lifecycle management

---

#### ✅ Checkpoint 28.1: Phase 1 Alpha Launch (1 week)

**Objective:** Launch private alpha to 25-50 users

**Pre-Launch Checklist:**
- [ ] Security review (internal)
- [ ] Load test: 100 devices, 1000 jobs/hour
- [ ] Database backup + restore tested
- [ ] Monitoring alerts configured
- [ ] Documentation written
  - Getting started guide
  - CLI reference
  - SDK documentation
  - Troubleshooting guide
- [ ] Support channel set up (Discord/Slack)

**Alpha Metrics to Track:**
- Weekly active users
- Jobs per user per week
- Job success rate
- p50/p95/p99 latency by workload
- Device churn rate
- Support tickets

**Success Criteria:**
- ✅ 50% of users run ≥1 job per week
- ✅ 70% job success rate
- ✅ <5s p95 latency for embeddings
- ✅ <3 critical bugs per week

**Deliverable:** Private alpha launch

---

## Phase 2: Production Hardening (16-20 weeks after Phase 1)

### Advanced Scheduling (4 weeks)

**Tasks:**
- [ ] Implement eligibility filters
  - Network membership check
  - Policy validation (role, tier, sensitivity)
  - Device state check (battery, thermal)
- [ ] Implement weighted scoring
  ```rust
  fn score_device(device: &Device, job: &Job) -> f64 {
      let rtt_score = -0.30 * device.rtt_ms as f64;
      let power_score = if device.is_plugged_in { 0.20 } else { 0.0 };
      let thermal_score = 0.20 * device.thermal_headroom;
      let idle_score = 0.15 * device.idle_percentage;
      let efficiency_score = 0.15 * device.credit_efficiency;

      rtt_score + power_score + thermal_score + idle_score + efficiency_score
  }
  ```
- [ ] Add scheduler tuning UI
  - Adjust scoring weights per network
  - Set custom policies
- [ ] Test fairness (Gini coefficient should be <0.3)

**Deliverable:** Advanced job scheduler

---

### P2P Fast-Path (6 weeks)

**Tasks:**
- [ ] Implement ICE-like NAT traversal
  - STUN for address discovery
  - TURN for relay fallback
  - Candidate gathering + exchange
- [ ] Add local LAN discovery (mDNS)
- [ ] Implement connection upgrade (relay → direct)
- [ ] Add hole-punching for symmetric NAT
- [ ] Measure bandwidth savings (target: 50-70% reduction)

**Deliverable:** P2P optimization for LAN peers

---

### Model Signing & Security Hardening (4 weeks)

**Tasks:**
- [ ] Implement model manifest signing
  ```json
  {
    "model_id": "embeddings-v1",
    "version": "1.0.0",
    "sha256": "abc123...",
    "signature": "sig...",
    "signed_by": "meshai.com"
  }
  ```
- [ ] Add hash verification before model load
- [ ] External security audit (hire firm)
- [ ] Implement GDPR compliance
  - User data export API
  - User data deletion API
  - Privacy policy + ToS
- [ ] Implement abuse monitoring dashboard
  - Flag devices with suspicious credit patterns
  - Flag users running unusual workloads

**Deliverable:** Security audit report + hardened system

---

### Performance Optimization (4 weeks)

**Tasks:**
- [ ] Optimize relay forwarding
  - Zero-copy packet forwarding
  - Connection pooling
  - Batch sending
- [ ] Optimize model loading
  - Lazy loading
  - Memory-mapped files
  - Model compression (quantization)
- [ ] Optimize database queries
  - Add indexes
  - Query optimization
  - Read replicas for ledger queries
- [ ] Optimize WebSocket connections
  - Compression (permessage-deflate)
  - Batching small messages
- [ ] Load test: 1000 devices, 10,000 jobs/hour
- [ ] Measure improvement
  - Target: p99 latency <5s for embeddings
  - Target: relay p99 latency <100ms

**Deliverable:** Performance optimization report

---

## Risk Mitigation Strategies

### Critical Risk: iOS Background Compute Failure (40% probability)

**If iOS cannot sustain background inference:**

**Pivot Plan:**
1. Redesign iOS app for foreground-only mode
   - Show persistent "Contributing" screen
   - Allow user to lock screen AFTER job starts (may work for charging mode)
   - Market as "Charge your phone, earn credits"
2. Adjust marketing messaging
   - "Desktop AI pooling with mobile monitoring"
   - Focus on desktop as primary contributor
3. Add mobile job submission UX
   - Mobile becomes job requester, not executor
   - "Run embeddings on your laptop from your phone"

**Timeline Impact:** -2 weeks (simpler iOS implementation)

---

### Critical Risk: Relay Bandwidth Costs Prohibitive (30% probability)

**If relay costs are >$500/month for 500 devices:**

**Mitigation Plan:**
1. Accelerate P2P fast-path to Phase 1.5
   - Prioritize LAN discovery first (easiest)
   - Target: 70% of jobs on LAN = 70% bandwidth savings
2. Implement bandwidth-based pricing tiers
   - Free tier: 1GB relay bandwidth/month
   - Pro tier: 10GB relay bandwidth/month ($5/month)
3. Add job size optimization
   - Compress payloads (gzip)
   - Chunk large jobs

**Timeline Impact:** +4 weeks for P2P fast-path

---

### Critical Risk: Credit System Gaming (50% probability at scale)

**If users game the credit system:**

**Mitigation Plan:**
1. Invest in spot-check validation early
   - Validate 10% of jobs randomly
   - Increase rate for suspicious devices
2. Implement rate limits aggressively
   - Max credits per device per day
   - Max credits per user per day
3. Use social trust
   - Private networks = members know each other
   - Require invite approval
   - Admin can ban devices
4. Monitor for anomalies
   - Flag devices with >3σ credit earnings
   - Flag users with unusual job patterns
5. Add manual review queue
   - Suspicious events flagged for admin review

**Timeline Impact:** Ongoing investment (0.5 FTE for monitoring)

---

## Tech Stack Summary

### Control Plane
- **Language:** TypeScript (Node.js) or Rust (Axum)
- **Framework:** Express.js or Axum
- **Database:** PostgreSQL 15+
- **Cache:** Redis 7+
- **Auth:** NextAuth.js (TS) or OAuth libraries (Rust)
- **Deployment:** Docker + Fly.io/Railway/DigitalOcean

### Relay Gateway
- **Language:** Rust
- **Framework:** tokio + libp2p
- **Protocol:** QUIC (quinn) + WebSocket fallback (tokio-tungstenite)
- **Deployment:** Docker + cloud VM

### Desktop Agent
- **Language:** Rust
- **Frameworks:** tokio (async runtime), libp2p (networking)
- **ML:** llama.cpp, ONNX Runtime
- **Packaging:** cargo-bundle (macOS app), WiX (Windows installer), systemd (Linux)

### Mobile Agents
- **iOS:** Swift, Core ML
- **Android:** Kotlin, TFLite/ONNX Runtime

### Web Dashboard
- **Framework:** Next.js 14
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts or Chart.js

### Dependencies
```toml
# Rust (agent + relay)
libp2p = { version = "0.56", features = ["tcp", "quic", "noise", "yamux", "relay", "dcutr", "identify"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
ciborium = "0.2"  # CBOR serialization
uuid = { version = "1", features = ["v4", "serde"] }
ed25519-dalek = "2"
multibase = "0.9"
ringbuf = "0.4"
tracing = "0.1"
prometheus = "0.13"
opentelemetry = "0.21"

# TypeScript (control plane + dashboard)
"@prisma/client": "^5.0.0"
"next": "14.0.0"
"next-auth": "^4.24.0"
"pg": "^8.11.0"
"redis": "^4.6.0"
"zod": "^3.22.0"
```

---

## Architectural Principles

**Inspired by Production Mesh Networks:**
Our networking layer adopts patterns proven in production mesh VPN systems:

1. **Relay-First Architecture (v1)**
   - All devices maintain persistent connection to relay server
   - Relay multiplexes job streams between devices
   - Simplest NAT traversal strategy
   - Reference: Similar to Tailscale's DERP relay, but for job streams instead of IP packets

2. **Bounded Async Queues**
   - Lock-free ring buffers (64 jobs per device)
   - Prevents memory exhaustion under burst load
   - Graceful overflow handling (drop oldest job)

3. **Async State Machine for Protocol Handlers**
   - Clean separation: Reading → Writing → Executing → Idle
   - Proper futures lifecycle management
   - No blocking operations in event loop

4. **Ed25519 Identity + mTLS Transport**
   - Device identity = Ed25519 keypair
   - Transport security = mTLS over QUIC
   - Certificate-based trust (no passwords)

5. **Soft-State Presence Model**
   - Heartbeat every 5 seconds
   - Offline after 20 seconds grace period
   - No expensive distributed consensus

---

## Attribution & License Compliance

**Open Source Attribution:**
- libp2p (MIT/Apache-2.0): Core P2P networking library
- Network architecture inspired by mesh VPN patterns in open-source projects
- All MeshAI-specific code uses original naming and implementations adapted for job execution use case

**License:** TBD (recommend MIT or Apache-2.0 for agent/SDK, proprietary for control plane)

---

## Next Steps

1. **Week 0 (Before Coding):**
   - [ ] Finalize team hiring (iOS engineer priority #1)
   - [ ] Set up development infrastructure
     - GitHub repo + CI/CD
     - Staging environment (Fly.io or similar)
     - Monitoring (Grafana + Prometheus)
   - [ ] Design database schema (control plane Postgres)
   - [ ] Design API contract (OpenAPI spec)

2. **Week 1-2: Critical Validation**
   - [ ] Run iOS background compute test
   - [ ] Run relay throughput benchmark
   - [ ] Run mTLS mobile test
   - **DECISION POINT:** Proceed with Phase 0 or pivot based on results

3. **Week 3-8: Phase 0 Implementation**
   - [ ] Follow Week 3-8 checklist above
   - [ ] Weekly demos to stakeholders
   - [ ] Bi-weekly retrospectives

4. **Week 9+: Phase 1 Implementation**
   - [ ] Follow Phase 1 checklist
   - [ ] Private alpha launch at Week 28

---

## Success Metrics by Phase

### Phase 0 Success (Week 8)
- ✅ 2 desktop devices join network via relay
- ✅ Embeddings job executes successfully
- ✅ NAT traversal works for mobile device
- ✅ Ledger tracks credits burned
- ✅ End-to-end latency <5s (p99)

### Phase 1 Success (Week 28)
- ✅ 25-50 alpha users
- ✅ iOS + Android apps functional
- ✅ 50% weekly active users run ≥1 job
- ✅ 70% job success rate
- ✅ Web dashboard operational
- ✅ 2 workloads (embeddings + OCR)

### Phase 2 Success (Week 48)
- ✅ 500-1000 beta users
- ✅ 99.5% relay uptime
- ✅ P2P fast-path reduces relay bandwidth by 50%+
- ✅ Security audit complete
- ✅ <5s p99 latency for embeddings
- ✅ Ready for public launch

---

## Conclusion

MeshAI Net is an ambitious but achievable project. The critical path is:

1. **Validate iOS constraints early** (Week 1-2) - this determines mobile strategy
2. **Leverage proven P2P patterns** (libp2p relay + DCUTR) - saves weeks of NAT debugging
3. **Cut scope ruthlessly for Phase 0** - desktop-only, embeddings-only, CLI-only
4. **Invest in mobile expertise** - iOS background execution is the #1 technical risk
5. **Plan for 8-11 months to production** - not 3.5-6 months

The good news: We can adopt battle-tested networking patterns from open-source mesh VPN implementations. This provides a solid foundation for relay architecture, NAT traversal, and async protocol handlers.

The biggest unknown: iOS background compute. Everything else is standard distributed systems engineering with known solutions.

**Recommendation: Proceed with Phase 0, but validate iOS constraints in Week 1-2 before committing to full mobile implementation.**
