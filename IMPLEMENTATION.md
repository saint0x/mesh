# Mesh: Implementation Roadmap

> **Desktop-First Strategy:** Build working desktop prototype first (Weeks 1-4), validate mobile constraints second (Week 5), then decide on Phase 1 mobile approach.

> **Note on External Research:** This implementation plan incorporates architectural patterns inspired by open-source P2P networking projects (particularly mesh VPN implementations using libp2p). All code will be implemented with Mesh-specific naming and adaptations for our job execution use case.

---

## Production Difficulty Assessment

**Overall Complexity: 8.5/10**
**Realistic Timeline: 8-11 months**
**Team Size: 5-7 people (~7 FTE)**

### Complexity by Component

| Component | Difficulty | Timeline | Critical Risks |
|-----------|-----------|----------|----------------|
| Control Plane API | 6.5/10 | 2-3 weeks | Auth integration, ledger consistency |
| Relay Gateway | 7.5/10 | 2-3 weeks | Connection churn, bandwidth costs |
| Desktop Agent | 6.0/10 | 2-3 weeks | Cross-platform services, GPU detection |
| **Mobile Agents** | **9.0/10** | **4-6 weeks each** | **iOS background limits (SHOWSTOPPER RISK)** |
| Workload Runtime | 7.0/10 | 3-4 weeks | Model optimization, OOM crashes |
| Credit System | 7.0/10 | 2-3 weeks | Gaming detection, validation |
| mTLS Management | 6.5/10 | 1-2 weeks | Rotation, revocation propagation |

---

## Phase 0: Desktop Prototype (6-8 weeks)

**Goal:** Prove core concept with desktop-only network, then validate mobile constraints

**Deliverable:** Working desktop demo → validation report → Phase 1 decision

---

## Phase 0A: Desktop Foundation (Weeks 1-4)

**Goal:** Build functional desktop-to-desktop job execution through relay

### Week 1: Project Foundation & Device Identity

#### ✅ Module 1.1: Project Setup & Infrastructure (1-2 days)

**Objective:** Set up development environment and project structure

**Tasks:**
- [ ] Create Cargo workspace structure
  ```toml
  [workspace]
  members = ["agent", "relay-server", "control-plane"]
  resolver = "2"
  ```
- [ ] Set up Git repository with comprehensive .gitignore
- [ ] Configure CI/CD pipeline (GitHub Actions)
  - Rust build + test
  - Clippy lints
  - Format checks
- [ ] Set up development database (PostgreSQL + Docker Compose)
- [ ] Create initial README and CONTRIBUTING.md
- [ ] Set up project management (GitHub Projects/Linear)
- [ ] Document development workflow

**Success Criteria:**
- ✅ Team can clone, build, and run all components
- ✅ CI/CD runs on every push
- ✅ Database starts with docker-compose up

**Deliverable:** Working development environment

---

#### ✅ Module 1.2: Device Configuration & Identity (COMPLETED)

**Files:**
- `agent/src/device/mod.rs` - Main device module
- `agent/src/device/capabilities.rs` - Hardware detection and tier classification
- `agent/src/device/keypair.rs` - Ed25519 keypair with multibase serialization
- `agent/src/errors.rs` - Production error types

**Implemented:**
- ✅ Add dependencies to Cargo.toml (ed25519-dalek, serde, toml, uuid, multibase, sysinfo, dirs, rand)
- ✅ Create `DeviceConfig` struct with all required fields
- ✅ Implement Ed25519 keypair generation using `rand::rngs::OsRng`
- ✅ Add custom serde module for keypair (multibase Base58BTC encoding)
- ✅ Implement TOML config file serialization with atomic writes
- ✅ Add `DeviceCapabilities` struct with runtime detection
- ✅ Implement `Tier` enum (Tier0-Tier4) with credit multipliers
- ✅ Create `DeviceConfig::generate()` factory method
- ✅ Implement config file save/load from `~/.meshnet/device.toml`
- ✅ Add comprehensive test suite (23 tests, all passing)
- ✅ Production error handling with `thiserror`
- ✅ Structured logging with `tracing`

**Test Coverage:**
- ✅ Keypair generation and uniqueness
- ✅ Multibase Base58BTC serialization roundtrip
- ✅ Config save/load with atomic writes
- ✅ Capabilities detection and tier classification
- ✅ Error handling (invalid keys, missing files, etc.)
- ✅ All 23 tests passing
- ✅ Clippy clean (no warnings)
- ✅ Formatted with rustfmt

**Success Criteria:**
- ✅ Device generates unique Ed25519 keypair
- ✅ Config saves to `~/.meshnet/device.toml`
- ✅ Config loads correctly on restart
- ✅ Capabilities detection works on macOS/Linux/Windows
- ✅ Production-ready error handling
- ✅ Comprehensive test coverage

**Deliverable:** ✅ Production-ready device identity management system

---

#### ✅ Module 1.3: Control Plane Database Schema (1-2 days)

**Files:**
- `control-plane/docker-compose.yml`
- `control-plane/migrations/001_initial_schema.sql`

**Tasks:**
- [ ] Set up PostgreSQL 15 in Docker Compose
  ```yaml
  services:
    db:
      image: postgres:15-alpine
      environment:
        POSTGRES_DB: meshnet
        POSTGRES_USER: mesh
        POSTGRES_PASSWORD: dev_password
      ports:
        - "5432:5432"
      volumes:
        - postgres_data:/var/lib/postgresql/data
  ```
- [ ] Create database migration framework (choose SQLx or Diesel)
- [ ] Create `networks` table
  ```sql
  CREATE TABLE networks (
      network_id VARCHAR PRIMARY KEY,
      name VARCHAR NOT NULL,
      owner_user_id VARCHAR NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      settings JSONB NOT NULL DEFAULT '{}'
  );
  ```
- [ ] Create `devices` table
  ```sql
  CREATE TABLE devices (
      device_id UUID PRIMARY KEY,
      network_id VARCHAR NOT NULL REFERENCES networks(network_id),
      name VARCHAR NOT NULL,
      public_key BYTEA NOT NULL,
      capabilities JSONB NOT NULL,
      certificate BYTEA,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      last_seen TIMESTAMPTZ,
      status VARCHAR NOT NULL DEFAULT 'offline',  -- 'online', 'offline', 'revoked'
      UNIQUE(network_id, public_key)
  );

  CREATE INDEX idx_devices_network ON devices(network_id);
  CREATE INDEX idx_devices_status ON devices(status);
  ```
- [ ] Create `ledger_events` table
  ```sql
  CREATE TABLE ledger_events (
      event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      network_id VARCHAR NOT NULL REFERENCES networks(network_id),
      event_type VARCHAR NOT NULL,  -- 'job_started', 'job_completed', 'credits_burned'
      job_id UUID,
      device_id UUID REFERENCES devices(device_id),
      user_id VARCHAR,
      credits_amount DECIMAL(10, 2),
      metadata JSONB,
      timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
  );

  CREATE INDEX idx_ledger_network ON ledger_events(network_id, timestamp DESC);
  CREATE INDEX idx_ledger_device ON ledger_events(device_id, timestamp DESC);
  ```
- [ ] Add seed data for development (test network + sample device)
- [ ] Test migrations: up/down/reset
- [ ] Document schema design decisions

**Success Criteria:**
- ✅ Database schema applies cleanly
- ✅ Migrations can be rolled back
- ✅ Seed data populates correctly
- ✅ Indexes created for common queries

**Deliverable:** Working database schema with migrations

---

#### ✅ Module 1.4: Relay Server Foundation (2-3 days)

**File:** `relay-server/src/main.rs`

**Reference:** Inspired by Tailscale DERP relay architecture, adapted for job streams

**Tasks:**
- [ ] Create relay-server Cargo project
- [ ] Add dependencies
  ```toml
  [dependencies]
  libp2p = { version = "0.56", features = ["tcp", "quic", "relay", "identify", "noise", "yamux"] }
  tokio = { version = "1", features = ["full"] }
  tracing = "0.1"
  tracing-subscriber = { version = "0.3", features = ["env-filter"] }
  clap = { version = "4", features = ["derive"] }
  ```
- [ ] Implement relay server using `libp2p::relay::Behaviour`
- [ ] Configure listen addresses
  ```rust
  let tcp_addr = "/ip4/0.0.0.0/tcp/4001".parse()?;
  let quic_addr = "/ip4/0.0.0.0/udp/4001/quic-v1".parse()?;
  swarm.listen_on(tcp_addr)?;
  swarm.listen_on(quic_addr)?;
  ```
- [ ] Implement circuit relay v2 protocol
- [ ] Add relay reservation limits (max 100 concurrent reservations)
- [ ] Set up structured logging with tracing
  ```rust
  tracing_subscriber::fmt()
      .with_env_filter(EnvFilter::from_default_env())
      .init();
  ```
- [ ] Add connection event handlers (log connects/disconnects)
- [ ] Create Dockerfile for relay server
- [ ] Add CLI args for configuration (port, max connections, etc.)
- [ ] Test locally: Two test clients connect through relay

**Success Criteria:**
- ✅ Relay server starts and listens on both TCP and QUIC
- ✅ Two local test clients can connect through relay
- ✅ Logging shows connection lifecycle events
- ✅ Relay handles client disconnects gracefully

**Deliverable:** Working relay server (local deployment)

---

### Week 2-3: Network Layer & Job Protocol

#### ✅ Module 2.1: Network Swarm Setup (3 days)

**File:** `agent/src/network/mesh_swarm.rs`

**Reference:** Uses libp2p relay + DCUTR (Direct Connection Upgrade Through Relay) pattern

**Tasks:**
- [ ] Add libp2p dependencies to agent
  ```toml
  [dependencies]
  libp2p = { version = "0.56", features = ["tcp", "quic", "noise", "yamux", "relay", "dcutr", "identify"] }
  tokio = { version = "1", features = ["full"] }
  futures = "0.3"
  ```
- [ ] Create `MeshSwarm` struct composing libp2p behaviors
  ```rust
  use libp2p::swarm::NetworkBehaviour;

  #[derive(NetworkBehaviour)]
  pub struct MeshSwarm {
      pub identify: libp2p::identify::Behaviour,
      pub relay_client: libp2p::relay::client::Behaviour,
      pub dcutr: libp2p::dcutr::Behaviour,
      pub job_protocol: JobProtocol,  // Custom protocol
  }
  ```
- [ ] Implement `SwarmBuilder` initialization
  - Configure TCP transport with Noise encryption + Yamux multiplexing
  - Configure QUIC transport (built-in encryption)
  - Set up relay client behavior
  - Register custom job protocol handler (will implement next)
- [ ] Add connection event handlers
  ```rust
  SwarmEvent::ConnectionEstablished { peer_id, .. } => {
      info!("Connected to {}", peer_id);
  }
  SwarmEvent::ConnectionClosed { peer_id, .. } => {
      info!("Disconnected from {}", peer_id);
  }
  ```
- [ ] Implement relay connection logic
  - Parse relay multiaddr (e.g., `/ip4/127.0.0.1/tcp/4001/p2p/12D3...`)
  - Dial relay server
  - Listen on relay for incoming connections
- [ ] Create integration test: Two nodes connect through local relay
- [ ] Add logging for network events

**Success Criteria:**
- ✅ Desktop agent connects to relay server
- ✅ Two desktop agents connect to each other via relay
- ✅ Connection lifecycle properly logged
- ✅ Relay upgrade (DCUTR) attempted (may not succeed yet)

**Deliverable:** Working libp2p swarm with relay connectivity

---

#### ✅ Module 2.2: Job Protocol Handler (4 days)

**File:** `agent/src/network/job_protocol.rs`

**Reference:** Custom libp2p protocol for job distribution

**Tasks:**
- [ ] Define `JobEnvelope` message type
  ```rust
  use serde::{Serialize, Deserialize};
  use uuid::Uuid;

  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct JobEnvelope {
      pub job_id: Uuid,
      pub network_id: String,
      pub workload_id: String,  // "embeddings-v1", "ocr-v1", etc.
      pub payload: Vec<u8>,     // CBOR-encoded workload-specific data
      pub timeout_ms: u64,
      pub auth_signature: Vec<u8>,  // Ed25519 signature
      pub created_at: u64,      // Unix timestamp
  }

  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct JobResult {
      pub job_id: Uuid,
      pub success: bool,
      pub result: Option<Vec<u8>>,  // CBOR-encoded result
      pub error: Option<String>,
      pub execution_time_ms: u64,
  }
  ```
- [ ] Implement CBOR serialization helpers
  ```rust
  use ciborium::{into_writer, from_reader};

  impl JobEnvelope {
      pub async fn read_from(stream: &mut Stream) -> io::Result<Self> {
          // Read u32 length prefix (big-endian)
          let mut len_buf = [0u8; 4];
          stream.read_exact(&mut len_buf).await?;
          let len = u32::from_be_bytes(len_buf) as usize;

          // Read CBOR payload
          let mut buf = vec![0u8; len];
          stream.read_exact(&mut buf).await?;

          from_reader(&buf[..]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
      }

      pub async fn write_to(&self, stream: &mut Stream) -> io::Result<()> {
          let mut buf = Vec::new();
          into_writer(self, &mut buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

          // Write u32 length prefix
          let len = buf.len() as u32;
          stream.write_all(&len.to_be_bytes()).await?;

          // Write CBOR payload
          stream.write_all(&buf).await?;
          stream.flush().await?;
          Ok(())
      }
  }
  ```
- [ ] Create custom libp2p protocol for jobs
  ```rust
  use libp2p::swarm::ConnectionHandler;
  use libp2p::StreamProtocol;

  const JOB_PROTOCOL: StreamProtocol = StreamProtocol::new("/mesh/job/1.0.0");

  #[derive(NetworkBehaviour)]
  pub struct JobProtocol {
      // Internal state
  }
  ```
- [ ] Implement async state machine for connection handler
  ```rust
  enum ConnectionState {
      Idle(Option<Stream>),
      Reading(BoxFuture<'static, io::Result<(Stream, JobEnvelope)>>),
      Writing(BoxFuture<'static, io::Result<Stream>>),
      Executing(Uuid, BoxFuture<'static, io::Result<JobResult>>),
  }
  ```
- [ ] Implement `NetworkBehaviour` trait for `JobProtocol`
  - Handle incoming streams (device B receiving job)
  - Handle outgoing streams (device A submitting job)
  - Poll job queue and dispatch to executor
- [ ] Add connection lifecycle management
  - Graceful disconnect handling
  - Stream timeout (close if no data for 30s)
  - Reconnection logic with exponential backoff
- [ ] Add signature verification for incoming jobs
- [ ] Create integration test: Device A sends job → Device B receives it

**Success Criteria:**
- ✅ Device A sends `JobEnvelope` to Device B via relay
- ✅ Device B receives, parses, and validates envelope
- ✅ Protocol handles device disconnect during transmission
- ✅ Signature verification works

**Deliverable:** Working job protocol with async state machine

---

#### ✅ Module 2.3: Control Plane Registration API (2-3 days)

**Files:**
- Backend: `control-plane/src/routes/devices.rs` (or `.ts` if using TypeScript)
- Agent: `agent/src/api/registration.rs`

**Tasks:**
- [ ] Choose control plane language (TypeScript/Node.js OR Rust/Axum)
- [ ] **If TypeScript:**
  - [ ] Set up Express.js or Fastify project
  - [ ] Add dependencies: `pg`, `dotenv`, `jose` (for JWT)
  - [ ] Connect to PostgreSQL using `pg` Pool
- [ ] **If Rust:**
  - [ ] Set up Axum project
  - [ ] Add dependencies: `axum`, `sqlx`, `tokio-postgres`
  - [ ] Connect to PostgreSQL using SQLx
- [ ] Implement `POST /api/devices/register` endpoint
  ```typescript
  // TypeScript example
  interface RegisterRequest {
      device_id: string;
      network_id: string;
      name: string;
      public_key: Buffer;
      capabilities: DeviceCapabilities;
  }

  interface RegisterResponse {
      success: boolean;
      certificate?: Buffer;
      relay_addresses?: string[];
  }

  router.post('/api/devices/register', async (req, res) => {
      // 1. Validate network exists and device is authorized
      // 2. Generate device certificate (self-signed for MVP, proper CA later)
      // 3. Insert device into database
      // 4. Return certificate + relay addresses
  });
  ```
- [ ] Implement certificate generation (temporary self-signed)
  ```rust
  // Will improve to proper CA in Phase 1
  pub fn generate_device_certificate(
      device_id: Uuid,
      public_key: &[u8],
  ) -> Result<Vec<u8>> {
      // For MVP: Just return signed blob
      // Phase 1: Proper X.509 cert signed by CA
      todo!("Implement certificate generation")
  }
  ```
- [ ] Add device heartbeat endpoint `POST /api/devices/:id/heartbeat`
  ```sql
  UPDATE devices
  SET last_seen = NOW(), status = 'online'
  WHERE device_id = $1
  ```
- [ ] Implement soft-state presence
  - Background task: Every 10 seconds, mark devices offline if `last_seen > 20s ago`
- [ ] **Agent:** Implement registration client
  ```rust
  pub async fn register_device(
      config: &DeviceConfig,
      control_plane_url: &str,
  ) -> Result<RegistrationResponse> {
      let client = reqwest::Client::new();
      let resp = client.post(format!("{}/api/devices/register", control_plane_url))
          .json(&RegisterRequest {
              device_id: config.device_id,
              network_id: config.network_id.clone(),
              name: config.name.clone(),
              public_key: config.keypair.verifying_key().to_bytes().to_vec(),
              capabilities: config.capabilities.clone(),
          })
          .send()
          .await?;

      resp.json().await
  }
  ```
- [ ] **Agent:** Implement heartbeat loop
  ```rust
  pub async fn heartbeat_loop(device_id: Uuid, control_plane_url: String) {
      let mut interval = tokio::time::interval(Duration::from_secs(5));
      loop {
          interval.tick().await;
          let _ = send_heartbeat(device_id, &control_plane_url).await;
      }
  }
  ```
- [ ] Add integration test: Agent registers → shows up in database

**Success Criteria:**
- ✅ Device registers successfully and receives certificate
- ✅ Device appears in control plane database with correct capabilities
- ✅ Heartbeat updates `last_seen` timestamp
- ✅ Devices go offline after 20s without heartbeat

**Deliverable:** Device registration and presence system

---

### Week 3-4: Job Execution & Integration

#### ✅ Module 3.1: Embeddings Workload Executor (3-4 days)

**File:** `agent/src/executor/embeddings.rs`

**Tasks:**
- [ ] Choose embedding library
  - **Option A:** ONNX Runtime (cross-platform, good performance)
  - **Option B:** llama.cpp (if using llama-based models)
  - **Recommended:** ONNX Runtime for MVP
- [ ] Add dependencies
  ```toml
  [dependencies]
  ort = "2.0"  # ONNX Runtime bindings
  ndarray = "0.15"
  tokenizers = "0.15"
  ```
- [ ] Download test model (all-MiniLM-L6-v2 ONNX, ~90MB)
  - Host on GitHub Releases or S3
  - URL: `https://github.com/your-org/mesh-models/releases/download/v1/all-MiniLM-L6-v2.onnx`
- [ ] Implement `EmbeddingsExecutor` struct
  ```rust
  use ort::{Session, Value};

  pub struct EmbeddingsExecutor {
      session: Session,
      tokenizer: Tokenizer,
      cache_dir: PathBuf,
  }

  impl EmbeddingsExecutor {
      pub fn new(cache_dir: PathBuf) -> Result<Self> {
          // Load model from cache or download
          let model_path = Self::ensure_model_downloaded(&cache_dir)?;
          let session = Session::builder()?.commit_from_file(&model_path)?;
          let tokenizer = Tokenizer::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", None)?;

          Ok(Self { session, tokenizer, cache_dir })
      }

      pub async fn execute(&self, input: &str) -> Result<Vec<f32>> {
          // 1. Tokenize input
          let encoding = self.tokenizer.encode(input, false)?;
          let input_ids = encoding.get_ids();

          // 2. Run inference
          let input_tensor = Value::from_array(ndarray::arr2(&[input_ids]))?;
          let outputs = self.session.run(ort::inputs!["input_ids" => input_tensor]?)?;

          // 3. Extract embeddings from output (typically last hidden state)
          let embeddings: Vec<f32> = outputs["embeddings"].extract_tensor()?.as_slice()?.to_vec();

          Ok(embeddings)
      }

      fn ensure_model_downloaded(cache_dir: &Path) -> Result<PathBuf> {
          let model_path = cache_dir.join("all-MiniLM-L6-v2.onnx");
          if !model_path.exists() {
              // Download model
              info!("Downloading embeddings model...");
              let resp = reqwest::blocking::get("https://...")?;
              let bytes = resp.bytes()?;
              std::fs::write(&model_path, bytes)?;
          }
          Ok(model_path)
      }
  }
  ```
- [ ] Add model caching (download once, reuse)
- [ ] Implement timeout handling
  ```rust
  pub async fn execute_with_timeout(
      &self,
      input: &str,
      timeout_ms: u64,
  ) -> Result<Vec<f32>> {
      tokio::time::timeout(
          Duration::from_millis(timeout_ms),
          self.execute(input)
      )
      .await
      .map_err(|_| Error::Timeout)?
  }
  ```
- [ ] Add resource limits (optional for MVP)
- [ ] Test inference on sample text: `"Hello world"`
- [ ] Measure latency (should be <500ms on modern CPU)
- [ ] Add unit tests

**Success Criteria:**
- ✅ Executor returns 384-dim vector for "Hello world"
- ✅ Model downloads automatically on first run
- ✅ Model caches correctly (no re-download)
- ✅ Timeout works (kills long-running jobs)
- ✅ Latency <1s for short text on CPU

**Deliverable:** Working embeddings executor

---

#### ✅ Module 3.2: Job Execution Loop (2-3 days)

**File:** `agent/src/executor/job_runner.rs`

**Tasks:**
- [ ] Create `JobQueue` using bounded channel
  ```rust
  use tokio::sync::mpsc;

  pub struct JobQueue {
      tx: mpsc::Sender<JobEnvelope>,
      rx: mpsc::Receiver<JobEnvelope>,
  }

  impl JobQueue {
      pub fn new(capacity: usize) -> Self {
          let (tx, rx) = mpsc::channel(capacity);
          Self { tx, rx }
      }
  }
  ```
- [ ] Create job execution event loop
  ```rust
  pub async fn run_executor(
      mut job_queue: JobConsumer,
      executor: EmbeddingsExecutor,
      network_tx: mpsc::Sender<JobResult>,
  ) -> Result<()> {
      info!("Starting job executor loop");

      loop {
          match job_queue.recv().await {
              Some(job) => {
                  info!("Received job {}", job.job_id);

                  // Execute job
                  let start = Instant::now();
                  let result = match executor.execute_with_timeout(
                      &String::from_utf8_lossy(&job.payload),
                      job.timeout_ms,
                  ).await {
                      Ok(embeddings) => {
                          let result_bytes = ciborium::into_writer(&embeddings, Vec::new())?;
                          JobResult {
                              job_id: job.job_id,
                              success: true,
                              result: Some(result_bytes),
                              error: None,
                              execution_time_ms: start.elapsed().as_millis() as u64,
                          }
                      }
                      Err(e) => {
                          error!("Job {} failed: {}", job.job_id, e);
                          JobResult {
                              job_id: job.job_id,
                              success: false,
                              result: None,
                              error: Some(e.to_string()),
                              execution_time_ms: start.elapsed().as_millis() as u64,
                          }
                      }
                  };

                  // Send result back through network
                  network_tx.send(result).await?;
              }
              None => {
                  info!("Job queue closed, shutting down executor");
                  break;
              }
          }
      }

      Ok(())
  }
  ```
- [ ] Integrate job queue with network layer (JobProtocol)
  - Incoming jobs → push to queue
  - Completed results → send back through network
- [ ] Add execution state tracking
  ```rust
  pub struct ExecutorState {
      pub current_job: Option<(Uuid, Instant)>,
      pub jobs_completed: u64,
      pub jobs_failed: u64,
  }
  ```
- [ ] Implement graceful shutdown
  ```rust
  tokio::select! {
      _ = shutdown_signal() => {
          info!("Shutdown signal received, finishing current job...");
          // Close queue, finish current job, then exit
      }
      result = run_executor(...) => {
          result?;
      }
  }
  ```
- [ ] Add structured logging
  ```rust
  info!(
      job_id = %job.job_id,
      duration_ms = result.execution_time_ms,
      success = result.success,
      "Job completed"
  );
  ```

**Success Criteria:**
- ✅ Receives job from network → executes → sends result back
- ✅ Logs show complete job lifecycle
- ✅ Handles concurrent jobs (if queue has multiple)
- ✅ Graceful shutdown works

**Deliverable:** End-to-end job execution loop

---

#### ✅ Module 3.3: CLI for Job Submission (2 days)

**File:** `agent/src/cli.rs`

**Tasks:**
- [ ] Add CLI dependencies
  ```toml
  [dependencies]
  clap = { version = "4", features = ["derive"] }
  ```
- [ ] Create CLI commands
  ```rust
  use clap::{Parser, Subcommand};

  #[derive(Parser)]
  #[command(name = "mesh")]
  #[command(about = "Mesh AI compute network agent")]
  struct Cli {
      #[command(subcommand)]
      command: Commands,
  }

  #[derive(Subcommand)]
  enum Commands {
      /// Initialize device and join network
      Init {
          #[arg(short, long)]
          network_id: String,

          #[arg(short, long, default_value = "My Device")]
          name: String,
      },

      /// Start agent daemon
      Start,

      /// Submit a job
      Job {
          #[arg(short, long)]
          workload: String,  // "embeddings", "ocr", etc.

          #[arg(short, long)]
          input: String,

          #[arg(short, long)]
          target: Option<String>,  // Optional target device ID
      },

      /// Show device status
      Status,
  }
  ```
- [ ] Implement `mesh init` command
  - Generate device config
  - Register with control plane
  - Save certificate
- [ ] Implement `mesh start` command
  - Load device config
  - Connect to relay
  - Start job executor
  - Start heartbeat loop
- [ ] Implement `mesh job` command
  - Create JobEnvelope
  - Sign with device keypair
  - Submit to network (via relay)
  - Wait for result and print
- [ ] Implement `mesh status` command
  - Show device info
  - Show network status
  - Show recent jobs
- [ ] Add progress indicators (using `indicatif` crate)
- [ ] Add colored output (using `owo-colors`)

**Success Criteria:**
- ✅ `mesh init` creates config and registers device
- ✅ `mesh start` runs agent daemon
- ✅ `mesh job --workload embeddings --input "Hello"` submits job and shows result
- ✅ CLI has good UX (colors, progress bars, clear messages)

**Deliverable:** CLI for managing agent and submitting jobs

---

#### ✅ Checkpoint 3.4: Desktop Integration Test (2-3 days)

**Objective:** Validate complete flow from device registration to job execution

**Test Scenario:**
```
1. Start relay server locally
2. Device A: mesh init --network-id test-net --name "Laptop A"
3. Device B: mesh init --network-id test-net --name "Laptop B"
4. Device A: mesh start (in background)
5. Device B: mesh start (in background)
6. Device A: mesh job --workload embeddings --input "Hello world" --target device-b-id
7. Verify: Job executes on Device B, result returns to Device A
8. Check: Database shows job event in ledger
9. Test disconnect: Kill Device B mid-job, verify job marked as failed
10. Test reconnect: Restart Device B, verify it reconnects to relay
```

**Tasks:**
- [ ] Set up test environment (2 terminals or Docker containers)
- [ ] Write integration test script (Bash or Python)
- [ ] Run test scenario end-to-end
- [ ] Measure latency (should be <5s for embeddings)
- [ ] Check for memory leaks (run 100 jobs, monitor RSS)
- [ ] Document any bugs found
- [ ] Fix critical bugs
- [ ] Re-run tests until passing

**Success Criteria:**
- ✅ 100% of jobs complete successfully in happy path
- ✅ End-to-end latency <5s (p99)
- ✅ Database ledger events match jobs submitted
- ✅ Graceful handling of device disconnects
- ✅ No memory leaks over 100 jobs

**Deliverable:** Working desktop prototype + integration test report

---

#### ✅ Checkpoint 3.5: Desktop Demo & Documentation (1 day)

**Objective:** Prepare demo for stakeholders and document the system

**Demo Flow:**
1. Show relay server running
2. Show two devices joining network (`mesh init`)
3. Show devices starting (`mesh start` with logs)
4. Submit job from Device A to Device B
5. Show result returned in <5 seconds
6. Show ledger events in database
7. Demonstrate device disconnect → job failure
8. Demonstrate device reconnect → system recovery

**Documentation Tasks:**
- [ ] Write README for project root
  - Architecture overview
  - Quick start guide
  - Development setup
- [ ] Write agent README (`agent/README.md`)
  - CLI usage
  - Configuration options
  - Troubleshooting
- [ ] Write relay server README (`relay-server/README.md`)
  - Deployment guide
  - Configuration
- [ ] Write control plane README
  - API documentation
  - Database schema
- [ ] Create architecture diagram (using Excalidraw or similar)
- [ ] Record demo video (3-5 minutes)
- [ ] Prepare stakeholder presentation

**Deliverable:** Polished demo + comprehensive documentation

---

## Phase 0B: Mobile Validation (Week 5)

**Goal:** Validate iOS/Android constraints now that desktop prototype works

**Decision Point:** Based on these validation results, finalize Phase 1 mobile strategy

### ✅ Checkpoint 4.1: iOS Background Compute Test (2-3 days)

**Objective:** Determine if iOS can sustain AI inference in background

**Tasks:**
- [ ] Create minimal iOS app (Swift, Xcode project)
- [ ] Add Core ML or ONNX Runtime for iOS
- [ ] Integrate tiny test model (e.g., MobileNet or quantized Whisper)
- [ ] Implement background task handler
  ```swift
  import BackgroundTasks

  func scheduleBackgroundTask() {
      let request = BGProcessingTaskRequest(identifier: "com.mesh.compute")
      request.requiresNetworkConnectivity = true
      request.requiresExternalPower = true  // CRITICAL constraint

      try? BGTaskScheduler.shared.submit(request)
  }

  func handleBackgroundTask(task: BGProcessingTask) {
      // Run inference
      // Monitor time limit (iOS gives ~30s typically)
      task.expirationHandler = {
          // Task killed by iOS
      }
  }
  ```
- [ ] Test 30-second background execution
- [ ] Test extended background time request
- [ ] Monitor thermal state
  ```swift
  let thermalState = ProcessInfo.processInfo.thermalState
  // .nominal, .fair, .serious, .critical
  ```
- [ ] Monitor battery drain
  ```swift
  UIDevice.current.isBatteryMonitoringEnabled = true
  let batteryLevel = UIDevice.current.batteryLevel
  ```
- [ ] Test under various conditions:
  - WiFi vs cellular
  - Charging vs battery
  - Screen on vs screen off
  - App foreground vs background
- [ ] Test on physical device (NOT simulator)
- [ ] Run inference for 1 hour, measure battery drain

**Success Criteria (REQUIRED for Phase 1 mobile):**
- ✅ Completes 30s+ inference without being killed by iOS
- ✅ Can request extended background time successfully
- ✅ Battery drain <10%/hour while charging
- ✅ Works reliably when device is charging + on WiFi

**Failure Scenario (40% probability):**
- ❌ iOS kills task after 30 seconds consistently
- ❌ Extended background time not granted
- ❌ Battery drain too high (>20%/hour)

**Pivot Plan if iOS Fails:**
1. Redesign for foreground-only mode
   - Show "Contributing" screen while app is active
   - Allow screen lock AFTER job starts (sometimes works)
   - Market as "Charge your phone, earn credits while plugged in"
2. Adjust marketing: "Desktop AI pooling with mobile monitoring"
3. Mobile becomes job requester instead of executor
4. **Timeline Impact:** -2 weeks (simpler iOS implementation)

**Deliverable:** Decision document on iOS background strategy

---

### ✅ Checkpoint 4.2: Relay Throughput Benchmark (2-3 days)

**Objective:** Validate relay can handle production load

**Tasks:**
- [ ] Create load testing tool
  ```rust
  // relay-loadtest/src/main.rs
  use tokio::task::JoinSet;

  async fn simulate_device(device_id: usize, relay_addr: &str) {
      // Connect to relay
      // Send/receive jobs
      // Measure latency
  }

  #[tokio::main]
  async fn main() {
      let mut set = JoinSet::new();

      // Spawn 100 concurrent device simulators
      for i in 0..100 {
          set.spawn(simulate_device(i, "/ip4/127.0.0.1/tcp/4001"));
      }

      // Collect metrics
  }
  ```
- [ ] Simulate 100 concurrent device connections
- [ ] Simulate 1MB job payloads forwarded through relay
- [ ] Run load test for 30 minutes
- [ ] Monitor relay server:
  - Memory usage (should stay <500MB)
  - CPU usage
  - Connection count
  - Packet loss
- [ ] Measure p50/p95/p99 latency for job forwarding
- [ ] Check for connection leaks
  - netstat shows connections close properly
  - Memory doesn't grow unbounded

**Success Criteria:**
- ✅ p99 latency <100ms for job forwarding
- ✅ Stable memory usage under load (<500MB for 100 connections)
- ✅ No connection leaks over 30-minute test
- ✅ Relay handles reconnections gracefully

**Failure Scenario:**
- ❌ High latency (>200ms p99)
- ❌ Memory leaks or OOM
- ❌ Connection instability

**Mitigation:**
- Optimize buffer management
- Investigate QUIC instead of TCP
- Add connection pooling

**Deliverable:** Relay performance report with benchmarks

---

### ✅ Checkpoint 4.3: mTLS on Mobile (1 day)

**Objective:** Confirm iOS/Android support client certificates (for future mTLS auth)

**Tasks:**
- [ ] Generate test CA + client certificates
  ```bash
  openssl genrsa -out ca.key 2048
  openssl req -x509 -new -nodes -key ca.key -days 365 -out ca.crt
  openssl genrsa -out client.key 2048
  openssl req -new -key client.key -out client.csr
  openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt
  ```
- [ ] Create test HTTPS server requiring client cert
  ```rust
  use axum_server::tls_rustls::RustlsConfig;

  let config = RustlsConfig::from_pem_file("server.crt", "server.key").await?;
  // Add client cert verification
  ```
- [ ] Build minimal iOS app connecting with mTLS
  ```swift
  let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)

  func urlSession(
      _ session: URLSession,
      didReceive challenge: URLAuthenticationChallenge,
      completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
  ) {
      // Present client certificate
  }
  ```
- [ ] Build minimal Android app connecting with mTLS
  ```kotlin
  val keyStore = KeyStore.getInstance("PKCS12")
  // Load client cert
  val sslContext = SSLContext.getInstance("TLS")
  sslContext.init(keyManagerFactory.keyManagers, trustManagerFactory.trustManagers, null)
  ```
- [ ] Test on physical devices (iOS + Android)
- [ ] Verify handshake succeeds with proper cert
- [ ] Verify handshake fails without cert

**Success Criteria:**
- ✅ Both iOS and Android can establish mTLS connection
- ✅ Certificate validation works correctly
- ✅ No platform restrictions blocking client certs

**Failure Scenario:**
- ❌ Platform sandboxing blocks client certificates

**Pivot Plan:**
- Use bearer token authentication instead of mTLS
- Store tokens in secure keychain/keystore
- **Timeline Impact:** -1 week (simpler auth system)

**Deliverable:** mTLS compatibility report

---

### ✅ Checkpoint 4.4: Phase 1 Decision Point (0.5 days)

**Objective:** Review validation results and finalize Phase 1 scope

**Decision Criteria:**

**iOS Background Decision:**
- ✅ **If iOS works:** Proceed with full mobile agents in Phase 1
  - Allocate 4-6 weeks for iOS development
  - Allocate 4-6 weeks for Android development
  - Target: Mobile devices as job executors
- ❌ **If iOS fails:** Pivot to foreground-only or defer mobile execution
  - Build foreground-only iOS app (2 weeks)
  - Mobile as job requester only
  - Focus Phase 1 on desktop + web dashboard instead

**Relay Performance Decision:**
- ✅ **If relay performs well:** Continue with relay-first architecture
- ❌ **If relay has issues:** Prioritize P2P fast-path in Phase 1.5

**mTLS Decision:**
- ✅ **If mTLS works:** Implement proper certificate-based auth in Phase 1
- ❌ **If mTLS fails:** Use bearer tokens instead

**Phase 1 Priorities (rank these based on validation):**
1. **Must-have:** Web dashboard (for usability)
2. **Must-have:** One more workload (OCR recommended - easier than chat)
3. **Conditional:** Mobile agents (iOS + Android) - depends on iOS validation
4. **Nice-to-have:** Streaming support (defer to Phase 2 if time-constrained)
5. **Nice-to-have:** P2P fast-path (defer to Phase 2 unless relay costs prohibitive)

**Deliverable:** Phase 1 scope document + stakeholder sign-off

---

## Phase 0C: Polish & Production Prep (Week 6-8)

**Goal:** Harden desktop prototype for early alpha users

### Week 6: Error Handling & Observability

#### ✅ Module 5.1: Error Handling Framework (2 days)

**File:** `agent/src/errors/mod.rs`

**Tasks:**
- [ ] Create error chain type using `thiserror`
  ```rust
  use thiserror::Error;

  #[derive(Error, Debug)]
  pub enum MeshError {
      #[error("Network error: {0}")]
      Network(#[from] std::io::Error),

      #[error("Job execution failed: {0}")]
      Execution(String),

      #[error("Job timeout after {0}ms")]
      Timeout(u64),

      #[error("Configuration error: {0}")]
      Config(String),

      #[error("Database error: {0}")]
      Database(#[from] sqlx::Error),
  }

  pub type Result<T> = std::result::Result<T, MeshError>;
  ```
- [ ] Add context helpers
  ```rust
  pub trait ResultExt<T> {
      fn context(self, msg: &str) -> Result<T>;
  }

  impl<T, E: Into<MeshError>> ResultExt<T> for std::result::Result<T, E> {
      fn context(self, msg: &str) -> Result<T> {
          self.map_err(|e| {
              let err: MeshError = e.into();
              error!("{}: {:?}", msg, err);
              err
          })
      }
  }
  ```
- [ ] Add pretty error printing
  ```rust
  use owo_colors::OwoColorize;

  impl MeshError {
      pub fn pretty_print(&self) {
          eprintln!("{} {}", "Error:".red().bold(), self);
      }
  }
  ```
- [ ] Replace unwrap() calls with proper error handling crate-wide
- [ ] Add error tests

**Success Criteria:**
- ✅ All errors have helpful messages
- ✅ Error context shows full chain
- ✅ Terminal output is readable with colors
- ✅ No panics in normal error conditions

**Deliverable:** Unified error handling

---

#### ✅ Module 5.2: Basic Observability (2 days)

**Tasks:**
- [ ] Add structured logging throughout codebase
  ```rust
  use tracing::{info, warn, error, debug, instrument};

  #[instrument(skip(executor))]
  async fn execute_job(job: &JobEnvelope, executor: &Executor) -> Result<JobResult> {
      info!(job_id = %job.job_id, workload = %job.workload_id, "Starting job");
      let start = Instant::now();

      let result = executor.execute(&job.payload).await?;

      info!(
          job_id = %job.job_id,
          duration_ms = start.elapsed().as_millis(),
          success = result.success,
          "Job completed"
      );

      Ok(result)
  }
  ```
- [ ] Add metrics counters (simple for MVP, Prometheus later)
  ```rust
  pub struct Metrics {
      pub jobs_completed: AtomicU64,
      pub jobs_failed: AtomicU64,
      pub total_execution_time_ms: AtomicU64,
  }
  ```
- [ ] Add `mesh metrics` CLI command to show stats
- [ ] Add log file output (in addition to stdout)
  ```rust
  let file_appender = tracing_appender::rolling::daily("~/.meshnet/logs", "agent.log");
  ```
- [ ] Document logging levels (RUST_LOG environment variable)

**Success Criteria:**
- ✅ All important operations logged
- ✅ Logs include job_id, device_id, timestamps
- ✅ Can adjust log level at runtime
- ✅ Logs written to both stdout and file

**Deliverable:** Production-ready logging

---

#### ✅ Module 5.3: Basic Ledger Tracking (2 days)

**Files:**
- Backend: `control-plane/src/routes/ledger.rs`
- Agent: `agent/src/telemetry/ledger.rs`

**Tasks:**
- [ ] **Backend:** Implement `POST /api/ledger/events` endpoint
  ```rust
  #[derive(Deserialize)]
  struct LedgerEvent {
      network_id: String,
      event_type: String,
      job_id: Option<Uuid>,
      device_id: Uuid,
      credits_amount: Option<f64>,
      metadata: serde_json::Value,
  }

  async fn create_ledger_event(
      State(pool): State<PgPool>,
      Json(event): Json<LedgerEvent>,
  ) -> Result<StatusCode> {
      sqlx::query!(
          "INSERT INTO ledger_events (network_id, event_type, job_id, device_id, credits_amount, metadata)
           VALUES ($1, $2, $3, $4, $5, $6)",
          event.network_id,
          event.event_type,
          event.job_id,
          event.device_id,
          event.credits_amount,
          event.metadata,
      )
      .execute(&pool)
      .await?;

      Ok(StatusCode::CREATED)
  }
  ```
- [ ] **Agent:** Send events on job lifecycle
  ```rust
  // On job start
  pub async fn log_job_started(job_id: Uuid, device_id: Uuid) {
      let event = LedgerEvent {
          event_type: "job_started".to_string(),
          job_id: Some(job_id),
          device_id,
          ...
      };
      let _ = send_to_control_plane(event).await;
  }

  // On job complete
  pub async fn log_job_completed(job_id: Uuid, device_id: Uuid, duration_ms: u64) {
      // Calculate credits (simplified for MVP)
      let credits = calculate_credits(duration_ms, tier);

      let event = LedgerEvent {
          event_type: "job_completed".to_string(),
          job_id: Some(job_id),
          device_id,
          credits_amount: Some(credits),
          ...
      };
      let _ = send_to_control_plane(event).await;
  }
  ```
- [ ] Implement basic credits calculation
  ```rust
  fn calculate_credits(duration_ms: u64, tier: Tier) -> f64 {
      let base_rate = match tier {
          Tier::Tier0 => 1.0,
          Tier::Tier1 => 2.0,
          Tier::Tier2 => 4.0,
          Tier::Tier3 => 8.0,
          Tier::Tier4 => 16.0,
      };

      (duration_ms as f64 / 1000.0) * base_rate
  }
  ```
- [ ] Add retry logic for ledger events (if control plane is down)
- [ ] Test: Run 10 jobs → verify 10 ledger events in database

**Success Criteria:**
- ✅ All jobs create ledger events
- ✅ Credits calculation is consistent
- ✅ Events survive agent restarts (queued if control plane down)

**Deliverable:** Basic ledger tracking

---

### Week 7: Job Queue & Reliability

#### ✅ Module 6.1: Bounded Job Queue (2 days)

**File:** `agent/src/scheduler/job_queue.rs`

**Reference:** Lock-free ring buffer pattern

**Tasks:**
- [ ] Replace unbounded channel with bounded queue
  ```toml
  [dependencies]
  tokio = { version = "1", features = ["sync"] }
  ```
  ```rust
  // Use tokio's bounded channel
  pub struct JobQueue {
      tx: mpsc::Sender<JobEnvelope>,
      rx: mpsc::Receiver<JobEnvelope>,
  }

  impl JobQueue {
      pub fn new(capacity: usize) -> Self {
          let (tx, rx) = mpsc::channel(capacity);
          Self { tx, rx }
      }
  }
  ```
- [ ] Set capacity to 64 jobs per device
- [ ] Implement overflow policy
  ```rust
  pub enum OverflowPolicy {
      DropOldest,   // Drop oldest job when full
      RejectNew,    // Reject new job when full
  }
  ```
- [ ] Add timeout tracking
  ```rust
  // Remove jobs older than their timeout
  pub fn cleanup_expired(&mut self) {
      // Check job.created_at + job.timeout_ms < now
      // Remove expired jobs
  }
  ```
- [ ] Add metrics (queue size, dropped jobs)
- [ ] Test: Fill queue to capacity → verify overflow behavior

**Success Criteria:**
- ✅ Queue handles 64 concurrent jobs
- ✅ Overflow policy works (drops oldest or rejects new)
- ✅ No memory leaks under sustained load
- ✅ Expired jobs removed automatically

**Deliverable:** Production-ready job queue

---

#### ✅ Module 6.2: Job Retry Logic (2 days)

**File:** `agent/src/scheduler/retry.rs`

**Tasks:**
- [ ] Implement retry policy
  ```rust
  pub struct RetryPolicy {
      pub max_attempts: u32,
      pub backoff_ms: Vec<u64>,  // [250, 500, 1000, 2000]
      pub retry_on: HashSet<FailureReason>,
  }

  #[derive(Debug, Clone, Hash, Eq, PartialEq)]
  pub enum FailureReason {
      DeviceDisconnect,
      Timeout,
      DeviceDeclined,
      ExecutionError,
  }
  ```
- [ ] Add retry state tracking
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
  async fn retry_job(job: &mut JobState, policy: &RetryPolicy) -> Result<()> {
      if job.attempts >= policy.max_attempts {
          warn!("Job {} failed after {} attempts", job.job_id, job.attempts);
          return Err(MeshError::JobFailed);
      }

      let backoff = policy.backoff_ms
          .get(job.attempts as usize)
          .copied()
          .unwrap_or(2000);

      info!("Retrying job {} in {}ms (attempt {})", job.job_id, backoff, job.attempts + 1);
      tokio::time::sleep(Duration::from_millis(backoff)).await;

      job.attempts += 1;
      job.last_attempt_at = Some(Instant::now());

      Ok(())
  }
  ```
- [ ] Test retry scenarios:
  - Device disconnects mid-job → retry on same/different device
  - Job times out → retry with same timeout
  - Execution error → retry up to max attempts
  - Max attempts reached → mark as permanently failed

**Success Criteria:**
- ✅ Failed jobs retry with exponential backoff
- ✅ Max attempts limit respected
- ✅ Retry only on retryable failures
- ✅ Permanently failed jobs logged

**Deliverable:** Retry mechanism

---

### Week 8: Final Integration & Alpha Prep

#### ✅ Module 7.1: Cloud Relay Deployment (1-2 days)

**Objective:** Deploy relay server to production cloud

**Tasks:**
- [ ] Choose cloud provider
  - **Recommended:** Fly.io (low egress costs, good for relay)
  - **Alternative:** DigitalOcean ($6/month droplet)
  - **Avoid:** AWS (high egress costs)
- [ ] Create Dockerfile for relay
  ```dockerfile
  FROM rust:1.75 as builder
  WORKDIR /app
  COPY . .
  RUN cargo build --release --bin relay-server

  FROM debian:bookworm-slim
  RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
  COPY --from=builder /app/target/release/relay-server /usr/local/bin/
  EXPOSE 4001
  CMD ["relay-server"]
  ```
- [ ] Deploy to Fly.io
  ```bash
  fly launch
  fly deploy
  ```
- [ ] Configure firewall rules
  - Allow TCP 4001
  - Allow UDP 4001 (for QUIC)
- [ ] Set up monitoring (Fly.io metrics + logs)
- [ ] Test NAT traversal: Mobile device → Cloud relay → Desktop
- [ ] Update agent to use production relay address
  ```rust
  const PRODUCTION_RELAY: &str = "/ip4/123.456.789.0/tcp/4001/p2p/12D3...";
  ```
- [ ] Run 24-hour stability test

**Success Criteria:**
- ✅ Relay runs stably for 24+ hours
- ✅ Mobile device behind NAT connects successfully
- ✅ Jobs forwarded with <50ms added latency
- ✅ Monitoring shows healthy metrics

**Deliverable:** Production relay server

---

#### ✅ Module 7.2: End-to-End Alpha Test (2 days)

**Objective:** Final validation before inviting alpha users

**Test Scenario:**
```
1. User creates network "alpha-test"
2. User adds 3 devices (2 laptops, 1 phone if iOS works)
3. Submit 100 jobs from Device A
4. Verify all jobs execute across available devices
5. Check latency distribution (p50, p95, p99)
6. Check failure rate
7. Check ledger events match jobs
8. Test edge cases:
   - Device disconnect during job
   - Network interruption
   - Relay restart
   - Control plane restart
9. Monitor resource usage (CPU, memory, network)
10. Verify no crashes over 2-hour test
```

**Tasks:**
- [ ] Write automated test script
- [ ] Run full test scenario
- [ ] Collect metrics
- [ ] Document bugs
- [ ] Fix critical bugs
- [ ] Re-run tests
- [ ] Create test report

**Success Criteria:**
- ✅ >95% job success rate
- ✅ p99 latency <5s for embeddings
- ✅ No crashes during 2-hour test
- ✅ Ledger 100% consistent with jobs
- ✅ All edge cases handled gracefully

**Deliverable:** Alpha test report

---

#### ✅ Checkpoint 7.3: Phase 0 Complete - Alpha Launch (1 day)

**Objective:** Launch private alpha to 5-10 early users

**Pre-Launch Checklist:**
- [ ] Documentation complete
  - Installation guide
  - Quick start tutorial
  - CLI reference
  - Troubleshooting guide
  - FAQ
- [ ] Known issues documented
- [ ] Support channel set up (Discord/Slack)
- [ ] Feedback form created
- [ ] Release notes written
- [ ] GitHub Release created
- [ ] Binaries built for macOS/Linux/Windows
- [ ] Deploy control plane to production
- [ ] Monitoring dashboards set up

**Alpha User Onboarding:**
- [ ] Send invitations to 5-10 users
- [ ] Schedule onboarding calls
- [ ] Walk through installation
- [ ] Help users submit first job
- [ ] Collect initial feedback

**Alpha Metrics to Track:**
- Daily active devices
- Jobs submitted per user
- Job success rate
- p50/p95/p99 latency
- Support tickets
- Reported bugs
- Feature requests

**Success Criteria:**
- ✅ All alpha users successfully submit ≥1 job
- ✅ >80% job success rate in production
- ✅ <3 critical bugs reported in week 1
- ✅ Positive user feedback

**Deliverable:** Private alpha launch + first user feedback

---

## Phase 1: MVP with Mobile & Web (12-16 weeks)

**Goal:** Production-ready alpha with mobile agents (if iOS validation passed), web dashboard, 2+ workloads

**Scope depends on Phase 0B validation results**

### If iOS Validation Succeeded:

#### Week 9-12: Mobile Agents (iOS)
- [ ] iOS app foundation (SwiftUI)
- [ ] iOS background execution
- [ ] Core ML integration
- [ ] Mesh SDK integration via FFI

#### Week 13-16: Mobile Agents (Android)
- [ ] Android app foundation (Jetpack Compose)
- [ ] Android foreground service
- [ ] TFLite/ONNX Runtime integration
- [ ] Mesh SDK integration via JNI

### If iOS Validation Failed:

#### Week 9-12: Web Dashboard (prioritized instead)
- [ ] Next.js 14 setup
- [ ] Authentication (NextAuth)
- [ ] Network management UI
- [ ] Device list & status
- [ ] Job history & ledger
- [ ] Real-time updates

### For All Scenarios:

#### Week 13-16: Additional Workload (OCR)
- [ ] OCR executor (Tesseract or PaddleOCR)
- [ ] Image preprocessing
- [ ] Multi-format support
- [ ] Accuracy testing

#### Week 17-20: Web Dashboard (if not done) OR Streaming
- [ ] Web dashboard pages
- [ ] Ledger export (CSV/JSON)
- [ ] Credit balance visualizations
- [ ] OR: Streaming support for chat workloads

#### Week 21-24: Roles & Policies
- [ ] Member management
- [ ] Role-based access control
- [ ] Policy engine
- [ ] Data sensitivity controls

#### Week 25-28: Production Hardening
- [ ] Job retry logic (advanced)
- [ ] Metrics & observability (Prometheus + Grafana)
- [ ] Certificate rotation & revocation
- [ ] Load testing (100 devices, 1000 jobs/hour)
- [ ] Security review
- [ ] Alpha launch to 25-50 users

---

## Phase 2: Production Hardening (16-20 weeks)

### Advanced Scheduling (4 weeks)
- [ ] Eligibility filters
- [ ] Weighted device scoring
- [ ] Scheduler tuning UI
- [ ] Fairness testing

### P2P Fast-Path (6 weeks)
- [ ] ICE-like NAT traversal
- [ ] Local LAN discovery (mDNS)
- [ ] Connection upgrade (relay → direct)
- [ ] Hole-punching
- [ ] Bandwidth savings measurement

### Model Signing & Security (4 weeks)
- [ ] Model manifest signing
- [ ] Hash verification
- [ ] External security audit
- [ ] GDPR compliance
- [ ] Abuse monitoring dashboard

### Performance Optimization (4 weeks)
- [ ] Relay optimization (zero-copy forwarding)
- [ ] Model loading optimization
- [ ] Database query optimization
- [ ] Load testing: 1000 devices, 10k jobs/hour

---

## Risk Mitigation Strategies

### Critical Risk: iOS Background Failure (40% probability)

**If iOS cannot sustain background inference:**

**Pivot Plan:**
1. Foreground-only iOS app (show "Contributing" screen)
2. Market as "Desktop AI pooling with mobile monitoring"
3. Mobile becomes job requester, not executor
4. **Timeline Impact:** -2 weeks (simpler implementation)

---

### Critical Risk: Relay Bandwidth Costs (30% probability)

**If relay costs >$500/month for 500 devices:**

**Mitigation:**
1. Accelerate P2P fast-path to Phase 1.5
2. Prioritize LAN discovery (70% bandwidth savings)
3. Implement bandwidth-based pricing tiers
4. Add job payload compression
5. **Timeline Impact:** +4 weeks for P2P

---

### Critical Risk: Credit System Gaming (50% probability at scale)

**Mitigation:**
1. Spot-check validation (10% of jobs)
2. Rate limits (max credits per device/user/day)
3. Social trust (private networks, invite-only)
4. Anomaly detection (>3σ flagged)
5. Manual review queue
6. **Timeline Impact:** Ongoing (0.5 FTE for monitoring)

---

## Tech Stack Summary

### Control Plane
- **Language:** TypeScript (Node.js) or Rust (Axum)
- **Framework:** Express.js or Axum
- **Database:** PostgreSQL 15+
- **Cache:** Redis 7+ (optional for Phase 1)
- **Auth:** NextAuth.js or OAuth libraries
- **Deployment:** Docker + Fly.io/Railway/DigitalOcean

### Relay Gateway
- **Language:** Rust
- **Framework:** tokio + libp2p
- **Protocol:** QUIC (primary) + TCP fallback
- **Deployment:** Docker + Fly.io/DigitalOcean

### Desktop Agent
- **Language:** Rust
- **Frameworks:** tokio, libp2p
- **ML:** ONNX Runtime, llama.cpp
- **Packaging:** cargo-bundle (macOS), WiX (Windows), systemd (Linux)

### Mobile Agents
- **iOS:** Swift, Core ML/ONNX Runtime
- **Android:** Kotlin, TFLite/ONNX Runtime

### Web Dashboard
- **Framework:** Next.js 14
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **Charts:** Recharts

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
tracing = "0.1"
ort = "2.0"  # ONNX Runtime
clap = { version = "4", features = ["derive"] }
```

---

## Success Metrics by Phase

### Phase 0 Success (Week 8)
- ✅ 2 desktop devices join network via relay
- ✅ Embeddings job executes successfully
- ✅ NAT traversal works
- ✅ Ledger tracks credits
- ✅ End-to-end latency <5s (p99)
- ✅ 5-10 alpha users onboarded

### Phase 1 Success (Week 24-28)
- ✅ 25-50 alpha users
- ✅ iOS + Android apps (if validation passed) OR web dashboard + OCR
- ✅ 50% weekly active users run ≥1 job
- ✅ 70% job success rate
- ✅ 2+ workloads operational

### Phase 2 Success (Week 44-48)
- ✅ 500-1000 beta users
- ✅ 99.5% relay uptime
- ✅ P2P fast-path live (50%+ bandwidth savings)
- ✅ Security audit complete
- ✅ Ready for public launch

---

## Next Steps (Week 0)

### Before Starting Phase 0A:

- [ ] Finalize team roles
  - Who owns relay server?
  - Who owns control plane?
  - Who owns agent?
  - Who owns mobile (if applicable)?
- [ ] Set up development infrastructure
  - GitHub repo + CI/CD
  - PostgreSQL staging database
  - Monitoring (can be basic for MVP)
- [ ] Decide on control plane language (TypeScript vs Rust)
- [ ] Schedule weekly demos/standups
- [ ] Create project board with Phase 0A tasks

### Week 1 Kickoff:

- [ ] All team members clone repo and run `cargo build`
- [ ] All team members can run PostgreSQL locally
- [ ] Module 1.1 (Project Setup) completed
- [ ] Module 1.2 (Device Config) started

---

## Conclusion

**Desktop-First Strategy:** This revised roadmap prioritizes a working desktop prototype (Weeks 1-4), then validates mobile constraints (Week 5), allowing us to make informed decisions about Phase 1 scope based on real data, not assumptions.

**Key Advantages:**
1. **Visible progress in 4 weeks** (desktop demo)
2. **De-risk mobile early** (but don't block on it)
3. **Flexible Phase 1 scope** based on validation
4. **Better stakeholder confidence** (show working system first)

**Critical Path:**
1. Desktop prototype (Weeks 1-4)
2. Mobile validation (Week 5)
3. Phase 1 scope decision (Week 5)
4. Polished alpha (Week 8)
5. Phase 1 execution (12-16 weeks, scope depends on validation)

**Recommendation:** Start with Module 1.1 (Project Setup) this week. Get the team onboarded, repo set up, and CI/CD running. Then dive into networking core next week.

Good luck! 🚀
