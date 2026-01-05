# Mesh: Architectural Insight - Tensor-Parallel Cooperative Pooling

> **Higher-Level Vision:** This document describes the architectural vision for Mesh - a cooperative compute pooling system that enables groups of people to collectively run large AI models that no single device could run alone. The current `IMPLEMENTATION.md` roadmap focuses on building the foundational execution layer. This document explains the complete tensor-parallel architecture that enables distributed inference at scale.

---

## Executive Summary

Mesh is a **tensor-parallel distributed inference system** that allows groups of people to pool their compute resources to collectively run AI models too large for any individual machine.

### The Core Problem

**You can't run a 64GB Llama-70B model on your 8GB laptop. But 10 friends with 8GB laptops CAN run it together.**

### The Solution

Mesh uses **tensor parallelism** with **ring all-reduce** to split model weights across devices, allowing cooperative execution of inference workloads with low latency (~10 seconds for full generation).

### Key Architecture Components

1. **Workers** - Devices that contribute locked compute resources and execute model shards
2. **Executors** - Users (often workers themselves) who SSH into containers to access inference
3. **Control Plane** - Orchestrates worker ring topology, manages jobs, handles checkpointing
4. **Ring Topology** - P2P network between workers for tensor all-reduce operations
5. **Resource Locking** - Client app locks memory/compute with cooldown period for stability

### What Makes This Unique

- ✅ **Cooperative, not competitive** - Workers pool resources, share benefits
- ✅ **Tensor parallelism** - All workers participate in every layer (not pipeline)
- ✅ **Ring all-reduce** - Optimal bandwidth utilization, same as NCCL/Horovod
- ✅ **SSH container access** - Executors get familiar OpenAI-compatible API
- ✅ **Fair allocation** - Contribute X% resources → use X% capacity
- ✅ **Fault tolerant** - Checkpointing every N tokens enables recovery

---

## The Core Concept: Cooperative Tensor Parallelism

### Traditional Approach (Doesn't Work)

```
Single Device:
- Needs: 64GB GPU memory
- Reality: Consumer devices have 8-16GB
- Result: Can't run Llama-70B locally ❌
```

### Mesh Approach (Works!)

```
10 Devices in Cooperative Pool:
- Device 1: 7GB locked → Holds columns 0-10% of ALL weight matrices
- Device 2: 7GB locked → Holds columns 10-20% of ALL weight matrices
- Device 3: 7GB locked → Holds columns 20-30% of ALL weight matrices
- ...
- Device 10: 7GB locked → Holds columns 90-100% of ALL weight matrices

Total Pool: 70GB available
Can run: 64GB Llama-70B model ✅

Each inference pass:
1. All 10 devices receive same input
2. Each computes partial matrix multiplication (their columns only)
3. Ring all-reduce combines results
4. All devices have full activations
5. Repeat for next layer (70 layers total)
```

### Why Tensor Parallelism (Not Pipeline Parallelism)

**Pipeline Parallelism (Slower):**
```
Input → Device 1 (layers 1-7)  → 50ms network
      → Device 2 (layers 8-14)  → 50ms network
      → Device 3 (layers 15-21) → 50ms network
      → ...
      → Device 10 (layers 64-70) → 50ms network

Latency: 10 hops × 50ms = 500ms per token
Total: 500ms × 100 tokens = 50 seconds ❌ TOO SLOW
```

**Tensor Parallelism (Faster):**
```
Each Layer (all 70 layers):
1. All devices receive input simultaneously      → 0ms (broadcast)
2. Each computes partial matmul (their columns)  → 50ms (parallel)
3. Ring all-reduce combines results              → 40ms (18 steps × 2ms)
4. All devices have complete activations         → ready for next layer

Latency: (50ms compute + 40ms all-reduce) × 70 layers = 6.3 seconds
Per token: ~6.3s / 100 tokens = 63ms per token ✅ FAST ENOUGH
```

**Tensor parallelism is 8x faster because work happens in parallel, not sequentially.**

---

## Architecture Overview

### Three-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROL PLANE                             │
│  • Job queue and distribution                                │
│  • Worker ring topology management                           │
│  • Health monitoring (heartbeats)                            │
│  • Checkpoint coordination                                   │
│  • Executor container orchestration                          │
│  • Credit accounting                                         │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼ (topology mgmt)                    ▼ (container access)
┌─────────────────────────────┐    ┌──────────────────────────┐
│    WORKER RING (P2P)        │    │   EXECUTOR CONTAINERS    │
│                             │    │                          │
│  Device 1 ←→ Device 2       │    │  Container 1:            │
│      ↑            ↓         │    │  - SSH access (port 2222)│
│  Device 10 ←→ Device 3      │    │  - API client (OpenAI)   │
│      ↑            ↓         │    │  - Credit quota          │
│  Device 9  ←→ Device 4      │    │                          │
│      ↑            ↓         │    │  Container 2: ...        │
│  Device 8  ←→ ... ←→ D5     │    │  Container 3: ...        │
│                             │    │                          │
│  Each device:               │    │  Executors submit jobs:  │
│  - 7GB locked memory        │    │  curl localhost:8080/v1/ │
│  - Model shard (10% cols)   │    │     chat/completions     │
│  - Ring all-reduce          │    │                          │
└─────────────────────────────┘    └──────────────────────────┘
         │                                    │
         └────────────────┬───────────────────┘
                          │
                   Inference Flow:
         Executor → Control Plane → Worker Ring → Result
```

### Data Plane vs Control Plane Separation

**Data Plane (Ring P2P Network):**
- Direct peer-to-peer connections between adjacent workers
- Tensor passing for all-reduce operations
- Model weight storage and activation buffers
- **NOT routed through control plane** (direct for lowest latency)
- Uses libp2p DCUTR for NAT traversal, falls back to relay

**Control Plane (Centralized Coordinator):**
- Job queue and distribution to worker ring
- Worker registration and health monitoring
- Ring topology updates (device join/leave)
- Checkpoint coordination and storage
- Executor container lifecycle management
- Credit balance and quota enforcement
- **Only control messages, NOT tensor data**

**Critical Design Principle:** Control plane orchestrates, but stays out of the hot path for tensor operations.

---

## Worker Model: Cooperative Resource Contribution

### What is a Worker?

A **worker** is a device that contributes locked compute resources to the cooperative pool and participates in distributed inference.

```rust
struct Worker {
    device_id: DeviceId,

    // Resource contribution
    contributed_memory: u64,        // 7GB locked
    contributed_compute: f32,       // 75% of GPU utilization
    lock_status: LockStatus,        // Locked, PendingUnlock
    cooldown_period: Duration,      // 24 hours minimum

    // Ring topology
    ring_position: u32,             // 0-9 (position in ring)
    left_neighbor: DeviceId,        // Device 9 (if position=0)
    right_neighbor: DeviceId,       // Device 1 (if position=0)

    // Model shard
    shard: ModelShard {
        model_id: "llama-70b",
        column_range: (0, 10),      // Columns 0-10% of weight matrices
        memory_usage: 6.4GB,        // Actual shard size
    },

    // Credit economy
    contribution_percent: f64,      // 10% of pool capacity
    credit_balance: i64,            // +1000 credits/hour contributed
}
```

### Resource Locking and Memory Pinning

Workers lock memory to ensure stable pool operation:

```rust
// Client app resource manager
struct ResourceManager {
    total_memory: u64,              // 10 GB detected by client
    user_allocated: u64,            // 7 GB (user sets slider to 70%)
    locked_memory: u64,             // 7.5 GB (7GB + 7% safety buffer)
    lock_timestamp: SystemTime,
    cooldown_period: Duration,      // 24 hours
}

impl ResourceManager {
    // Lock memory for pool contribution
    fn lock_resources(&mut self) -> Result<()> {
        // Add 7% safety buffer to prevent OOM
        let buffer = (self.user_allocated as f64 * 1.07) as u64;
        self.locked_memory = buffer;

        // Pin memory pages to prevent swapping (critical for performance)
        #[cfg(target_os = "linux")]
        unsafe {
            libc::mlock(
                self.memory_region.as_ptr() as *const libc::c_void,
                self.locked_memory as libc::size_t
            );
        }

        #[cfg(target_os = "macos")]
        unsafe {
            libc::mlock(
                self.memory_region.as_ptr() as *const libc::c_void,
                self.locked_memory as libc::size_t
            );
        }

        #[cfg(target_os = "windows")]
        unsafe {
            windows_sys::Win32::System::Memory::VirtualLock(
                self.memory_region.as_ptr() as *mut _,
                self.locked_memory as usize
            );
        }

        self.lock_timestamp = SystemTime::now();
        info!("Locked {}GB memory for pool contribution", self.locked_memory / 1_000_000_000);
        Ok(())
    }

    // Unlock requires cooldown period (prevents churn)
    fn request_unlock(&mut self) -> Result<()> {
        let time_locked = SystemTime::now()
            .duration_since(self.lock_timestamp)?;

        if time_locked < self.cooldown_period {
            let remaining = self.cooldown_period - time_locked;
            return Err(MeshError::CooldownActive {
                remaining_hours: remaining.as_secs() / 3600,
            });
        }

        // Notify control plane 24 hours before leaving
        // Allows pool to redistribute shard to replacement device
        notify_pending_unlock(self.device_id).await?;

        Ok(())
    }
}
```

**Why cooldown is critical:**
- Pool needs predictable resource availability
- Shard redistribution takes time (download 6.4GB to new device)
- Prevents thrashing (lock/unlock/lock spam)
- 24-hour notice allows graceful handoff

### Worker Lifecycle

```
1. Join Pool:
   ├─ User sets contribution (7GB via slider)
   ├─ Client locks memory with buffer
   ├─ Device registers with control plane
   ├─ Control plane assigns ring position
   └─ Downloads model shard (column range)

2. Active Contribution:
   ├─ Participate in ring all-reduce for every inference
   ├─ Send heartbeats every 5 seconds
   ├─ Maintain P2P connections to left/right neighbors
   ├─ Store checkpoints locally
   └─ Earn credits proportional to contribution

3. Leave Pool:
   ├─ Request unlock (requires 24h cooldown elapsed)
   ├─ Control plane finds replacement device
   ├─ Replacement downloads shard
   ├─ Graceful handoff (replacement ready before departure)
   └─ Memory unlocked, device deregistered
```

---

## Executor Model: Consuming Inference Capacity

### What is an Executor?

An **executor** is a user who SSH's into a Docker container to access the cooperative pool's inference API. Executors can be external users OR workers using their earned allocation.

```rust
struct Executor {
    user_id: UserId,

    // Container access
    container_id: ContainerId,
    ssh_port: u16,                  // 2222, 2223, etc.
    ssh_public_key: Vec<u8>,

    // Resource quota
    credit_balance: i64,            // Available credits
    allocation_percent: f64,        // If worker: contribution_percent, else: purchased
    max_requests_per_hour: u32,     // Rate limit
    max_concurrent_jobs: u32,       // Concurrency limit (default: 1)

    // Relationship to worker pool
    is_worker: bool,                // True if also contributing resources
    worker_device_id: Option<DeviceId>,
}
```

### SSH Container Architecture

Each executor gets an isolated Docker container with API access:

```rust
// Container lifecycle management
async fn spawn_executor_container(executor: Executor) -> Result<ExecutorContainer> {
    // Build container with SSH server + API client
    let container = docker.create_container(CreateContainerOptions {
        image: "meshnet/executor:latest",
        exposed_ports: vec![2222],  // SSH port
        env: vec![
            format!("EXECUTOR_ID={}", executor.user_id),
            format!("CREDIT_BALANCE={}", executor.credit_balance),
            format!("WORKER_POOL_API=http://control-plane:8080/v1"),
            format!("MAX_RPH={}", executor.max_requests_per_hour),
        ],
        volumes: vec![
            // Mount SSH keys
            format!("{}:/home/executor/.ssh/authorized_keys", executor.ssh_key_path),
        ],
    }).await?;

    docker.start_container(&container.id).await?;

    info!("Spawned executor container {} on port {}", container.id, executor.ssh_port);

    Ok(ExecutorContainer {
        container_id: container.id,
        ssh_port: executor.ssh_port,
        api_endpoint: "http://localhost:8080".to_string(),
    })
}
```

### Executor Usage Example

```bash
# Executor SSH's into their container
ssh -p 2222 executor@mesh.example.com

# Inside container, interact with OpenAI-compatible API
$ curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-70b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 500
  }'

# Response streamed from worker ring
{
  "id": "job-123",
  "model": "llama-70b",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Quantum computing leverages quantum mechanics..."
    }
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 87,
    "total_tokens": 92,
    "credits_charged": 18
  }
}

# Can also run Python scripts, Jupyter notebooks, etc.
$ python my_inference_script.py
```

### Fair Allocation Model

**If executor is also a worker (common case):**
```
I contribute 7GB out of 70GB pool = 10% contribution
→ I earn 10% allocation of inference capacity
→ I can submit jobs using up to 10% of pool throughput
→ Fair and incentive-aligned!
```

**If executor is external user:**
```
I purchase 5% allocation for $50/month
→ I get 5% of pool throughput
→ Credits deducted per token generated
→ Workers earn credits proportional to participation
```

---

## Ring Topology and Tensor Parallelism

### Ring All-Reduce Algorithm

The **ring all-reduce** algorithm is the core of efficient tensor parallelism. It's the same algorithm NCCL, Horovod, and DeepSpeed use for multi-GPU training.

```rust
// Simplified ring all-reduce for tensor parallelism
async fn ring_all_reduce(
    workers: &WorkerRing,
    partial_result: Tensor,  // My partial matmul result
) -> Result<Tensor> {
    let n = workers.len();           // 10 devices
    let my_pos = workers.my_position(); // 0-9

    // Split tensor into n chunks
    let mut chunks = partial_result.chunk(n);

    // PHASE 1: Reduce-Scatter (n-1 steps)
    // Each device accumulates one chunk
    for step in 0..(n-1) {
        let send_idx = (my_pos - step + n) % n;
        let recv_idx = (my_pos - step - 1 + n) % n;

        // Send chunk to right neighbor, receive from left
        let received = workers.send_to_right_recv_from_left(
            chunks[send_idx].clone()
        ).await?;

        // Accumulate (sum) received chunk
        chunks[recv_idx] = chunks[recv_idx].add(&received)?;
    }

    // After reduce-scatter: each device has one fully reduced chunk

    // PHASE 2: All-Gather (n-1 steps)
    // Distribute all chunks to all devices
    for step in 0..(n-1) {
        let send_idx = (my_pos - step + 1 + n) % n;
        let recv_idx = (my_pos - step + n) % n;

        // Send fully reduced chunk around ring
        chunks[recv_idx] = workers.send_to_right_recv_from_left(
            chunks[send_idx].clone()
        ).await?;
    }

    // All devices now have identical complete tensor
    Ok(Tensor::concat(chunks))
}
```

### Why Ring is Optimal

**Bandwidth efficiency:**
```
Each device sends/receives: 2 × (N-1)/N × tensor_size
For N=10: Each device transfers ~1.8× tensor size
This is OPTIMAL (no redundant transfers!)

Compare to naive all-to-all: N × (N-1) transfers = 90 transfers
Ring: 2 × (N-1) = 18 total steps
```

**Network utilization:**
```
All links used simultaneously:
Device 0 → Device 1 (sending)
Device 1 → Device 2 (sending)
...
Device 9 → Device 0 (sending)

Full bisection bandwidth utilized!
```

**Latency analysis:**
```
Each step:
- Network RTT: 5ms (P2P connection)
- Tensor transfer: 3ms (chunk_size / bandwidth)
- Total: 8ms per step

Total steps: 2 × (N-1) = 18 steps
Total latency: 18 × 8ms = 144ms per all-reduce

For 70 layers: 70 × 144ms = 10 seconds
Plus compute: 70 × 50ms = 3.5 seconds
Total inference: ~13.5 seconds for first token
```

### Tensor Parallelism Forward Pass

```rust
// Single transformer layer with tensor parallelism
async fn forward_pass_tensor_parallel(
    workers: &WorkerRing,
    input: Tensor,          // Shape: [batch, seq_len, hidden_dim]
    layer_idx: usize,
) -> Result<Tensor> {
    // Each worker holds different columns of weight matrix
    let my_shard = workers.get_my_shard(layer_idx);

    // Step 1: Each worker computes partial matrix multiplication
    // Input: [batch, seq, 4096]
    // My weights: [4096, 819]  (10% of 8192 columns)
    // Partial output: [batch, seq, 819]
    let partial_output = input.matmul(&my_shard.weights)?;
    let start = Instant::now();

    // Step 2: Ring all-reduce to combine results
    // All workers contribute their partial outputs
    // Result: [batch, seq, 8192] (full output)
    let full_output = ring_all_reduce(workers, partial_output).await?;

    debug!("All-reduce latency: {:?}", start.elapsed());

    // Step 3: Apply activation (each worker does this identically)
    let activated = full_output.gelu()?;

    Ok(activated)
}

// Full model forward pass
async fn generate_token(
    workers: &WorkerRing,
    input_tokens: Vec<u32>,
) -> Result<u32> {
    let mut hidden = embed_tokens(&input_tokens)?;

    // Process all 70 layers with tensor parallelism
    for layer_idx in 0..70 {
        hidden = forward_pass_tensor_parallel(
            workers,
            hidden,
            layer_idx
        ).await?;
    }

    // Final layer: sample next token
    let logits = workers.get_my_shard(70).weights.matmul(&hidden)?;
    let full_logits = ring_all_reduce(workers, logits).await?;
    let next_token = sample_token(&full_logits)?;

    Ok(next_token)
}
```

### Network Topology: P2P Ring Connections

```rust
// Worker maintains two P2P connections (left and right neighbors)
struct WorkerRingConnections {
    my_position: u32,

    // Left neighbor (receive from)
    left_peer_id: PeerId,
    left_connection: Connection,

    // Right neighbor (send to)
    right_peer_id: PeerId,
    right_connection: Connection,
}

// Establish ring topology on pool initialization
async fn establish_ring_topology(workers: Vec<Worker>) -> Result<()> {
    let n = workers.len();

    for (i, worker) in workers.iter().enumerate() {
        let left_idx = (i + n - 1) % n;
        let right_idx = (i + 1) % n;

        // Connect to left neighbor (for receiving)
        let left_addr = workers[left_idx].relay_addr.clone();
        worker.connect_to_neighbor(left_addr, Direction::Left).await?;

        // Connect to right neighbor (for sending)
        let right_addr = workers[right_idx].relay_addr.clone();
        worker.connect_to_neighbor(right_addr, Direction::Right).await?;

        info!(
            "Worker {} connected in ring: {} ← [{}] → {}",
            worker.device_id,
            workers[left_idx].device_id,
            worker.device_id,
            workers[right_idx].device_id
        );
    }

    Ok(())
}

// Use DCUTR for direct P2P, fallback to relay if needed
async fn connect_to_neighbor(
    &mut self,
    neighbor_addr: Multiaddr,
    direction: Direction,
) -> Result<Connection> {
    // Try direct connection first (DCUTR)
    match self.swarm.dial(neighbor_addr.clone()).await {
        Ok(conn) => {
            info!("Direct P2P connection established to neighbor");
            Ok(conn)
        },
        Err(_) => {
            // Fallback to relay if direct connection fails
            warn!("Direct connection failed, using relay");
            let relay_circuit = format!("{}/p2p-circuit/{}",
                self.relay_addr, neighbor_addr);
            self.swarm.dial(relay_circuit).await
        }
    }
}
```

---

## Fault Tolerance: Checkpointing Strategy

### Why Checkpointing is Critical

**Problem:** Device failures during long-running generation (100+ tokens)

```
Without checkpointing:
1. Generate 80 tokens (13 seconds × 80 = 17 minutes)
2. Device 5 crashes
3. ENTIRE JOB LOST ❌
4. Must restart from beginning

With checkpointing:
1. Generate 80 tokens, checkpoint every 50 tokens
2. Device 5 crashes
3. Load checkpoint from token 50
4. Redistribute Device 5's shard to Device 11
5. Resume from token 50 ✅
6. Only lost 30 tokens (6 minutes)
```

### Checkpoint Implementation

```rust
// Checkpoint manager (runs on control plane)
struct CheckpointManager {
    checkpoint_interval: u32,       // Every 50 tokens
    checkpoint_storage: PathBuf,    // ~/.meshnet/checkpoints/
    max_checkpoints: u32,           // Keep last 5 checkpoints
}

// Distributed checkpointing (each worker saves their portion)
async fn checkpoint_inference_state(
    job_id: JobId,
    token_idx: u32,
    workers: &WorkerRing,
) -> Result<()> {
    let checkpoint_id = format!("{}_{}", job_id, token_idx);

    // Each worker saves their portion in parallel
    for worker in workers.iter() {
        tokio::spawn(async move {
            worker.save_checkpoint(CheckpointData {
                checkpoint_id: checkpoint_id.clone(),
                kv_cache: worker.get_kv_cache(),      // Cached attention keys/values
                generated_tokens: worker.get_tokens(), // Tokens generated so far
                rng_state: worker.get_rng_state(),     // For deterministic sampling
            }).await
        });
    }

    // Control plane saves metadata
    save_checkpoint_metadata(CheckpointMetadata {
        checkpoint_id,
        job_id,
        token_idx,
        timestamp: SystemTime::now(),
        worker_count: workers.len(),
    }).await?;

    info!("Checkpoint {} saved at token {}", checkpoint_id, token_idx);
    Ok(())
}

// Generation with automatic checkpointing
async fn generate_with_checkpointing(
    workers: &WorkerRing,
    prompt: &str,
    max_tokens: u32,
) -> Result<String> {
    let job_id = JobId::new();
    let mut generated_tokens = Vec::new();

    for token_idx in 0..max_tokens {
        // Generate next token via tensor-parallel forward pass
        let next_token = generate_token(workers, &generated_tokens).await?;
        generated_tokens.push(next_token);

        // Checkpoint every 50 tokens
        if token_idx % 50 == 0 && token_idx > 0 {
            checkpoint_inference_state(job_id, token_idx, workers).await?;
        }
    }

    // Decode tokens to string
    let text = decode_tokens(&generated_tokens)?;
    Ok(text)
}
```

### Recovery from Device Failure

```rust
// Control plane monitors worker health via heartbeats
async fn handle_worker_failure(
    failed_worker: Worker,
    active_jobs: Vec<JobId>,
) -> Result<()> {
    error!("Worker {} failed, initiating recovery", failed_worker.device_id);

    // Step 1: Find replacement worker
    let replacement = find_replacement_worker(failed_worker.shard.memory_usage).await?;

    // Step 2: Download model shard to replacement
    let shard_source = get_shard_backup_location(&failed_worker.shard)?;
    replacement.download_shard(shard_source).await?;

    // Step 3: Update ring topology
    let left_neighbor = failed_worker.left_neighbor;
    let right_neighbor = failed_worker.right_neighbor;

    reconnect_ring(left_neighbor, replacement.device_id, right_neighbor).await?;

    // Step 4: Recover active jobs from checkpoints
    for job_id in active_jobs {
        let checkpoint = load_latest_checkpoint(job_id).await?;

        // Resume inference from checkpoint
        resume_inference(job_id, checkpoint, workers).await?;

        info!("Job {} resumed from checkpoint at token {}",
            job_id, checkpoint.token_idx);
    }

    Ok(())
}

// Replacement worker downloads shard
async fn download_shard(&mut self, shard_source: ShardSource) -> Result<()> {
    info!("Downloading shard {} from backup", shard_source.shard_id);

    // Download model weights (6.4GB)
    let weights = download_from_backup(shard_source).await?;

    // Load into GPU memory (pre-allocated locked region)
    self.load_weights_to_gpu(weights).await?;

    // Verify checksum
    if !verify_shard_checksum(&weights, &shard_source.checksum) {
        return Err(MeshError::ShardCorrupted);
    }

    info!("Shard loaded successfully, ready for inference");
    Ok(())
}
```

### Checkpoint Storage Strategy

**Distributed storage (each worker stores checkpoints):**
- ✅ No central storage bottleneck
- ✅ Workers already have the data in memory
- ✅ Fast checkpoint saves (write to local disk)
- ❌ Must replicate checkpoints (3x redundancy)

**Control plane storage:**
- ✅ Centralized recovery point
- ✅ Easy to manage
- ❌ Bandwidth bottleneck (all workers → control plane)
- ❌ Single point of failure

**Recommendation:** Hybrid approach
- Workers store checkpoints locally (fast writes)
- Replicate to 2 other workers (3x redundancy)
- Control plane stores metadata only (job_id, token_idx, worker_list)

---

## Credit Economy and Fair Allocation

### Contribution-Based Allocation

```rust
struct CreditSystem {
    pool_capacity: u64,  // 70GB total
}

impl CreditSystem {
    // Calculate fair allocation based on contribution
    fn calculate_allocation(
        &self,
        worker: &Worker
    ) -> ResourceQuota {
        let contribution_percent =
            worker.contributed_memory as f64 / self.pool_capacity as f64;

        ResourceQuota {
            // If I contribute 10% of pool, I can use 10% of throughput
            max_requests_per_hour: (1000.0 * contribution_percent) as u32,
            max_concurrent_jobs: 1,  // Sequential for MVP
            priority: Priority::Normal,
        }
    }

    // Credit earning (workers earn by contributing)
    fn earn_credits(
        &self,
        worker: &Worker,
        duration: Duration,
    ) -> i64 {
        let hours = duration.as_secs_f64() / 3600.0;
        let contribution_percent =
            worker.contributed_memory as f64 / self.pool_capacity as f64;

        // Base rate: 1000 credits/hour for 100% contribution
        (1000.0 * contribution_percent * hours) as i64
    }

    // Credit spending (executors spend credits for inference)
    fn calculate_cost(
        &self,
        job: &InferenceJob,
        result: &InferenceResult,
    ) -> i64 {
        // Cost model: credits per token generated
        let tokens_generated = result.completion_tokens;
        let credits_per_token = 1;  // Configurable

        tokens_generated as i64 * credits_per_token
    }
}
```

### Example Scenarios

**Scenario 1: Worker using their own allocation**
```
Alice contributes 7GB (10% of pool)
→ Earns 100 credits/hour
→ Can use 10% of throughput (100 requests/hour)
→ Submits job: 87 tokens generated → 87 credits spent
→ Net: (100 - 87) = +13 credits/hour
→ Fair: Alice contributes, Alice uses proportionally ✅
```

**Scenario 2: Worker with surplus credits**
```
Bob contributes 14GB (20% of pool)
→ Earns 200 credits/hour
→ Only uses 50 credits/hour (light usage)
→ Surplus: 150 credits/hour accumulate
→ Can use surplus later OR trade with other users
```

**Scenario 3: External executor purchasing allocation**
```
Carol (not a worker) purchases 5% allocation
→ Pays $50/month → receives 500 credits
→ Can submit up to 50 requests/hour
→ Credits spent: 1 credit per token
→ Workers earn credits proportional to participation
```

---

## Technical Implementation Details

### How Current Codebase Evolves

**What we've already built (IMPLEMENTATION.md):**

✅ `agent/src/network/mesh_swarm.rs` - P2P connectivity (use for ring topology)
✅ `agent/src/executor/job_runner.rs` - Job execution (extend for tensor ops)
✅ `control-plane/` - Worker registration and heartbeats (extend for ring mgmt)
✅ `relay-server/` - NAT traversal (fallback for P2P ring connections)
✅ Job protocol - Request-response (extend for tensor passing)

**What needs to be added:**

### 1. Ring Topology Manager (Control Plane)

```rust
// control-plane/src/topology/ring_manager.rs
struct RingTopologyManager {
    workers: Vec<Worker>,           // All workers in pool
    ring_sequence: Vec<DeviceId>,   // Ordered ring positions
    model_registry: ModelRegistry,  // Which model shards exist
}

impl RingTopologyManager {
    // Worker joins pool
    async fn add_worker(&mut self, worker: Worker) -> Result<RingPosition> {
        // Find position in ring (append to end initially)
        let position = self.workers.len() as u32;

        // Assign model shard (which columns to download)
        let shard = self.assign_shard(&worker, position)?;

        // Update ring connections
        self.update_ring_connections(position).await?;

        Ok(RingPosition {
            position,
            shard,
            left_neighbor: self.get_neighbor(position, -1),
            right_neighbor: self.get_neighbor(position, 1),
        })
    }

    // Assign model shard to worker
    fn assign_shard(&self, worker: &Worker, position: u32) -> Result<ModelShard> {
        let n = self.workers.len() + 1;
        let columns_per_shard = 8192 / n;  // Total columns divided by workers

        let start_col = position * columns_per_shard;
        let end_col = start_col + columns_per_shard;

        Ok(ModelShard {
            model_id: "llama-70b".to_string(),
            column_range: (start_col, end_col),
            layer_count: 70,
            estimated_memory: worker.contributed_memory,
        })
    }
}
```

### 2. Tensor Passing Protocol (Agent)

```rust
// agent/src/network/tensor_protocol.rs
use libp2p::request_response::{Codec, RequestResponseProtocol};

// Extend job protocol for tensor passing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMessage {
    pub job_id: JobId,
    pub layer_idx: u32,
    pub step_idx: u32,           // All-reduce step number
    pub tensor_data: Vec<f32>,   // Actual tensor chunk
    pub shape: Vec<usize>,       // Tensor dimensions
}

// Ring communication primitive
async fn send_to_right_recv_from_left(
    &mut self,
    tensor: Tensor,
) -> Result<Tensor> {
    // Send to right neighbor (non-blocking)
    let send_msg = TensorMessage {
        job_id: self.current_job,
        layer_idx: self.current_layer,
        step_idx: self.all_reduce_step,
        tensor_data: tensor.to_vec(),
        shape: tensor.shape(),
    };

    self.swarm.send_to_peer(self.right_neighbor, send_msg)?;

    // Receive from left neighbor (blocking)
    let recv_msg = self.swarm.recv_from_peer(self.left_neighbor).await?;

    // Deserialize tensor
    let received_tensor = Tensor::from_vec(
        recv_msg.tensor_data,
        recv_msg.shape
    )?;

    Ok(received_tensor)
}
```

### 3. Model Sharding and Loading (Agent)

```rust
// agent/src/model/shard_loader.rs
struct ShardLoader {
    model_id: String,
    column_range: (u32, u32),
    cache_dir: PathBuf,
}

impl ShardLoader {
    // Download shard from backup storage or peer
    async fn download_shard(&self) -> Result<ModelWeights> {
        let shard_url = format!(
            "https://models.mesh.network/{}/shard_{}_{}.safetensors",
            self.model_id,
            self.column_range.0,
            self.column_range.1
        );

        info!("Downloading shard from {}", shard_url);

        // Stream download with progress
        let weights = download_with_progress(&shard_url).await?;

        // Cache locally for future use
        let cache_path = self.cache_dir.join(format!(
            "{}_shard_{}_{}.safetensors",
            self.model_id,
            self.column_range.0,
            self.column_range.1
        ));

        tokio::fs::write(&cache_path, &weights).await?;

        Ok(ModelWeights::from_safetensors(weights)?)
    }

    // Load shard into locked GPU memory
    async fn load_to_gpu(&self, weights: ModelWeights) -> Result<()> {
        // Verify we have enough locked memory
        if weights.size_bytes() > self.contributed_memory {
            return Err(MeshError::InsufficientMemory);
        }

        // Load weights to GPU (use candle/burn for tensor ops)
        self.gpu_allocator.load_weights(weights).await?;

        info!("Shard loaded to GPU: {} GB", weights.size_bytes() / 1_000_000_000);
        Ok(())
    }
}
```

### 4. Resource Manager UI (Client App)

```rust
// Conceptual UI for resource locking
struct ResourceManagerUI {
    total_memory: u64,
    allocated_slider: f64,  // 0.0 to 1.0
}

impl ResourceManagerUI {
    fn render(&self) {
        // Slider: 0% to 100% of available memory
        // User drags slider to 70% → 7GB allocated

        println!("Available Memory: {}GB", self.total_memory / 1_000_000_000);
        println!("Allocated: {}%", self.allocated_slider * 100.0);
        println!("Locked: {}GB (+ buffer)", self.allocated_memory());
        println!("Cooldown: 24 hours to unlock");

        // Visual feedback
        draw_memory_bar(self.allocated_slider);
    }

    fn allocated_memory(&self) -> u64 {
        (self.total_memory as f64 * self.allocated_slider) as u64
    }
}
```

---

## Comparison to Existing Systems

### vs. Ray (Distributed Compute Framework)

**Ray:**
- Task parallelism (distribute independent tasks)
- Centralized scheduler
- Not optimized for model parallelism
- Research/enterprise focus

**Mesh:**
- Tensor parallelism (distribute single model inference)
- Decentralized worker ring (P2P)
- Optimized for consumer devices
- Cooperative pooling (not task queue)

### vs. DeepSpeed / Megatron (Training Frameworks)

**DeepSpeed/Megatron:**
- ✅ Same tensor parallelism strategy (ring all-reduce)
- ✅ Same optimization (NCCL for GPU clusters)
- ❌ Requires high-bandwidth interconnect (NVLink, InfiniBand)
- ❌ Enterprise GPU clusters only

**Mesh:**
- ✅ Same tensor parallelism strategy
- ✅ Optimized for consumer devices over WAN
- ✅ Works with consumer internet (50 Mbps+)
- ✅ Fault tolerance via checkpointing (training doesn't need this)

### vs. Petals (Collaborative Inference)

**Petals:**
- ✅ Collaborative inference across devices
- ❌ Pipeline parallelism (sequential, high latency)
- ❌ Public swarm (untrusted participants)
- ❌ No resource locking (unstable pool)

**Mesh:**
- ✅ Tensor parallelism (parallel, lower latency)
- ✅ Private cooperative pools (trusted workers)
- ✅ Resource locking with cooldown (stable pool)
- ✅ Fair allocation based on contribution

---

## Technical Challenges and Solutions

### Challenge 1: Network Latency for All-Reduce

**Problem:** Ring all-reduce requires 18 steps × 70 layers = 1,260 network round-trips

**Solutions:**
1. **DCUTR for direct P2P** - Bypass relay when possible (5-10ms RTT)
2. **Tensor compression** - Quantize FP32 → FP16 or INT8 (2-4× smaller)
3. **Batching** - Combine multiple all-reduce operations
4. **Geographic pools** - Workers in same region/city (lower RTT)

**Target:** <100ms per all-reduce → <7s per token → acceptable for chat

### Challenge 2: Device Failures During Inference

**Problem:** Long-running generation (100 tokens = 10 minutes) vulnerable to failures

**Solutions:**
1. **Checkpointing every 50 tokens** - Only lose 30 seconds max
2. **Replacement worker pool** - Pre-warmed standby devices
3. **Shard replication** - Each shard stored on 3 workers (2 backups)
4. **Graceful degradation** - Reduce pool size if no replacement available

**Target:** 99% job completion rate (1% retry acceptable)

### Challenge 3: Memory Management and OOM

**Problem:** Workers must fit model shard + activations + KV cache in locked memory

**Solutions:**
1. **Strict memory budgeting** - Reserve 10% buffer for activations
2. **KV cache quantization** - INT8 KV cache (4× smaller)
3. **Offloading** - Spill KV cache to CPU if GPU full
4. **Dynamic sharding** - Redistribute if worker joins/leaves

**Target:** No OOM crashes, graceful rejection if insufficient memory

### Challenge 4: Synchronization Across Workers

**Problem:** All workers must stay synchronized for each layer

**Solutions:**
1. **Barrier synchronization** - Wait for all workers before next layer
2. **Timeout detection** - Mark worker as slow/failed if >2s delay
3. **Heartbeat monitoring** - Detect failures before timeout
4. **Leader election** - One worker coordinates barrier

**Target:** <100ms synchronization overhead per layer

---

## Implementation Roadmap

### Phase 1: Single-Pool Tensor Parallelism (Current Focus)

**Goal:** Build working tensor-parallel inference for 10-device pool

**Components:**
1. ✅ Worker registration and heartbeat (DONE via IMPLEMENTATION.md)
2. ✅ P2P connectivity with DCUTR (DONE via IMPLEMENTATION.md)
3. 🚧 Ring topology manager (control plane)
4. 🚧 Model sharding and distribution
5. 🚧 Ring all-reduce implementation
6. 🚧 Tensor-parallel forward pass
7. 🚧 Resource locking UI (client app)
8. 🚧 Checkpointing system

**Timeline:** 8-12 weeks

### Phase 2: Executor Containers and API

**Goal:** Add SSH container access for executors

**Components:**
1. Docker container orchestration
2. SSH key management
3. OpenAI-compatible API server
4. Credit quota enforcement
5. Rate limiting per executor

**Timeline:** 3-4 weeks

### Phase 3: Fault Tolerance and Production Hardening

**Goal:** 99% reliability for production use

**Components:**
1. Advanced checkpointing (distributed storage)
2. Replacement worker pool
3. Shard replication (3× redundancy)
4. Dynamic rebalancing (workers join/leave)
5. Monitoring and alerting

**Timeline:** 6-8 weeks

### Phase 4: Multi-Pool Discovery (Future)

**Goal:** Multiple cooperative pools can advertise capacity

**Components:**
1. Gossipsub for pool announcements
2. Pool discovery by model_id
3. Cross-pool reputation system
4. Load balancing across pools

**Timeline:** 4-6 weeks after Phase 3

---

## Success Criteria

### Phase 1 Success: Single Pool Working

- ✅ 10 devices form stable worker ring
- ✅ Each worker locks 7GB memory with 24h cooldown
- ✅ Model shard distributed (each worker has 10% columns)
- ✅ Ring all-reduce completes in <200ms
- ✅ Full inference: <15 seconds per token
- ✅ Checkpointing every 50 tokens
- ✅ 90% job completion rate (with retries)

### Phase 2 Success: Executor Access

- ✅ Executors can SSH into containers
- ✅ OpenAI-compatible API works (curl requests)
- ✅ Credit quota enforced (executor can't exceed allocation)
- ✅ Fair allocation: 10% contribution → 10% usage
- ✅ Rate limiting prevents abuse

### Phase 3 Success: Production Ready

- ✅ 99% job completion rate
- ✅ Device failure recovery <60 seconds
- ✅ Shard replication prevents data loss
- ✅ Dynamic rebalancing (workers join/leave gracefully)
- ✅ Monitoring dashboard shows pool health

---

## Conclusion

Mesh is a **tensor-parallel distributed inference system** that enables cooperative pooling of compute resources. By using **ring all-reduce** (the same algorithm as NCCL/Horovod), we achieve low-latency inference across consumer devices connected over the internet.

### Core Innovations

1. **Tensor parallelism over WAN** - Brings datacenter techniques to consumer devices
2. **Cooperative pooling** - Workers contribute resources, share benefits fairly
3. **Resource locking with cooldown** - Ensures pool stability (no churn)
4. **SSH container access** - Familiar OpenAI-compatible API for executors
5. **Checkpointing for fault tolerance** - Graceful recovery from device failures

### What Makes This Possible

- Ring topology minimizes network traffic (optimal bandwidth)
- P2P connections with DCUTR reduce latency (bypass relay)
- Resource locking ensures predictable capacity
- Tensor parallelism allows parallelization (vs. sequential pipeline)
- Fair allocation aligns incentives (contribute more, use more)

### Next Steps

The current `IMPLEMENTATION.md` roadmap is building the foundational execution layer (P2P connectivity, job protocol, worker registration). This document provides the architectural vision for evolving that foundation into a full tensor-parallel cooperative pooling system.

**This is not a marketplace. This is not a blockchain. This is a cooperative compute pool using state-of-the-art distributed inference techniques.**

Like DeepSpeed, but for consumer devices. Like Tailscale, but for AI inference.

---

**Questions or feedback?** See `IMPLEMENTATION.md` for tactical implementation steps.
