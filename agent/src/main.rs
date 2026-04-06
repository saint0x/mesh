//! Mesh AI Agent - Command Line Interface
//!
//! The Mesh AI Agent is a production distributed inference daemon.
//!
//! ## Commands
//!
//! ### Device
//! - `mesh device init`
//! - `mesh device start`
//! - `mesh device status`
//!
//! ### Ring
//! - `mesh ring join`
//! - `mesh ring leave`
//! - `mesh ring status`
//! - `mesh ring topology`
//! - `mesh ring shard`
//!
//! ### Jobs and Ledger
//! - `mesh job run`
//! - `mesh job status`
//! - `mesh job watch`
//! - `mesh ledger summary`
//! - `mesh ledger events`
//!
//! ### Resources and Pools
//! - `mesh resource lock`
//! - `mesh resource unlock`
//! - `mesh resource status`
//! - `mesh pool create`
//! - `mesh pool join`
//! - `mesh pool list`
//! - `mesh pool peers`
//! - `mesh pool status`

mod ui;

use agent::pki::{
    DeviceKeyPair, MembershipRole, PeerCache, PoolConfig, PoolId, PoolMembershipCert,
};
use agent::{
    api::types::{PeerPunchPlan, RingTopologyResponse, WorkerInfo},
    build_direct_peer_candidates_from_records, format_bytes, init_production_logging,
    init_simple_logging, load_direct_candidate_seed_records, load_observed_reachability_addrs,
    parse_data_plane_endpoint, parse_memory_string, parse_tensor_plane_advertised_addr_env,
    parse_tensor_plane_bind_addr_env, persist_runtime_connectivity_state,
    select_direct_dial_addrs_from_candidates, set_selected_execution_provider,
    ConnectivityAttachmentKind, ConnectivityPath, ConnectivityStatus, DeviceConfig,
    DeviceConnectivityState, MeshSwarmBuilder, RegistrationClient, ResourceManager, TensorPlane,
    TensorPlaneConfig,
};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use libp2p::{Multiaddr, PeerId};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::debug;
use tracing::{error, info, warn};
use ui::cmd_ui;
use uuid::Uuid;

/// Mesh AI Agent - Distributed compute sharing network
#[derive(Parser, Debug)]
#[command(name = "mesh")]
#[command(about = "MeshNet production CLI", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Device lifecycle and daemon operations
    Device {
        #[command(subcommand)]
        command: DeviceCommands,
    },

    /// Resource commitment for ring participation
    Resource {
        #[command(subcommand)]
        command: ResourceCommands,
    },

    /// Model ring membership and topology
    Ring {
        #[command(subcommand)]
        command: RingCommands,
    },

    /// Distributed job submission and tracking
    Job {
        #[command(subcommand)]
        command: JobCommands,
    },

    /// Credit and ledger surfaces
    Ledger {
        #[command(subcommand)]
        command: LedgerCommands,
    },

    /// Pool management and LAN discovery
    Pool {
        #[command(subcommand)]
        command: PoolCommands,
    },

    /// Verify local setup and control-plane reachability
    Doctor,

    /// Launch the local Mesh UI
    Ui {
        /// UI port to serve on
        #[arg(long, default_value = "43110")]
        port: u16,

        /// Local API port to serve on
        #[arg(long, default_value = "43111")]
        api_port: u16,
    },
}

#[derive(Subcommand, Debug)]
enum DeviceCommands {
    /// Initialize device identity and register with the control plane
    Init {
        /// Network ID to join
        #[arg(short, long)]
        network_id: String,

        /// Device name
        #[arg(short = 'd', long, default_value = "My Device")]
        name: String,

        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,
    },
    /// Run the production agent daemon
    Start {
        /// Log level (trace, debug, info, warn, error)
        #[arg(short, long, default_value = "info")]
        log_level: String,
    },
    /// Show local device/runtime status
    Status,
}

#[derive(Subcommand, Debug)]
enum ResourceCommands {
    /// Lock resources for ring contribution
    Lock {
        /// Amount of memory to lock (for example `7GB`)
        #[arg(short, long)]
        memory: String,
    },
    /// Unlock resources after cooldown
    Unlock,
    /// Show resource lock status
    Status,
}

#[derive(Subcommand, Debug)]
enum RingCommands {
    /// Join a model ring
    Join {
        /// Model ID to serve
        #[arg(short, long)]
        model_id: String,

        /// Contributed memory for this ring membership (for example `8GB`)
        #[arg(short, long)]
        memory: Option<String>,
    },
    /// Leave the current ring membership
    Leave,
    /// Show this device's ring status
    Status,
    /// Show full ring topology
    Topology,
    /// Show this worker's shard assignment
    Shard,
}

#[derive(Subcommand, Debug)]
enum JobCommands {
    /// Submit a distributed inference job and watch it to completion
    Run {
        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Model ID
        #[arg(short, long, default_value = "tinyllama-1.1b-chat-v1.0")]
        model_id: String,

        /// Max tokens to generate
        #[arg(short = 'n', long, default_value = "100")]
        max_tokens: u32,

        /// Temperature (0.0-2.0)
        #[arg(short, long, default_value = "1.0")]
        temperature: f32,

        /// Top-p sampling threshold
        #[arg(long, default_value = "0.9")]
        top_p: f32,
    },
    /// Fetch one job status snapshot
    Status {
        /// Job ID
        #[arg(long)]
        job_id: String,
    },
    /// Watch a distributed job until it reaches a terminal state
    Watch {
        /// Job ID
        #[arg(long)]
        job_id: String,

        /// Poll interval in milliseconds
        #[arg(long, default_value = "2000")]
        poll_interval_ms: u64,
    },
    /// Show local inference runtime statistics
    Stats,
}

#[derive(Subcommand, Debug)]
enum LedgerCommands {
    /// Show ledger summary for the current network
    Summary {
        /// Optional job ID filter
        #[arg(long)]
        job_id: Option<String>,
    },
    /// List recent ledger events for the current network
    Events {
        /// Optional job ID filter
        #[arg(long)]
        job_id: Option<String>,

        /// Maximum number of events to print
        #[arg(long, default_value = "20")]
        limit: usize,
    },
}

#[derive(Subcommand, Debug)]
enum PoolCommands {
    /// Create a new pool and become admin
    Create {
        /// Pool name
        #[arg(short, long)]
        name: String,
    },
    /// Join an existing pool
    Join {
        /// Pool ID (hex)
        #[arg(long)]
        pool_id: String,

        /// Pool root public key (hex)
        #[arg(long)]
        pool_root_pubkey: String,

        /// Pool name (optional)
        #[arg(short, long)]
        name: Option<String>,
    },
    /// List configured pools
    List,
    /// Show discovered peers in a pool
    Peers {
        /// Pool ID (hex)
        #[arg(long)]
        pool_id: String,
    },
    /// Show pool/ring status for the current network
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Device { command } => match command {
            DeviceCommands::Init {
                network_id,
                name,
                control_plane_url,
            } => {
                init_simple_logging("warn")?;
                cmd_init(network_id, name, control_plane_url).await?;
            }
            DeviceCommands::Start { log_level } => {
                init_production_logging(&log_level, None)?;
                cmd_start().await?;
            }
            DeviceCommands::Status => {
                cmd_status().await?;
            }
        },
        Commands::Resource { command } => match command {
            ResourceCommands::Lock { memory } => {
                init_simple_logging("warn")?;
                cmd_lock_resources(memory).await?;
            }
            ResourceCommands::Unlock => {
                init_simple_logging("warn")?;
                cmd_unlock_resources().await?;
            }
            ResourceCommands::Status => {
                cmd_resource_status().await?;
            }
        },
        Commands::Ring { command } => match command {
            RingCommands::Join { model_id, memory } => {
                init_simple_logging("warn")?;
                cmd_join_ring(model_id, memory).await?;
            }
            RingCommands::Leave => {
                init_simple_logging("warn")?;
                cmd_leave_ring().await?;
            }
            RingCommands::Status => {
                cmd_ring_status().await?;
            }
            RingCommands::Topology => {
                cmd_topology().await?;
            }
            RingCommands::Shard => {
                cmd_shard_status().await?;
            }
        },
        Commands::Job { command } => match command {
            JobCommands::Run {
                prompt,
                model_id,
                max_tokens,
                temperature,
                top_p,
            } => {
                init_simple_logging("warn")?;
                cmd_inference(prompt, model_id, max_tokens, temperature, top_p).await?;
            }
            JobCommands::Status { job_id } => {
                cmd_job_status(&job_id, false, 0).await?;
            }
            JobCommands::Watch {
                job_id,
                poll_interval_ms,
            } => {
                cmd_job_status(&job_id, true, poll_interval_ms).await?;
            }
            JobCommands::Stats => {
                cmd_inference_stats().await?;
            }
        },
        Commands::Ledger { command } => match command {
            LedgerCommands::Summary { job_id } => {
                cmd_ledger_summary(job_id.as_deref()).await?;
            }
            LedgerCommands::Events { job_id, limit } => {
                cmd_ledger_events(job_id.as_deref(), limit).await?;
            }
        },
        Commands::Pool { command } => match command {
            PoolCommands::Create { name } => {
                init_simple_logging("warn")?;
                cmd_pool_create(name).await?;
            }
            PoolCommands::Join {
                pool_id,
                pool_root_pubkey,
                name,
            } => {
                init_simple_logging("warn")?;
                cmd_pool_join(pool_id, pool_root_pubkey, name).await?;
            }
            PoolCommands::List => {
                cmd_pool_list().await?;
            }
            PoolCommands::Peers { pool_id } => {
                cmd_pool_peers(pool_id).await?;
            }
            PoolCommands::Status => {
                init_simple_logging("warn")?;
                cmd_pool_status().await?;
            }
        },
        Commands::Doctor => {
            cmd_doctor().await?;
        }
        Commands::Ui { port, api_port } => {
            init_simple_logging("warn")?;
            cmd_ui(port, api_port).await?;
        }
    }

    Ok(())
}

/// Initialize device and register with control plane
async fn cmd_init(network_id: String, name: String, control_plane_url: String) -> Result<()> {
    println!("🔧 Initializing Mesh AI agent...\n");

    // Generate device configuration
    println!("📝 Generating device configuration...");
    let mut config =
        DeviceConfig::generate(name.clone(), network_id.clone(), control_plane_url.clone());

    println!("   Device ID: {}", config.device_id);
    println!("   Network ID: {}", network_id);
    println!("   Name: {}", name);
    println!("   Tier: {:?}", config.capabilities.tier);
    println!("   CPU Cores: {}", config.capabilities.cpu_cores);
    println!("   RAM: {} MB", config.capabilities.ram_mb);
    let available_providers = config
        .capabilities
        .execution_providers
        .iter()
        .filter(|provider| provider.available)
        .map(|provider| provider.kind.as_str())
        .collect::<Vec<_>>();
    println!(
        "   Execution Providers: {}",
        if available_providers.is_empty() {
            "none".to_string()
        } else {
            available_providers.join(", ")
        }
    );
    println!(
        "   Default Provider: {}",
        config.capabilities.default_execution_provider.as_str()
    );

    // Save configuration
    let config_path = DeviceConfig::default_path()?;
    config.save(&config_path)?;
    println!("\n✓ Configuration saved to: {}", config_path.display());

    // Register with control plane
    println!("\n🌐 Registering with control plane...");
    println!("   URL: {}", control_plane_url);

    let client = RegistrationClient::new(control_plane_url.clone())?;

    match client.register(&config).await {
        Ok(register_response) => {
            // Save certificate
            let signed_cert = register_response
                .certificate
                .as_ref()
                .context("Registration response missing certificate")?;
            config.save_certificate(&signed_cert)?;
            config.connectivity = register_response.connectivity.clone();
            config.save(&config_path)?;

            let cert_path = DeviceConfig::default_certificate_path()?;
            println!("✓ Registration successful!");
            println!("   Certificate saved to: {}", cert_path.display());
            if config.connectivity.attachments.is_empty() {
                println!("   Connectivity attachments: none advertised by control plane");
            } else {
                println!(
                    "   Preferred path: {:?}",
                    config.connectivity.preferred_path
                );
                println!("   Connectivity attachments:");
                for attachment in &config.connectivity.attachments {
                    println!(
                        "     - {:?} {} (priority {})",
                        attachment.kind, attachment.endpoint, attachment.priority
                    );
                }
            }
        }
        Err(e) => {
            error!(error = %e, "Registration failed");
            eprintln!("\n✗ Registration failed: {}", e);
            eprintln!(
                "   Make sure the control plane is running at {}",
                control_plane_url
            );
            return Err(e.into());
        }
    }

    println!("\n✅ Device initialized successfully!");
    println!("\nNext steps:");
    println!("  1. Start the agent:  mesh device start");
    println!("  2. Join a ring:      mesh ring join --model-id <model>");
    println!("  3. Submit inference: mesh job run --prompt \"Hello\"");

    Ok(())
}

fn listen_addrs_path() -> Result<std::path::PathBuf> {
    Ok(dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?
        .join(".meshnet")
        .join("listen_addrs.json"))
}

fn reset_persisted_listen_addrs() -> Result<()> {
    let path = listen_addrs_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, "[]")?;
    Ok(())
}

fn persist_listen_addr(local_peer_id: libp2p::PeerId, address: &Multiaddr) -> Result<()> {
    let path = listen_addrs_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let canonical = if address
        .iter()
        .any(|protocol| matches!(protocol, libp2p::multiaddr::Protocol::P2p(_)))
    {
        address.clone()
    } else {
        address
            .clone()
            .with(libp2p::multiaddr::Protocol::P2p(local_peer_id))
    };

    let mut addrs: Vec<String> = if path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };

    let canonical_str = canonical.to_string();
    if !addrs.iter().any(|addr| addr == &canonical_str) {
        addrs.push(canonical_str);
        std::fs::write(path, serde_json::to_string_pretty(&addrs)?)?;
    }

    Ok(())
}

fn persist_advertised_endpoint(endpoint: &str) -> Result<()> {
    let path = listen_addrs_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut addrs: Vec<String> = if path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&path)?)?
    } else {
        Vec::new()
    };

    if !addrs.iter().any(|addr| addr == endpoint) {
        addrs.push(endpoint.to_string());
        std::fs::write(path, serde_json::to_string_pretty(&addrs)?)?;
    }

    Ok(())
}

fn extract_tensor_addr(addrs: &[String]) -> Option<SocketAddr> {
    addrs
        .iter()
        .find_map(|addr| parse_data_plane_endpoint(addr))
}

fn load_local_listen_addrs() -> Vec<String> {
    let Some(path) = listen_addrs_path().ok() else {
        return Vec::new();
    };
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    serde_json::from_str(&content).unwrap_or_default()
}

fn print_direct_candidates_block(label: &str, candidates: &[agent::DirectPeerCandidate]) {
    println!("   {}: {}", label, candidates.len());
    for candidate in candidates.iter().take(5) {
        println!(
            "     - {:?}/{:?}/{:?} priority={} age={}s {}",
            candidate.scope,
            candidate.transport,
            candidate.source,
            candidate.priority,
            direct_candidate_age_seconds(candidate.last_updated_ms),
            candidate.endpoint
        );
    }
    if candidates.len() > 5 {
        println!("     - ... {} more", candidates.len() - 5);
    }
}

fn direct_candidate_age_seconds(last_updated_ms: u64) -> u64 {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    now_ms.saturating_sub(last_updated_ms) / 1000
}

fn build_worker_position_from_topology(
    topology: &RingTopologyResponse,
    device_id: &Uuid,
) -> Result<agent::inference::coordinator::WorkerPosition> {
    let self_worker = topology
        .workers
        .iter()
        .find(|worker| worker.device_id == device_id.to_string())
        .context("Current worker not found in ring topology")?;
    let left_worker = topology
        .workers
        .iter()
        .find(|candidate| candidate.device_id == self_worker.left_neighbor)
        .context("Left neighbor not found in ring topology")?;
    let right_worker = topology
        .workers
        .iter()
        .find(|candidate| candidate.device_id == self_worker.right_neighbor)
        .context("Right neighbor not found in ring topology")?;

    let left_peer_id = left_worker
        .peer_id
        .parse::<PeerId>()
        .context("Invalid left neighbor peer ID")?;
    let right_peer_id = right_worker
        .peer_id
        .parse::<PeerId>()
        .context("Invalid right neighbor peer ID")?;
    let left_punch_plan = find_peer_punch_plan(topology, device_id, &left_worker.device_id);
    let right_punch_plan = find_peer_punch_plan(topology, device_id, &right_worker.device_id);

    Ok(agent::inference::coordinator::WorkerPosition {
        position: self_worker.position,
        total_workers: topology.workers.len() as u32,
        left_neighbor: left_peer_id,
        left_neighbor_addrs: resolve_target_direct_addrs(
            topology,
            device_id,
            left_worker,
            left_peer_id,
        ),
        left_neighbor_punch_plan: left_punch_plan,
        left_neighbor_tensor_addr: extract_tensor_addr(&left_worker.listen_addrs)
            .context("Left neighbor has no dedicated tensor data-plane endpoint")?,
        right_neighbor: right_peer_id,
        right_neighbor_addrs: resolve_target_direct_addrs(
            topology,
            device_id,
            right_worker,
            right_peer_id,
        ),
        right_neighbor_punch_plan: right_punch_plan,
        right_neighbor_tensor_addr: extract_tensor_addr(&right_worker.listen_addrs)
            .context("Right neighbor has no dedicated tensor data-plane endpoint")?,
        shard_column_range: (self_worker.shard.column_start, self_worker.shard.column_end),
        shard_memory_bytes: self_worker.contributed_memory,
    })
}

fn find_peer_punch_plan(
    topology: &RingTopologyResponse,
    source_device_id: &Uuid,
    target_device_id: &str,
) -> Option<PeerPunchPlan> {
    topology
        .peer_punch_plans
        .iter()
        .find(|plan| {
            plan.source_device_id == source_device_id.to_string()
                && plan.target_device_id == target_device_id
        })
        .cloned()
}

fn resolve_target_direct_addrs(
    topology: &RingTopologyResponse,
    source_device_id: &Uuid,
    worker: &WorkerInfo,
    target_peer_id: PeerId,
) -> Vec<Multiaddr> {
    let candidates = find_peer_punch_plan(topology, source_device_id, &worker.device_id)
        .map(|plan| plan.target_candidates)
        .unwrap_or_else(|| worker.direct_candidates.clone());
    select_direct_dial_addrs_from_candidates(target_peer_id, &candidates)
}

async fn fetch_ring_topology(config: &DeviceConfig) -> Result<RingTopologyResponse> {
    let topology_url = format!(
        "{}/api/ring/topology?network_id={}",
        config.control_plane_url.trim_end_matches('/'),
        config.network_id
    );

    reqwest::Client::new()
        .get(&topology_url)
        .send()
        .await?
        .error_for_status()?
        .json::<RingTopologyResponse>()
        .await
        .map_err(Into::into)
}

fn resolve_ring_contributed_memory(config: &DeviceConfig, memory: Option<&str>) -> Result<u64> {
    if let Some(memory) = memory {
        return parse_memory_string(memory).context("Invalid memory format for ring join");
    }

    if let Ok(mut manager) = ResourceManager::new() {
        if manager.load_config().is_ok() && manager.is_locked() {
            return Ok(manager.locked_memory());
        }
    }

    if let Some(gpu_vram_mb) = config.capabilities.gpu_vram_mb {
        return Ok(gpu_vram_mb as u64 * 1024 * 1024);
    }

    Ok(config.capabilities.ram_mb as u64 * 1024 * 1024)
}

#[derive(Debug, serde::Deserialize)]
struct InferenceJobStatusResponse {
    success: bool,
    job_id: String,
    network_id: String,
    model_id: String,
    status: String,
    completion: Option<String>,
    completion_tokens: u32,
    execution_time_ms: u64,
    error: Option<String>,
    assignments: Vec<InferenceJobAssignmentStatus>,
}

#[derive(Debug, serde::Deserialize)]
struct InferenceJobAssignmentStatus {
    device_id: String,
    ring_position: u32,
    status: String,
    failure_reason: Option<String>,
    shard_column_start: u32,
    shard_column_end: u32,
    assigned_capacity_units: u32,
    execution_provider: String,
    execution_time_ms: u64,
}

#[derive(Debug, serde::Deserialize)]
struct LedgerSummaryResponse {
    network_id: String,
    job_id: Option<String>,
    total_events: u64,
    total_credits_earned: f64,
    total_credits_burned: f64,
    total_jobs_started: u64,
    total_jobs_completed: u64,
}

#[derive(Debug, serde::Deserialize)]
struct LedgerEventsResponse {
    events: Vec<LedgerEventRecord>,
}

#[derive(Debug, serde::Deserialize)]
struct LedgerEventRecord {
    event_id: String,
    network_id: String,
    event_type: String,
    job_id: Option<String>,
    device_id: Option<String>,
    credits_amount: Option<f64>,
    metadata: serde_json::Value,
    created_at: String,
}

fn control_plane_client() -> reqwest::Client {
    reqwest::Client::new()
}

fn control_plane_url(config: &DeviceConfig, path: &str) -> String {
    format!("{}{}", config.control_plane_url.trim_end_matches('/'), path)
}

/// Run agent daemon
async fn cmd_start() -> Result<()> {
    println!("🚀 Starting Mesh AI agent daemon...\n");

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    info!(
        device_id = %config.device_id,
        network_id = %config.network_id,
        name = %config.name,
        "Loaded device configuration"
    );

    println!("📋 Device: {} ({})", config.name, config.device_id);
    println!("   Network: {}", config.network_id);
    println!("   Tier: {:?}", config.capabilities.tier);
    let selected_provider = config.resolve_execution_provider()?;
    set_selected_execution_provider(selected_provider)?;
    println!("   Execution Provider: {}", selected_provider.as_str());

    let runtime_endpoint = config.connectivity.runtime_endpoint()?;

    reset_persisted_listen_addrs().context("Failed to reset persisted listen addresses")?;

    println!("\n🌐 Initializing mesh transport...");
    match &runtime_endpoint {
        Some(endpoint) => println!("   Endpoint: {}", endpoint),
        None => println!("   Mode: direct"),
    }

    // Create swarm
    let libp2p_keypair = agent::device::keypair::to_libp2p_keypair(&config.keypair);
    let mut swarm_builder = MeshSwarmBuilder::new(libp2p_keypair);
    if let Some(endpoint) = runtime_endpoint.clone() {
        swarm_builder = swarm_builder.with_relay_addr(endpoint);
    }
    let mut swarm = swarm_builder.build()?;

    let local_peer_id = swarm.local_peer_id().to_owned();
    println!("   Local PeerID: {}", local_peer_id);
    swarm.listen_on_direct_addrs()?;
    if matches!(
        config.connectivity.preferred_path,
        ConnectivityPath::Relayed
    ) {
        let relay_addr = runtime_endpoint.clone().context("relay endpoint missing")?;

        // Connect to relay
        swarm.connect_to_relay()?;

        // Wait for relay connection event
        println!("   Waiting for relay connection...");

        let local_peer_id_for_events = local_peer_id.clone();
        let relay_peer_id = loop {
            if let Some(event) = swarm.next_event().await {
                match event {
                    agent::MeshEvent::NewListenAddr { address } => {
                        let _ = persist_listen_addr(local_peer_id_for_events.clone(), &address);
                    }
                    agent::MeshEvent::RelayConnected { relay_peer_id, .. } => {
                        println!("   ✓ Connected to relay {}", relay_addr);
                        break relay_peer_id;
                    }
                    agent::MeshEvent::RelayConnectionFailed { error, .. } => {
                        anyhow::bail!("Relay connection failed: {}", error);
                    }
                    _ => {}
                }
            }
        };

        println!("   Creating relay reservation...");
        swarm.listen_on_relay(relay_peer_id)?;

        let local_peer_id_for_events = local_peer_id.clone();
        loop {
            if let Some(event) = swarm.next_event().await {
                match event {
                    agent::MeshEvent::NewListenAddr { address } => {
                        let _ = persist_listen_addr(local_peer_id_for_events.clone(), &address);
                    }
                    agent::MeshEvent::ReservationAccepted { .. } => {
                        println!("   ✓ Relay reservation accepted");
                        break;
                    }
                    agent::MeshEvent::ReservationDenied { .. } => {
                        anyhow::bail!("Relay reservation denied");
                    }
                    _ => {}
                }
            }
        }
    } else {
        let _ = persist_runtime_connectivity_state(&DeviceConnectivityState {
            active_path: ConnectivityPath::Direct,
            active_endpoint: None,
            status: ConnectivityStatus::Connected,
        });
        println!("   ✓ Direct listeners enabled");
    }

    // Start LAN beacon discovery (if pools configured)
    {
        use agent::discovery::{BeaconBroadcaster, BeaconListener};
        use agent::network::RingGossipService;
        use agent::pki::PoolConfig;

        // Check if pools exist before starting any background services
        let has_pools = match PoolConfig::list_pools() {
            Ok(pools) => !pools.is_empty(),
            Err(_) => false,
        };

        match PoolConfig::list_pools() {
            Ok(pools) if !pools.is_empty() => {
                println!("\n🔍 Starting LAN beacon discovery...");
                println!("   Pools: {}", pools.len());

                // Extract configs and certs
                let pool_data: Vec<(PoolConfig, PoolMembershipCert)> = pools
                    .into_iter()
                    .map(|(_, config, cert)| (config, cert))
                    .collect();

                // Create device keypair (should always succeed since we loaded the config)
                let device_keypair = DeviceKeyPair::from_private_bytes(config.keypair.to_bytes())
                    .expect("Failed to create device keypair - this should never happen");

                let node_id = device_keypair.node_id();
                println!("   Node ID: {}", node_id.to_hex());

                // Create beacon listener
                match BeaconListener::new(pool_data.clone(), &device_keypair).await {
                    Ok((listener, mut discovered_rx, ring_gossip_from_lan_rx)) => {
                        // Spawn listener task
                        tokio::spawn(async move {
                            if let Err(e) = listener.run().await {
                                error!("Beacon listener error: {}", e);
                            }
                        });

                        // Spawn discovery handler task (save peers to cache)
                        tokio::spawn(async move {
                            while let Some(peer) = discovered_rx.recv().await {
                                info!(
                                    pool_id = %peer.pool_id,
                                    node_id = %peer.node_id,
                                    lan_addr = %peer.lan_addr,
                                    "LAN peer discovered"
                                );

                                // Save to peer cache
                                if let Ok(mut cache) = PeerCache::load(&peer.pool_id) {
                                    cache.upsert_peer(peer.clone());
                                    if let Err(e) = cache.save(&peer.pool_id) {
                                        error!(error = %e, "Failed to persist peer cache");
                                    }
                                }
                            }
                        });

                        println!("   ✓ Beacon listener started");

                        // Create RingGossipService for each pool
                        // For MVP, we'll create one service for the first pool
                        if let Some((pool_config, _)) = pool_data.first() {
                            let pool_id = pool_config.pool_id;
                            let local_peer_id =
                                agent::device::keypair::to_libp2p_keypair(&config.keypair)
                                    .public()
                                    .to_peer_id();

                            info!(pool_id = %pool_id, "Creating RingGossipService");

                            // Create channels for ring gossip
                            let (ring_gossip_tx, ring_gossip_to_broadcaster_rx) =
                                tokio::sync::mpsc::channel(100);

                            // Create RingGossipService
                            let (ring_service, _ring_state, mut topology_rx) =
                                RingGossipService::new(
                                    pool_id,
                                    node_id,
                                    local_peer_id,
                                    device_keypair.clone(),
                                    ring_gossip_tx.clone(),
                                    ring_gossip_from_lan_rx,
                                );

                            // Spawn topology listener task (saves to file for InferenceCoordinator)
                            let pool_id_for_topology = pool_id;
                            tokio::spawn(async move {
                                while let Ok(topology) = topology_rx.recv().await {
                                    info!(
                                        pool_id = %topology.pool_id,
                                        members = topology.members.len(),
                                        position = topology.my_position,
                                        "Ring topology converged!"
                                    );

                                    // Save topology to file for InferenceCoordinator
                                    let topology_path = dirs::home_dir()
                                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                                        .join(".meshnet")
                                        .join(format!(
                                            "ring_topology_{}.json",
                                            pool_id_for_topology.to_hex()
                                        ));

                                    if let Some(parent) = topology_path.parent() {
                                        let _ = std::fs::create_dir_all(parent);
                                    }

                                    if let Ok(topology_json) =
                                        serde_json::to_string_pretty(&topology)
                                    {
                                        if let Err(e) =
                                            std::fs::write(&topology_path, topology_json)
                                        {
                                            error!(error = %e, "Failed to save ring topology");
                                        } else {
                                            info!(
                                                "Ring topology saved to: {}",
                                                topology_path.display()
                                            );
                                        }
                                    }
                                }
                            });

                            // Create beacon broadcaster with ring gossip channel
                            match BeaconBroadcaster::new(
                                pool_data,
                                device_keypair,
                                None,
                                4001,
                                ring_gossip_to_broadcaster_rx,
                            )
                            .await
                            {
                                Ok(broadcaster) => {
                                    // Spawn broadcaster task
                                    tokio::spawn(async move {
                                        if let Err(e) = broadcaster.run().await {
                                            error!("Beacon broadcaster error: {}", e);
                                        }
                                    });

                                    println!("   ✓ Beacon broadcaster started");
                                }
                                Err(e) => {
                                    error!("Failed to start beacon broadcaster: {}", e);
                                }
                            }

                            // Spawn RingGossipService task
                            tokio::spawn(async move {
                                let _ = ring_service.run().await;
                            });

                            println!("   ✓ Ring gossip service started");
                        }
                    }
                    Err(e) => {
                        // Don't fatal error if beacon listener fails to start
                        tracing::warn!(
                            error = %e,
                            "Failed to start beacon listener (another process may be using port {})",
                            42424
                        );
                        tracing::info!("Continuing in degraded mode (no LAN beacon discovery)");
                        println!("   ⚠️  Beacon listener failed to start (LAN discovery disabled)");
                    }
                }
            }
            Ok(_) => {
                info!("No pools configured - LAN discovery disabled");
                println!("\n💡 No pools configured. Use 'mesh pool create' or 'mesh pool join' to enable LAN discovery.");
            }
            Err(e) => {
                error!("Failed to list pools: {}", e);
            }
        }

        // Always heartbeat when the agent is started from a registered device config.
        // Pools enable LAN discovery, but control-plane deployments still require
        // periodic endpoint and tensor-plane updates for ring coordination.
        let heartbeat_config = config.clone();
        tokio::spawn(async move {
            let client = match RegistrationClient::new(heartbeat_config.control_plane_url.clone()) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to create heartbeat client (non-fatal)");
                    return;
                }
            };

            // Give transport and optional beacon services a moment to publish endpoints
            // before the first heartbeat snapshot is sent.
            let initial_delay_secs = if has_pools { 3 } else { 1 };
            tokio::time::sleep(tokio::time::Duration::from_secs(initial_delay_secs)).await;

            loop {
                if let Err(e) = client.heartbeat(&heartbeat_config).await {
                    tracing::warn!(error = %e, "Heartbeat failed (control plane may be offline)");
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        });
    }

    // Start inference coordinator (in background)
    let inference_config_task = config.clone();
    let runtime_endpoint_clone = runtime_endpoint.clone();
    tokio::spawn(async move {
        use agent::inference::coordinator::{InferenceConfig, InferenceCoordinator};
        use agent::inference::job::{GenerationConfig, InferenceRequest};
        use uuid::Uuid;

        let network_id = inference_config_task.network_id.clone();
        let device_id = inference_config_task.device_id;
        let registration_client =
            match RegistrationClient::new(inference_config_task.control_plane_url.clone()) {
                Ok(client) => client,
                Err(e) => {
                    error!(error = %e, "Failed to create inference registration client");
                    return;
                }
            };

        info!("Initializing inference coordinator");

        let mut inference_runtime_config = InferenceConfig::default();
        inference_runtime_config.recovery_max_attempts_per_job = inference_config_task
            .governance
            .recovery_max_attempts_per_job;
        inference_runtime_config.recovery_cooldown =
            std::time::Duration::from_millis(inference_config_task.governance.recovery_cooldown_ms);
        inference_runtime_config.recovery_max_checkpoint_loads_per_minute = inference_config_task
            .governance
            .recovery_max_checkpoint_loads_per_minute;
        let inference_swarm_keep_alive = inference_runtime_config.job_timeout
            + inference_runtime_config.allreduce_timeout
            + std::time::Duration::from_secs(60);

        // Create separate swarm for inference coordination (P2P communication during all-reduce)
        let libp2p_keypair =
            agent::device::keypair::to_libp2p_keypair(&inference_config_task.keypair);
        let mut swarm_builder =
            MeshSwarmBuilder::new(libp2p_keypair).with_keep_alive(inference_swarm_keep_alive);
        if let Some(endpoint) = runtime_endpoint_clone.clone() {
            swarm_builder = swarm_builder.with_relay_addr(endpoint);
        }
        let mut inference_swarm = match swarm_builder.build() {
            Ok(s) => s,
            Err(e) => {
                error!(error = %e, "Failed to create inference swarm");
                return;
            }
        };
        if let Err(e) = inference_swarm.listen_on_direct_addrs() {
            error!(error = %e, "Failed to enable direct listeners for inference swarm");
            return;
        }

        if matches!(
            inference_config_task.connectivity.preferred_path,
            ConnectivityPath::Relayed
        ) {
            if let Err(e) = inference_swarm.connect_to_relay() {
                error!(error = %e, "Failed to connect inference swarm to relay");
                return;
            }

            let relay_peer_id = loop {
                if let Some(event) = inference_swarm.next_event().await {
                    match event {
                        agent::MeshEvent::RelayConnected { relay_peer_id, .. } => {
                            info!("Inference swarm connected to relay");
                            break relay_peer_id;
                        }
                        agent::MeshEvent::RelayConnectionFailed { error, .. } => {
                            error!(error = %error, "Inference swarm relay connection failed");
                            return;
                        }
                        _ => {}
                    }
                }
            };

            if let Err(e) = inference_swarm.listen_on_relay(relay_peer_id) {
                error!(error = %e, "Failed to create inference swarm relay reservation");
                return;
            }

            loop {
                if let Some(event) = inference_swarm.next_event().await {
                    match event {
                        agent::MeshEvent::ReservationAccepted { .. } => break,
                        agent::MeshEvent::ReservationDenied { .. } => {
                            error!("Inference swarm relay reservation denied");
                            return;
                        }
                        _ => {}
                    }
                }
            }
        }

        let tensor_plane = match TensorPlane::bind(TensorPlaneConfig {
            bind_addr: parse_tensor_plane_bind_addr_env()
                .unwrap_or_else(|| TensorPlaneConfig::default().bind_addr),
            advertised_addr: parse_tensor_plane_advertised_addr_env(),
            max_message_bytes: inference_config_task
                .governance
                .tensor_plane_max_message_bytes,
            max_inbound_messages: inference_config_task
                .governance
                .tensor_plane_max_inbound_messages,
            max_inbound_queued_bytes: inference_config_task
                .governance
                .tensor_plane_max_inbound_queued_bytes,
            max_outbound_inflight_bytes: inference_config_task
                .governance
                .tensor_plane_max_outbound_inflight_bytes,
            max_send_bandwidth_bytes_per_sec: inference_config_task
                .governance
                .tensor_plane_max_send_bandwidth_bytes_per_sec,
            ..TensorPlaneConfig::default()
        })
        .await
        {
            Ok(plane) => plane,
            Err(e) => {
                error!(error = %e, "Failed to start dedicated tensor data plane");
                return;
            }
        };
        if let Err(e) = persist_advertised_endpoint(&tensor_plane.advertised_endpoint()) {
            warn!(error = %e, "Failed to persist tensor data-plane endpoint");
        } else {
            info!(
                endpoint = %tensor_plane.advertised_endpoint(),
                local_addr = %tensor_plane.local_addr(),
                "Dedicated tensor data plane ready"
            );
        }

        let mut coordinator =
            InferenceCoordinator::new(inference_swarm, tensor_plane, inference_runtime_config);

        info!("Inference coordinator initialized - starting assignment loop");

        let topology_url = format!(
            "{}/api/ring/topology?network_id={}",
            inference_config_task
                .control_plane_url
                .trim_end_matches('/'),
            inference_config_task.network_id
        );

        let mut ring_position = None;
        match reqwest::Client::new()
            .get(&topology_url)
            .send()
            .await
            .and_then(|response| response.error_for_status())
        {
            Ok(response) => match response.json::<RingTopologyResponse>().await {
                Ok(topology) => match build_worker_position_from_topology(&topology, &device_id) {
                    Ok(position) => {
                        if let Err(e) = coordinator.join_ring(position.clone()) {
                            error!(error = %e, "Failed to join ring in coordinator");
                        } else {
                            info!(
                                position = position.position,
                                total_workers = position.total_workers,
                                shard_range = ?position.shard_column_range,
                                "Loaded ring position for inference"
                            );
                            ring_position = Some(position);
                        }
                    }
                    Err(e) => warn!(error = %e, "Failed to build worker position from topology"),
                },
                Err(e) => warn!(error = %e, "Failed to parse ring topology"),
            },
            Err(e) => {
                warn!(error = %e, "Failed to refresh ring topology before inference startup")
            }
        }

        if ring_position.is_none() {
            warn!("No ring position found - worker must join ring first via 'mesh ring join'");
            warn!("Inference jobs will be logged but not executed");
        }

        // Claim and execute inference assignments
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            let assignment = match registration_client
                .claim_inference_assignment(device_id, &network_id)
                .await
            {
                Ok(Some(assignment)) => assignment,
                Ok(None) => continue,
                Err(e) => {
                    debug!(error = %e, "Inference assignment claim failed");
                    continue;
                }
            };

            let job_id = match Uuid::parse_str(&assignment.job_id) {
                Ok(job_id) => job_id,
                Err(e) => {
                    error!(job_id = %assignment.job_id, error = %e, "Invalid assignment job id");
                    continue;
                }
            };

            info!(
                job_id = %job_id,
                assignment_id = %assignment.assignment_id,
                model = %assignment.model_id,
                prompt_len = assignment.prompt_tokens.len(),
                max_tokens = assignment.max_tokens,
                "Claimed distributed inference assignment"
            );

            match reqwest::Client::new()
                .get(&topology_url)
                .send()
                .await
                .and_then(|response| response.error_for_status())
            {
                Ok(response) => match response.json::<RingTopologyResponse>().await {
                    Ok(topology) => {
                        match build_worker_position_from_topology(&topology, &device_id) {
                            Ok(position) => {
                                if let Err(e) = coordinator.join_ring(position.clone()) {
                                    error!(error = %e, "Failed to refresh ring position from topology");
                                } else {
                                    ring_position = Some(position);
                                }
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to build worker position for assignment")
                            }
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to parse topology during assignment refresh")
                    }
                },
                Err(e) => warn!(error = %e, "Failed to refresh topology during assignment"),
            }

            if ring_position.is_none() {
                warn!(job_id = %job_id, "Cannot process assignment while not in ring");
                let _ = registration_client
                    .report_inference_result(
                        job_id,
                        agent::api::types::ReportInferenceAssignmentRequest {
                            device_id: device_id.to_string(),
                            success: false,
                            completion: None,
                            completion_tokens: None,
                            execution_time_ms: 0,
                            error: Some("worker not in ring topology".to_string()),
                        },
                    )
                    .await;
                continue;
            }

            if let Err(e) = registration_client
                .acknowledge_inference_assignment(job_id, device_id)
                .await
            {
                error!(job_id = %job_id, error = %e, "Failed to acknowledge assignment");
                continue;
            }

            let request = InferenceRequest {
                job_id,
                network_id: assignment.network_id.clone(),
                model_id: assignment.model_id.clone(),
                prompt_tokens: assignment.prompt_tokens.clone(),
                config: GenerationConfig {
                    max_tokens: assignment.max_tokens,
                    temperature: assignment.temperature,
                    top_p: assignment.top_p,
                    ..Default::default()
                },
                executor_id: device_id.to_string(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };

            match coordinator.process_inference(request).await {
                Ok(result) => {
                    let completion = result.generated_tokens.as_ref().and_then(|tokens| {
                        match agent::model_assets::decode_tokens(&assignment.model_id, tokens) {
                            Ok(text) => Some(text),
                            Err(e) => {
                                warn!(
                                    job_id = %job_id,
                                    model_id = %assignment.model_id,
                                    error = %e,
                                    "Failed to decode generated tokens with model tokenizer"
                                );
                                None
                            }
                        }
                    });

                    if let Err(e) = registration_client
                        .report_inference_result(
                            job_id,
                            agent::api::types::ReportInferenceAssignmentRequest {
                                device_id: device_id.to_string(),
                                success: result.success,
                                completion,
                                completion_tokens: Some(result.completion_tokens),
                                execution_time_ms: result.execution_time_ms,
                                error: result.error.clone(),
                            },
                        )
                        .await
                    {
                        error!(job_id = %job_id, error = %e, "Failed to report inference result");
                    }
                }
                Err(e) => {
                    error!(job_id = %job_id, error = %e, "Inference job failed");
                    let _ = registration_client
                        .report_inference_result(
                            job_id,
                            agent::api::types::ReportInferenceAssignmentRequest {
                                device_id: device_id.to_string(),
                                success: false,
                                completion: None,
                                completion_tokens: None,
                                execution_time_ms: 0,
                                error: Some(e.to_string()),
                            },
                        )
                        .await;
                }
            }
        }
    });

    println!("\n✅ Agent ready - participating in mesh inference...");
    println!("   Press Ctrl+C to stop\n");

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Shutdown signal received");
                break;
            }
            event = swarm.next_event() => {
                let Some(event) = event else {
                    break;
                };

                match event {
                    agent::MeshEvent::NewListenAddr { address } => {
                        let _ = persist_listen_addr(local_peer_id, &address);
                    }
                    agent::MeshEvent::PeerConnected { peer_id, connection_info } => {
                        info!(
                            peer_id = %peer_id,
                            connection_type = ?connection_info.connection_type,
                            remote_addr = %connection_info.remote_addr,
                            "Primary swarm peer connected"
                        );
                    }
                    agent::MeshEvent::PeerDisconnected { peer_id } => {
                        info!(peer_id = %peer_id, "Primary swarm peer disconnected");
                    }
                    agent::MeshEvent::DirectConnectionUpgraded { peer_id } => {
                        info!(peer_id = %peer_id, "Primary swarm upgraded to direct connectivity");
                    }
                    agent::MeshEvent::DirectConnectionUpgradeFailed { peer_id, error } => {
                        warn!(peer_id = %peer_id, error = %error, "Primary swarm direct upgrade failed");
                    }
                    agent::MeshEvent::RelayDisconnected { relay_peer_id } => {
                        warn!(relay_peer_id = %relay_peer_id, "Relay disconnected");
                    }
                    agent::MeshEvent::RelayConnectionFailed { relay_addr, error } => {
                        warn!(relay_addr = %relay_addr, error = %error, "Relay connection failed");
                    }
                    agent::MeshEvent::ReservationDenied { relay_peer_id } => {
                        warn!(relay_peer_id = %relay_peer_id, "Relay reservation denied");
                    }
                    agent::MeshEvent::ExternalAddrCandidateDiscovered { address } => {
                        debug!(address = %address, "Observed external address candidate");
                    }
                    agent::MeshEvent::ExternalAddrConfirmed { address } => {
                        info!(address = %address, "Confirmed external address");
                    }
                    agent::MeshEvent::PunchPathAttemptInitiated {
                        peer_id,
                        reason,
                        candidate_count,
                        relay_rendezvous_required,
                    } => {
                        info!(
                            peer_id = %peer_id,
                            reason = %reason,
                            candidate_count,
                            relay_rendezvous_required,
                            "Punch-path attempt initiated"
                        );
                    }
                    agent::MeshEvent::PeerIdentified { .. }
                    | agent::MeshEvent::RelayConnected { .. }
                    | agent::MeshEvent::ReservationAccepted { .. } => {}
                }
            }
        }
    }

    Ok(())
}

/// Show device and network status
async fn cmd_status() -> Result<()> {
    println!("📊 Mesh AI Agent Status\n");

    // Try to load configuration
    match DeviceConfig::load(&DeviceConfig::default_path()?) {
        Ok(config) => {
            println!("✅ Device Configured");
            println!("   Device ID: {}", config.device_id);
            println!("   Name: {}", config.name);
            println!("   Network ID: {}", config.network_id);
            println!("   Tier: {:?}", config.capabilities.tier);
            println!("\n   Capabilities:");
            println!("     CPU Cores: {}", config.capabilities.cpu_cores);
            println!("     RAM: {} MB", config.capabilities.ram_mb);
            println!("     OS: {}", config.capabilities.os);
            println!("     Architecture: {}", config.capabilities.arch);

            // Check if certificate exists
            if config.has_certificate() {
                println!("\n✅ Certificate: Present");
            } else {
                println!("\n⚠️  Certificate: Missing (run 'mesh device init' again)");
            }

            println!("\n📡 Control Plane: {}", config.control_plane_url);
            println!(
                "📡 Preferred Path: {:?}",
                config.connectivity.preferred_path
            );
            let connectivity_state = config.connectivity.current_state();
            println!("📡 Active Connectivity: {:?}", connectivity_state.status);
            println!("   Active Path: {:?}", connectivity_state.active_path);
            if let Some(endpoint) = connectivity_state.active_endpoint {
                println!("   Active Endpoint: {}", endpoint);
            }
            if config.connectivity.attachments.is_empty() {
                println!("   Connectivity Attachments: none");
            } else {
                for attachment in &config.connectivity.attachments {
                    let label = match attachment.kind {
                        ConnectivityAttachmentKind::Libp2pRelay => "libp2p_relay",
                    };
                    println!(
                        "   Connectivity Attachment: {} {} (priority {})",
                        label, attachment.endpoint, attachment.priority
                    );
                }
            }

            let local_listen_addrs = load_local_listen_addrs();
            let observed_addrs = load_observed_reachability_addrs().unwrap_or_default();
            let candidate_seed_records = load_direct_candidate_seed_records().unwrap_or_default();
            let local_peer_id = agent::device::keypair::to_libp2p_keypair(&config.keypair)
                .public()
                .to_peer_id();
            let direct_candidates =
                build_direct_peer_candidates_from_records(local_peer_id, &candidate_seed_records);

            println!("\n📡 Local Reachability:");
            println!("   Listen Addresses: {}", local_listen_addrs.len());
            for addr in local_listen_addrs.iter().take(3) {
                println!("     - {}", addr);
            }
            if local_listen_addrs.len() > 3 {
                println!("     - ... {} more", local_listen_addrs.len() - 3);
            }

            println!("   Observed External Addresses: {}", observed_addrs.len());
            for addr in observed_addrs.iter().take(3) {
                println!("     - {}", addr);
            }
            if observed_addrs.len() > 3 {
                println!("     - ... {} more", observed_addrs.len() - 3);
            }

            print_direct_candidates_block("Ranked Direct Candidates", &direct_candidates);
        }
        Err(_) => {
            println!("⚠️  Device not initialized");
            println!("\nRun 'mesh device init' to set up this device.");
        }
    }

    Ok(())
}

/// Lock resources for pool contribution
async fn cmd_lock_resources(memory: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Locking Resources".bold().cyan());
    println!("{}", "=================".cyan());

    // Parse memory string
    let bytes = parse_memory_string(&memory).context("Invalid memory format")?;

    // Create and configure resource manager
    let mut manager = ResourceManager::new().context("Failed to initialize resource manager")?;

    manager
        .load_config()
        .context("Failed to load resource config")?;

    // Check if already locked
    if manager.is_locked() {
        println!("\n{}", "Memory is already locked!".yellow());
        if let Some(remaining) = manager.time_until_unlock() {
            println!("  Time until unlock: {} hours", remaining.as_secs() / 3600);
        }
        return Ok(());
    }

    // Set allocation
    manager
        .set_allocation(bytes)
        .context("Failed to set allocation")?;

    println!("\n{}", "Allocation:".bold());
    println!(
        "  Requested:     {}",
        format_bytes(manager.user_allocated())
    );
    println!(
        "  With buffer:   {} (7% safety margin)",
        format_bytes(manager.locked_memory())
    );
    println!("  Total system:  {}", format_bytes(manager.total_memory()));

    // Lock memory
    println!("\n{}", "Locking memory...".bold());

    match manager.lock_memory() {
        Ok(()) => {
            println!("\n{}", "Memory locked successfully!".green().bold());
            println!("  Lock timestamp: {:?}", manager.lock_timestamp());
            println!("  Cooldown period: 24 hours");
            println!(
                "\n{}",
                "Note: Memory will remain locked for 24 hours.".yellow()
            );
            println!(
                "{}",
                "Use 'mesh resource status' to check lock status.".yellow()
            );
        }
        Err(e) => {
            println!("\n{}", format!("Failed to lock memory: {}", e).red());
            println!("\n{}", "Possible causes:".yellow());
            println!("  - Insufficient privileges (try running with sudo)");
            println!("  - System memory limits (check ulimit -l)");
            println!("  - Requested memory exceeds available resources");
            return Err(e.into());
        }
    }

    Ok(())
}

/// Request resource unlock (requires 24h cooldown)
async fn cmd_unlock_resources() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Unlocking Resources".bold().cyan());
    println!("{}", "===================".cyan());

    // Create resource manager and load config
    let mut manager = ResourceManager::new().context("Failed to initialize resource manager")?;

    manager
        .load_config()
        .context("Failed to load resource config")?;

    // Check if locked
    if !manager.is_locked() {
        println!("\n{}", "Memory is not locked.".yellow());
        return Ok(());
    }

    // Try to unlock
    match manager.unlock_memory() {
        Ok(()) => {
            println!("\n{}", "Memory unlocked successfully!".green().bold());
            println!("{}", "Resources are now available for other uses.".green());
        }
        Err(agent::AgentError::CooldownActive { remaining_hours }) => {
            println!("\n{}", "Cannot unlock yet - cooldown active!".red().bold());
            println!("  Remaining time: {} hours", remaining_hours);
            println!(
                "\n{}",
                "The 24-hour cooldown period has not elapsed.".yellow()
            );
            println!(
                "{}",
                "This prevents frequent resource changes in the pool.".yellow()
            );
        }
        Err(e) => {
            println!("\n{}", format!("Failed to unlock memory: {}", e).red());
            return Err(e.into());
        }
    }

    Ok(())
}

/// Show resource lock status
async fn cmd_resource_status() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Resource Lock Status".bold().cyan());
    println!("{}", "====================".cyan());

    // Create resource manager and load config
    let mut manager = ResourceManager::new().context("Failed to initialize resource manager")?;

    manager
        .load_config()
        .context("Failed to load resource config")?;

    println!("\n{}", "System Memory:".bold());
    println!("  Total:         {}", format_bytes(manager.total_memory()));

    println!("\n{}", "Allocation:".bold());
    println!(
        "  User request:  {}",
        format_bytes(manager.user_allocated())
    );
    println!("  Locked (buf):  {}", format_bytes(manager.locked_memory()));

    println!("\n{}", "Lock Status:".bold());
    if manager.is_locked() {
        println!("  Status:        {}", "LOCKED".green().bold());

        if let Some(ts) = manager.lock_timestamp() {
            match ts.duration_since(SystemTime::UNIX_EPOCH) {
                Ok(duration) => {
                    let datetime = chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0)
                        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "Invalid timestamp".to_string());
                    println!("  Locked at:     {}", datetime);
                }
                Err(_) => {
                    warn!("Lock timestamp is before UNIX epoch - possible system clock issue");
                    println!("  Locked at:     {}", "Invalid (clock error)".red());
                }
            }
        }

        if let Some(remaining) = manager.time_until_unlock() {
            let hours = remaining.as_secs() / 3600;
            let minutes = (remaining.as_secs() % 3600) / 60;
            println!("  Unlock in:     {}h {}m", hours, minutes);
        } else {
            println!("  Unlock in:     {}", "Ready to unlock".green());
        }
    } else {
        println!("  Status:        {}", "UNLOCKED".yellow());
        println!("  Unlock in:     N/A");
    }

    println!();

    Ok(())
}

/// Show ring topology and worker position
async fn cmd_ring_status() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Ring Topology Status".bold().cyan());
    println!("{}", "====================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = match DeviceConfig::load(&config_path) {
        Ok(c) => c,
        Err(_) => {
            println!("\n{}", "Device not initialized".yellow());
            println!("Run 'mesh device init' to set up this device.\n");
            return Ok(());
        }
    };

    println!("\n{}", "Worker Identity:".bold());
    println!("  Device ID:     {}", config.device_id);
    println!("  Network ID:    {}", config.network_id);

    match fetch_ring_topology(&config).await {
        Ok(topology) => {
            if let Some(worker) = topology
                .workers
                .iter()
                .find(|worker| worker.device_id == config.device_id.to_string())
            {
                println!("\n{}", "Ring Position:".bold());
                println!("  Position:      {}", worker.position.to_string().green());
                println!("  Total Workers: {}", topology.workers.len());
                println!("  Left Neighbor: {}", worker.left_neighbor);
                println!("  Right Neighbor: {}", worker.right_neighbor);
            } else {
                println!("\n{}", "Ring Status:".bold());
                println!("  Status:        {}", "NOT JOINED".yellow());
                println!(
                    "\n{}",
                    "Use 'mesh ring join --model-id <model>' to join the production ring.".dimmed()
                );
            }
        }
        Err(e) => {
            println!("\n{}", format!("Failed to load topology: {}", e).yellow());
        }
    }

    println!();
    Ok(())
}

async fn cmd_topology() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Ring Topology".bold().cyan());
    println!("{}", "=============".cyan());

    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;
    let topology = fetch_ring_topology(&config).await?;

    println!("\n{}", "Network:".bold());
    println!("  Network ID:    {}", config.network_id);
    println!(
        "  Ring Stable:   {}",
        if topology.ring_stable {
            "yes".green()
        } else {
            "no".yellow()
        }
    );
    println!("  Workers:       {}", topology.workers.len());
    println!("  Punch Plans:   {}", topology.peer_punch_plans.len());

    if topology.workers.is_empty() {
        println!(
            "\n{}",
            "No workers are currently present in the ring.".yellow()
        );
        return Ok(());
    }

    println!("\n{}", "Workers:".bold());
    for worker in &topology.workers {
        let marker = if worker.device_id == config.device_id.to_string() {
            "→"
        } else {
            " "
        };
        let connectivity = worker
            .connectivity_state
            .as_ref()
            .map(|state| format!("{:?}/{:?}", state.active_path, state.status))
            .unwrap_or_else(|| "unknown".to_string());
        println!(
            "{} pos={} device={} status={} provider_candidates={} shard={}..{} connectivity={}",
            marker,
            worker.position,
            worker.device_id,
            worker.status,
            worker.direct_candidates.len(),
            worker.shard.column_start,
            worker.shard.column_end,
            connectivity
        );
        if let Some(best) = worker.direct_candidates.first() {
            println!(
                "    best-direct {:?}/{:?}/{:?} priority={} age={}s {}",
                best.scope,
                best.transport,
                best.source,
                best.priority,
                direct_candidate_age_seconds(best.last_updated_ms),
                best.endpoint
            );
        }
    }

    if !topology.peer_punch_plans.is_empty() {
        println!("\n{}", "Punch Plans:".bold());
        for plan in topology.peer_punch_plans.iter().take(10) {
            println!(
                "  {} -> {} reason={:?} candidates={} relay_rendezvous={}",
                plan.source_device_id,
                plan.target_device_id,
                plan.reason,
                plan.target_candidates.len(),
                plan.relay_rendezvous_required
            );
        }
        if topology.peer_punch_plans.len() > 10 {
            println!(
                "  ... {} more punch plans",
                topology.peer_punch_plans.len() - 10
            );
        }
    }

    println!();
    Ok(())
}

/// Show shard assignment status
async fn cmd_shard_status() -> Result<()> {
    use agent::{ShardRegistry, ShardStatus};
    use colored::Colorize;

    println!("\n{}", "Shard Status".bold().cyan());
    println!("{}", "============".cyan());

    // Load shard registry
    let registry = match ShardRegistry::with_defaults() {
        Ok(r) => r,
        Err(e) => {
            println!(
                "\n{}",
                format!("Failed to load shard registry: {}", e).red()
            );
            return Ok(());
        }
    };

    // Load from disk
    if let Err(e) = registry.load().await {
        println!("\n{}", format!("Failed to load shard data: {}", e).yellow());
    }

    let shards = registry.list_shards().await;

    if shards.is_empty() {
        println!("\n{}", "No shards assigned".yellow());
        println!(
            "{}",
            "Shards are assigned when you join a ring topology.".dimmed()
        );
    } else {
        println!("\n{}", "Assigned Shards:".bold());
        for (model_id, info, status) in &shards {
            let status_str = match status {
                ShardStatus::Pending => "PENDING".yellow().to_string(),
                ShardStatus::Downloading => "DOWNLOADING".blue().to_string(),
                ShardStatus::Downloaded => "DOWNLOADED".cyan().to_string(),
                ShardStatus::Ready => "READY".green().bold().to_string(),
                ShardStatus::Error => "ERROR".red().to_string(),
            };

            println!("\n  Model: {}", model_id.bold());
            println!("    Status:       {}", status_str);
            println!(
                "    Columns:      {} - {}",
                info.assignment.column_start, info.assignment.column_end
            );
            println!(
                "    Worker Pos:   {}/{}",
                info.assignment.worker_position, info.assignment.total_workers
            );
            if info.memory_bytes > 0 {
                println!("    Memory:       {}", format_bytes(info.memory_bytes));
            }
            if info.download_progress > 0.0 && info.download_progress < 1.0 {
                println!("    Download:     {:.1}%", info.download_progress * 100.0);
            }
        }
    }

    // Show total memory usage
    let total_mem = registry.total_memory_usage().await;
    if total_mem > 0 {
        println!("\n{}", "Total Memory:".bold());
        println!("  Shard Memory:  {}", format_bytes(total_mem));
    }

    println!();
    Ok(())
}

/// Show inference statistics
async fn cmd_inference_stats() -> Result<()> {
    use colored::Colorize;
    use std::path::PathBuf;

    println!("\n{}", "Inference Statistics".bold().cyan());
    println!("{}", "====================".cyan());

    // Load inference stats from file
    let stats_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".meshnet")
        .join("inference_stats.json");

    if !stats_path.exists() {
        println!("\n{}", "No inference statistics available".yellow());
        println!(
            "{}",
            "Statistics are recorded during inference jobs.".dimmed()
        );
        println!();
        return Ok(());
    }

    let stats_json =
        std::fs::read_to_string(&stats_path).context("Failed to read inference stats file")?;

    let stats: serde_json::Value =
        serde_json::from_str(&stats_json).context("Failed to parse inference stats")?;

    println!("\n{}", "Job Statistics:".bold());
    if let Some(completed) = stats.get("jobs_completed") {
        println!("  Completed:       {}", completed.to_string().green());
    }
    if let Some(failed) = stats.get("jobs_failed") {
        println!("  Failed:          {}", failed.to_string().red());
    }
    if let Some(rate) = stats.get("success_rate") {
        println!(
            "  Success Rate:    {:.1}%",
            rate.as_f64().unwrap_or(0.0) * 100.0
        );
    }

    println!("\n{}", "Token Statistics:".bold());
    if let Some(tokens) = stats.get("total_tokens_generated") {
        println!("  Tokens Generated: {}", tokens);
    }
    if let Some(prompt) = stats.get("total_prompt_tokens") {
        println!("  Prompt Tokens:    {}", prompt);
    }
    if let Some(tps) = stats.get("avg_tokens_per_second") {
        println!("  Avg Tokens/sec:   {:.2}", tps.as_f64().unwrap_or(0.0));
    }

    println!("\n{}", "Ring Performance:".bold());
    if let Some(ops) = stats.get("allreduce_operations") {
        println!("  All-Reduce Ops:   {}", ops);
    }
    if let Some(latency) = stats.get("avg_allreduce_latency_ms") {
        println!(
            "  Avg Latency:      {:.2}ms",
            latency.as_f64().unwrap_or(0.0)
        );
    }
    if let Some(layers) = stats.get("total_layers_processed") {
        println!("  Layers Processed: {}", layers);
    }
    if let Some(bytes_sent) = stats.get("tensor_bytes_sent") {
        println!("  Tensor Bytes Sent: {}", bytes_sent);
    }
    if let Some(bytes_received) = stats.get("tensor_bytes_received") {
        println!("  Tensor Bytes Recv: {}", bytes_received);
    }
    if let Some(wait_count) = stats.get("tensor_outbound_backpressure_wait_count") {
        println!("  Send Waits:        {}", wait_count);
    }
    if let Some(wait_ms) = stats.get("tensor_outbound_backpressure_wait_ms") {
        println!("  Send Wait Time:    {}ms", wait_ms);
    }
    if let Some(wait_count) = stats.get("tensor_outbound_bandwidth_wait_count") {
        println!("  Bandwidth Waits:   {}", wait_count);
    }
    if let Some(wait_ms) = stats.get("tensor_outbound_bandwidth_wait_ms") {
        println!("  Bandwidth Wait:    {}ms", wait_ms);
    }
    if let Some(queue_drops) = stats.get("tensor_inbound_queue_full_rejections") {
        println!("  Queue Drops:       {}", queue_drops);
    }
    if let Some(byte_budget_drops) = stats.get("tensor_inbound_byte_budget_rejections") {
        println!("  Byte Budget Drops: {}", byte_budget_drops);
    }
    if let Some(oversize_drops) = stats.get("tensor_oversized_message_rejections") {
        println!("  Oversize Drops:    {}", oversize_drops);
    }

    println!("\n{}", "Fault Tolerance:".bold());
    if let Some(ckpts) = stats.get("checkpoints_created") {
        println!("  Checkpoints:      {}", ckpts);
    }
    if let Some(recoveries) = stats.get("checkpoint_recoveries") {
        println!("  Recoveries:       {}", recoveries);
    }
    if let Some(attempts) = stats.get("recovery_attempts") {
        println!("  Recovery Attempts: {}", attempts);
    }
    if let Some(cooldown_hits) = stats.get("recovery_cooldown_rejections") {
        println!("  Recovery Cooldowns: {}", cooldown_hits);
    }
    if let Some(budget_hits) = stats.get("recovery_budget_rejections") {
        println!("  Recovery Budget Hit: {}", budget_hits);
    }
    if let Some(misses) = stats.get("recovery_checkpoint_misses") {
        println!("  Recovery Misses:   {}", misses);
    }

    println!("\n{}", "System:".bold());
    if let Some(uptime) = stats.get("uptime") {
        println!(
            "  Uptime:           {}",
            uptime.as_str().unwrap_or("unknown")
        );
    }
    if let Some(updated) = stats.get("last_updated") {
        println!(
            "  Last Updated:     {}",
            updated.as_str().unwrap_or("unknown")
        );
    }

    println!();
    Ok(())
}

async fn fetch_job_status(
    config: &DeviceConfig,
    job_id: &str,
) -> Result<InferenceJobStatusResponse> {
    let url = control_plane_url(config, &format!("/api/inference/jobs/{}", job_id));
    control_plane_client()
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json::<InferenceJobStatusResponse>()
        .await
        .map_err(Into::into)
}

fn print_job_status_snapshot(status: &InferenceJobStatusResponse) {
    use colored::Colorize;

    println!("\n{}", "Job Status".bold().cyan());
    println!("{}", "==========".cyan());
    println!("  Job ID:          {}", status.job_id);
    println!("  Network ID:      {}", status.network_id);
    println!("  Model ID:        {}", status.model_id);
    println!("  Success Flag:    {}", status.success);
    println!("  Status:          {}", status.status);
    println!("  Tokens:          {}", status.completion_tokens);
    println!("  Execution Time:  {}ms", status.execution_time_ms);
    if let Some(error) = &status.error {
        println!("  Error:           {}", error.red());
    }

    println!("\n{}", "Assignments:".bold());
    for assignment in &status.assignments {
        println!(
            "  pos={} device={} status={} provider={} shard={}..{} capacity={} time={}ms",
            assignment.ring_position,
            assignment.device_id,
            assignment.status,
            assignment.execution_provider,
            assignment.shard_column_start,
            assignment.shard_column_end,
            assignment.assigned_capacity_units,
            assignment.execution_time_ms
        );
        if let Some(reason) = &assignment.failure_reason {
            println!("    failure: {}", reason);
        }
    }

    if let Some(completion) = &status.completion {
        println!("\n{}", "Completion:".bold());
        println!("{}", completion);
    }
}

async fn cmd_job_status(job_id: &str, watch: bool, poll_interval_ms: u64) -> Result<()> {
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    if !watch {
        let status = fetch_job_status(&config, job_id).await?;
        print_job_status_snapshot(&status);
        println!();
        return Ok(());
    }

    let poll_interval = std::time::Duration::from_millis(poll_interval_ms.max(250));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
    let mut last_status = None::<String>;

    loop {
        if std::time::Instant::now() > deadline {
            anyhow::bail!("Timed out waiting for job {}", job_id);
        }

        let status = fetch_job_status(&config, job_id).await?;
        if last_status.as_deref() != Some(status.status.as_str()) {
            println!("  Status: {}", status.status);
            last_status = Some(status.status.clone());
        }

        match status.status.as_str() {
            "completed" => {
                print_job_status_snapshot(&status);
                println!();
                return Ok(());
            }
            "failed" => {
                print_job_status_snapshot(&status);
                anyhow::bail!(
                    "Job {} failed: {}",
                    job_id,
                    status
                        .error
                        .unwrap_or_else(|| "unknown job failure".to_string())
                );
            }
            _ => tokio::time::sleep(poll_interval).await,
        }
    }
}

async fn cmd_ledger_summary(job_id: Option<&str>) -> Result<()> {
    use colored::Colorize;

    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;
    let mut url = control_plane_url(
        &config,
        &format!("/api/ledger/summary?network_id={}", config.network_id),
    );
    if let Some(job_id) = job_id {
        url.push_str("&job_id=");
        url.push_str(job_id);
    }

    let summary = control_plane_client()
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json::<LedgerSummaryResponse>()
        .await?;

    println!("\n{}", "Ledger Summary".bold().cyan());
    println!("{}", "==============".cyan());
    println!("  Network ID:      {}", summary.network_id);
    println!(
        "  Job ID:          {}",
        summary.job_id.unwrap_or_else(|| "all jobs".to_string())
    );
    println!("  Total Events:    {}", summary.total_events);
    println!("  Jobs Started:    {}", summary.total_jobs_started);
    println!("  Jobs Completed:  {}", summary.total_jobs_completed);
    println!("  Credits Earned:  {:.3}", summary.total_credits_earned);
    println!("  Credits Burned:  {:.3}", summary.total_credits_burned);
    println!();
    Ok(())
}

async fn cmd_ledger_events(job_id: Option<&str>, limit: usize) -> Result<()> {
    use colored::Colorize;

    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;
    let mut url = control_plane_url(
        &config,
        &format!("/api/ledger/events?network_id={}", config.network_id),
    );
    if let Some(job_id) = job_id {
        url.push_str("&job_id=");
        url.push_str(job_id);
    }

    let events = control_plane_client()
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json::<LedgerEventsResponse>()
        .await?;

    println!("\n{}", "Ledger Events".bold().cyan());
    println!("{}", "=============".cyan());
    for event in events.events.iter().take(limit) {
        println!(
            "  {} {} network={} job={} device={} credits={} event_id={}",
            event.created_at,
            event.event_type,
            event.network_id,
            event.job_id.as_deref().unwrap_or("-"),
            event.device_id.as_deref().unwrap_or("-"),
            event
                .credits_amount
                .map(|value| format!("{:.3}", value))
                .unwrap_or_else(|| "-".to_string()),
            event.event_id
        );
        if event.metadata != serde_json::Value::Null && event.metadata != serde_json::json!({}) {
            println!("    metadata: {}", event.metadata);
        }
    }
    if events.events.len() > limit {
        println!("  ... {} more events", events.events.len() - limit);
    }
    println!();
    Ok(())
}

async fn cmd_doctor() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Mesh Doctor".bold().cyan());
    println!("{}", "===========".cyan());

    let config_path = DeviceConfig::default_path()?;
    let config = match DeviceConfig::load(&config_path) {
        Ok(config) => config,
        Err(_) => {
            println!("  {} no device config found", "FAIL".red().bold());
            println!("  Run `mesh device init` first.\n");
            return Ok(());
        }
    };

    println!("  {} device config loaded", "OK".green().bold());
    println!("    path: {}", config_path.display());
    println!(
        "  {} certificate {}",
        if config.has_certificate() {
            "OK".green().bold().to_string()
        } else {
            "FAIL".red().bold().to_string()
        },
        if config.has_certificate() {
            "present"
        } else {
            "missing"
        }
    );

    let provider = config.resolve_execution_provider()?;
    println!(
        "  {} execution provider {}",
        "OK".green().bold(),
        provider.as_str()
    );

    match fetch_ring_topology(&config).await {
        Ok(topology) => {
            println!(
                "  {} control plane reachable, workers={}, ring_stable={}",
                "OK".green().bold(),
                topology.workers.len(),
                topology.ring_stable
            );
        }
        Err(error) => {
            println!(
                "  {} control plane unreachable: {}",
                "FAIL".red().bold(),
                error
            );
        }
    }

    let candidate_seed_records = load_direct_candidate_seed_records().unwrap_or_default();
    let local_peer_id = agent::device::keypair::to_libp2p_keypair(&config.keypair)
        .public()
        .to_peer_id();
    let direct_candidates =
        build_direct_peer_candidates_from_records(local_peer_id, &candidate_seed_records);
    println!(
        "  {} direct candidates={} observed_addrs={} listen_addrs={}",
        "OK".green().bold(),
        direct_candidates.len(),
        load_observed_reachability_addrs().unwrap_or_default().len(),
        load_local_listen_addrs().len()
    );
    println!();
    Ok(())
}

/// Show pool status (all workers from control plane)
async fn cmd_pool_status() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Pool Status".bold().cyan());
    println!("{}", "===========".cyan());

    // Load device configuration to get network ID
    let config_path = DeviceConfig::default_path()?;
    let config = match DeviceConfig::load(&config_path) {
        Ok(c) => c,
        Err(_) => {
            println!("\n{}", "Device not initialized".yellow());
            println!("Run 'mesh device init' to set up this device.\n");
            return Ok(());
        }
    };

    println!("\n{}", "Fetching pool status...".dimmed());

    // Fetch ring topology from control plane
    let client = control_plane_client();
    let url = control_plane_url(
        &config,
        &format!("/api/ring/topology?network_id={}", config.network_id),
    );

    match client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(data) = response.json::<RingTopologyResponse>().await {
                    println!("\n{}", "Ring Topology:".bold());
                    println!(
                        "  Total Workers: {}",
                        data.workers.len().to_string().green()
                    );
                    println!(
                        "  Ring Stable:   {}",
                        if data.ring_stable {
                            "yes".green()
                        } else {
                            "no".yellow()
                        }
                    );
                    println!();

                    for (i, worker) in data.workers.iter().enumerate() {
                        let is_self = worker.device_id == config.device_id.to_string();
                        let prefix = if is_self { "→ " } else { "  " };

                        let status_colored = match worker.status.as_str() {
                            "online" => "ONLINE".green().to_string(),
                            "offline" => "OFFLINE".red().to_string(),
                            _ => worker.status.yellow().to_string(),
                        };

                        let connectivity = worker
                            .connectivity_state
                            .as_ref()
                            .map(|state| format!("{:?}/{:?}", state.status, state.active_path))
                            .unwrap_or_else(|| "unknown".to_string());

                        println!(
                            "{}Worker {} (pos {}): {} - {} RAM - {}",
                            prefix,
                            i,
                            worker.position,
                            status_colored,
                            format_bytes(worker.contributed_memory),
                            connectivity
                        );
                        if let Some(best) = worker.direct_candidates.first() {
                            println!(
                                "    best direct: {:?}/{:?} priority={} {}",
                                best.scope, best.transport, best.priority, best.endpoint
                            );
                            println!("    direct candidates: {}", worker.direct_candidates.len());
                        } else {
                            println!("    direct candidates: 0");
                        }
                    }

                    if data.workers.is_empty() {
                        println!("  {}", "No workers in ring".yellow());
                    }
                } else {
                    println!("  {}", "Failed to parse response".red());
                }
            } else {
                println!(
                    "  {} (HTTP {})",
                    "Failed to fetch pool status".red(),
                    response.status()
                );
            }
        }
        Err(e) => {
            println!("  {}", format!("Connection failed: {}", e).red());
            println!("  {}", "Make sure the control plane is running.".yellow());
        }
    }

    println!();
    Ok(())
}

/// Join ring topology for distributed inference
async fn cmd_join_ring(model_id: String, memory: Option<String>) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Joining Ring Topology".bold().cyan());
    println!("{}", "=====================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    println!("\n{}", "Device Identity:".bold());
    println!("  Device ID:     {}", config.device_id);
    println!("  Model ID:      {}", model_id);
    println!("  Connectivity:  {:?}", config.connectivity.preferred_path);
    let contributed_memory = resolve_ring_contributed_memory(&config, memory.as_deref())?;
    println!("  Contribution:  {}", format_bytes(contributed_memory));

    // Connect to control plane
    println!("\n{}", "Requesting ring join...".dimmed());
    let client = control_plane_client();
    let url = control_plane_url(&config, "/api/ring/join");

    #[derive(serde::Serialize)]
    struct JoinRingRequest {
        device_id: String,
        network_id: String,
        model_id: String,
        contributed_memory: u64,
    }

    let request = JoinRingRequest {
        device_id: config.device_id.to_string(),
        network_id: config.network_id.clone(),
        model_id,
        contributed_memory,
    };

    match client.post(&url).json(&request).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(data) = response.json::<serde_json::Value>().await {
                    println!("\n{}", "✓ Joined ring successfully!".green().bold());

                    if let Some(position) = data.get("position") {
                        println!("  Position:        {}", position.to_string().green());
                    }
                    if let Some(shard) = data.get("shard") {
                        if let Some(col_start) = shard.get("column_start") {
                            if let Some(col_end) = shard.get("column_end") {
                                println!("  Column Range:    {} - {}", col_start, col_end);
                            }
                        }
                    }
                    if let Some(left) = data.get("left_neighbor") {
                        println!("  Left Neighbor:   {}", left.as_str().unwrap_or("unknown"));
                    }
                    if let Some(right) = data.get("right_neighbor") {
                        println!("  Right Neighbor:  {}", right.as_str().unwrap_or("unknown"));
                    }

                    println!("\n{}", "Next steps:".bold());
                    println!("  1. Start the agent:     mesh device start");
                    println!("  2. Submit inference:    mesh job run --prompt \"Hello\"");
                }
            } else {
                println!(
                    "  {}",
                    format!("Failed to join ring (HTTP {})", response.status()).red()
                );
                if let Ok(body) = response.text().await {
                    println!("  Error: {}", body);
                }
                anyhow::bail!("Ring join failed");
            }
        }
        Err(e) => {
            println!("  {}", format!("Connection failed: {}", e).red());
            println!("  {}", "Make sure the control plane is running.".yellow());
            return Err(e.into());
        }
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent::api::types::{
        PeerPunchPlan, PunchPathReason, PunchPathStrategy, ShardInfo, WorkerInfo,
    };

    fn test_worker(
        device_id: &str,
        peer_id: &PeerId,
        direct_candidates: Vec<agent::DirectPeerCandidate>,
    ) -> WorkerInfo {
        WorkerInfo {
            device_id: device_id.to_string(),
            peer_id: peer_id.to_string(),
            position: 0,
            status: "online".to_string(),
            contributed_memory: 8_000_000_000,
            shard: ShardInfo {
                model_id: "test-model".to_string(),
                column_start: 0,
                column_end: 4096,
                estimated_memory: 8_000_000_000,
            },
            left_neighbor: device_id.to_string(),
            right_neighbor: device_id.to_string(),
            connectivity_state: None,
            listen_addrs: vec![
                "/ip4/127.0.0.1/tcp/7001/p2p/12D3KooWTestTensor".to_string(),
                "dataplane://127.0.0.1:9001".to_string(),
            ],
            direct_candidates,
        }
    }

    fn tcp_candidate(endpoint: &str) -> agent::DirectPeerCandidate {
        agent::DirectPeerCandidate {
            endpoint: endpoint.to_string(),
            transport: agent::DirectCandidateTransport::Tcp,
            scope: agent::DirectCandidateScope::Private,
            source: agent::DirectCandidateSource::LocalListen,
            priority: 10,
            last_updated_ms: 1_700_000_000_000,
        }
    }

    #[test]
    fn resolve_target_direct_addrs_prefers_punch_plan_candidates() {
        let source_device_id = Uuid::new_v4();
        let source_peer_id = PeerId::random();
        let target_peer_id = PeerId::random();
        let worker = test_worker(
            "target",
            &target_peer_id,
            vec![tcp_candidate("/ip4/10.0.0.8/tcp/4100")],
        );
        let topology = RingTopologyResponse {
            workers: vec![
                test_worker(
                    &source_device_id.to_string(),
                    &source_peer_id,
                    vec![tcp_candidate("/ip4/10.0.0.7/tcp/4100")],
                ),
                worker.clone(),
            ],
            ring_stable: true,
            peer_punch_plans: vec![PeerPunchPlan {
                source_device_id: source_device_id.to_string(),
                target_device_id: worker.device_id.clone(),
                target_peer_id: target_peer_id.to_string(),
                strategy: PunchPathStrategy::SimultaneousDial,
                reason: PunchPathReason::RelayPath,
                relay_rendezvous_required: true,
                attempt_window_ms: 5_000,
                issued_at_ms: 1_700_000_000_000,
                target_candidates: vec![tcp_candidate("/ip4/34.120.0.10/tcp/4001")],
            }],
        };

        let resolved =
            resolve_target_direct_addrs(&topology, &source_device_id, &worker, target_peer_id);

        assert_eq!(resolved.len(), 1);
        assert!(resolved[0].to_string().contains("34.120.0.10"));
    }

    #[test]
    fn build_worker_position_from_topology_attaches_neighbor_punch_plans() {
        let self_device_id = Uuid::new_v4();
        let left_peer_id = PeerId::random();
        let right_peer_id = PeerId::random();

        let topology = RingTopologyResponse {
            workers: vec![
                WorkerInfo {
                    device_id: self_device_id.to_string(),
                    peer_id: PeerId::random().to_string(),
                    position: 0,
                    status: "online".to_string(),
                    contributed_memory: 8_000_000_000,
                    shard: ShardInfo {
                        model_id: "test-model".to_string(),
                        column_start: 0,
                        column_end: 4096,
                        estimated_memory: 8_000_000_000,
                    },
                    left_neighbor: "left".to_string(),
                    right_neighbor: "right".to_string(),
                    connectivity_state: None,
                    listen_addrs: vec!["dataplane://127.0.0.1:9000".to_string()],
                    direct_candidates: vec![tcp_candidate("/ip4/10.0.0.5/tcp/4100")],
                },
                WorkerInfo {
                    device_id: "left".to_string(),
                    peer_id: left_peer_id.to_string(),
                    position: 1,
                    status: "online".to_string(),
                    contributed_memory: 8_000_000_000,
                    shard: ShardInfo {
                        model_id: "test-model".to_string(),
                        column_start: 4096,
                        column_end: 6144,
                        estimated_memory: 8_000_000_000,
                    },
                    left_neighbor: "right".to_string(),
                    right_neighbor: self_device_id.to_string(),
                    connectivity_state: None,
                    listen_addrs: vec!["dataplane://127.0.0.1:9001".to_string()],
                    direct_candidates: vec![tcp_candidate("/ip4/10.0.0.6/tcp/4100")],
                },
                WorkerInfo {
                    device_id: "right".to_string(),
                    peer_id: right_peer_id.to_string(),
                    position: 2,
                    status: "online".to_string(),
                    contributed_memory: 8_000_000_000,
                    shard: ShardInfo {
                        model_id: "test-model".to_string(),
                        column_start: 6144,
                        column_end: 8192,
                        estimated_memory: 8_000_000_000,
                    },
                    left_neighbor: self_device_id.to_string(),
                    right_neighbor: "left".to_string(),
                    connectivity_state: None,
                    listen_addrs: vec!["dataplane://127.0.0.1:9002".to_string()],
                    direct_candidates: vec![tcp_candidate("/ip4/10.0.0.7/tcp/4100")],
                },
            ],
            ring_stable: true,
            peer_punch_plans: vec![
                PeerPunchPlan {
                    source_device_id: self_device_id.to_string(),
                    target_device_id: "left".to_string(),
                    target_peer_id: left_peer_id.to_string(),
                    strategy: PunchPathStrategy::SimultaneousDial,
                    reason: PunchPathReason::RelayPath,
                    relay_rendezvous_required: true,
                    attempt_window_ms: 5_000,
                    issued_at_ms: 1_700_000_000_000,
                    target_candidates: vec![tcp_candidate("/ip4/34.120.0.11/tcp/4001")],
                },
                PeerPunchPlan {
                    source_device_id: self_device_id.to_string(),
                    target_device_id: "right".to_string(),
                    target_peer_id: right_peer_id.to_string(),
                    strategy: PunchPathStrategy::SimultaneousDial,
                    reason: PunchPathReason::PrivateReachabilityOnly,
                    relay_rendezvous_required: false,
                    attempt_window_ms: 5_000,
                    issued_at_ms: 1_700_000_000_000,
                    target_candidates: vec![tcp_candidate("/ip4/34.120.0.12/tcp/4001")],
                },
            ],
        };

        let position = build_worker_position_from_topology(&topology, &self_device_id).unwrap();

        assert!(position.left_neighbor_punch_plan.is_some());
        assert!(position.right_neighbor_punch_plan.is_some());
        assert!(position.left_neighbor_addrs[0]
            .to_string()
            .contains("34.120.0.11"));
        assert!(position.right_neighbor_addrs[0]
            .to_string()
            .contains("34.120.0.12"));
    }
}

/// Leave ring topology
async fn cmd_leave_ring() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Leaving Ring Topology".bold().cyan());
    println!("{}", "=====================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;
    println!("\n{}", "Requesting ring leave...".dimmed());
    let client = control_plane_client();
    let url = control_plane_url(&config, &format!("/api/ring/leave/{}", config.device_id));

    match client.delete(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                println!("\n{}", "✓ Left ring successfully!".green().bold());
            } else {
                println!(
                    "  {}",
                    format!("Failed to leave ring (HTTP {})", response.status()).red()
                );
                if let Ok(body) = response.text().await {
                    println!("  Error: {}", body);
                }
                anyhow::bail!("Ring leave failed");
            }
        }
        Err(e) => {
            println!("  {}", format!("Connection failed: {}", e).red());
            return Err(e.into());
        }
    }

    println!();
    Ok(())
}

/// Submit distributed inference job
async fn cmd_inference(
    prompt: String,
    model_id: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Submitting Inference Job".bold().cyan());
    println!("{}", "========================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    println!("\n{}", "Job Configuration:".bold());
    println!("  Model:           {}", model_id);
    println!("  Prompt:          \"{}\"", prompt);
    println!("  Max Tokens:      {}", max_tokens);
    println!("  Temperature:     {}", temperature);
    println!("  Top-p:           {}", top_p);

    // Create inference request payload
    #[derive(serde::Serialize)]
    struct InferenceJobRequest {
        device_id: String,
        network_id: String,
        model_id: String,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    }

    let request = InferenceJobRequest {
        device_id: config.device_id.to_string(),
        network_id: config.network_id.clone(),
        model_id: model_id.clone(),
        prompt: prompt.clone(),
        max_tokens,
        temperature,
        top_p,
    };

    // Submit to control plane
    println!("\n{}", "Submitting job...".dimmed());
    let client = control_plane_client();
    let url = control_plane_url(&config, "/api/inference/submit");

    match client.post(&url).json(&request).send().await {
        Ok(response) => {
            if response.status().is_success() {
                let submit = response
                    .json::<serde_json::Value>()
                    .await
                    .context("Failed to parse inference submit response")?;
                let job_id = submit
                    .get("job_id")
                    .and_then(|value| value.as_str())
                    .context("Inference submit response did not include job_id")?;

                println!("\n{}", "✓ Inference submitted".green().bold());
                println!("  Job ID:          {}", job_id);
                println!("{}", "Watching distributed job...".dimmed());
                cmd_job_status(job_id, true, 2_000).await?;
            } else {
                println!(
                    "  {}",
                    format!("Inference failed (HTTP {})", response.status()).red()
                );
                if let Ok(body) = response.text().await {
                    println!("  Error: {}", body);
                }
                anyhow::bail!("Inference job failed");
            }
        }
        Err(e) => {
            println!("  {}", format!("Connection failed: {}", e).red());
            println!("  {}", "Make sure the control plane is running.".yellow());
            return Err(e.into());
        }
    }

    println!();
    Ok(())
}

/// Create a new pool (become admin)
async fn cmd_pool_create(name: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Creating New Pool".bold().cyan());
    println!("{}", "=================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    // Create device keypair from config
    let device_keypair = DeviceKeyPair::from_private_bytes(config.keypair.to_bytes())?;

    // Create pool
    let (pool_config, pool_root, membership_cert) =
        PoolConfig::create_pool(name.clone(), &device_keypair)?;

    // Save pool configuration
    pool_config.save(&membership_cert, Some(&pool_root))?;

    println!("\n{}", "✓ Pool created successfully!".green().bold());
    println!("\n{}", "Pool Details:".bold());
    println!(
        "  Pool ID:           {}",
        pool_config.pool_id.to_hex().green()
    );
    println!("  Pool Name:         {}", name);
    println!("  Your Role:         {}", "admin".green());
    println!(
        "\n{}",
        "Pool Root Public Key (share this with team members):".bold()
    );
    println!("  {}", hex::encode(pool_root.public));

    println!("\n{}", "Saved to:".dimmed());
    println!(
        "  {}",
        PoolConfig::pool_dir(&pool_config.pool_id)?.display()
    );

    println!("\n{}", "Next steps:".bold());
    println!("  1. Share the Pool ID and Pool Root Public Key with team members");
    println!("  2. Team members run: mesh pool join \\");
    println!("       --pool-id {} \\", pool_config.pool_id.to_hex());
    println!(
        "       --pool-root-pubkey {}",
        hex::encode(pool_root.public)
    );
    println!("  3. Start the agent to begin LAN discovery: mesh device start");

    println!();
    Ok(())
}

/// Discover pool admin by listening for beacons
async fn discover_pool_admin(
    socket: &tokio::net::UdpSocket,
    pool_id: PoolId,
    timeout_duration: tokio::time::Duration,
) -> Result<agent::discovery::PoolBeacon> {
    use agent::discovery::BeaconMessage;
    use tokio::time::Instant;

    let start = Instant::now();
    let mut buf = vec![0u8; 2048];

    loop {
        if start.elapsed() > timeout_duration {
            anyhow::bail!(
                "Timeout discovering pool admin.\n\nPossible issues:\n  • Pool admin is not online on the same LAN\n  • Pool admin has not started their agent\n  • Firewall is blocking UDP multicast (239.192.0.1:42424)"
            );
        }

        // Listen for beacons with 2-second timeout per attempt
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(2),
            socket.recv_from(&mut buf),
        )
        .await
        {
            Ok(Ok((len, _addr))) => {
                // Try to parse as BeaconMessage
                if let Ok(BeaconMessage::PoolBeacon(beacon)) =
                    ciborium::de::from_reader::<BeaconMessage, _>(&buf[..len])
                {
                    // Check if this is an admin for our pool with capability to sign certs
                    if beacon.pool_id == pool_id
                        && beacon.can_sign_certs()
                        && beacon.is_accepting_joins()
                    {
                        tracing::info!(
                            node_id = %beacon.node_id.to_hex(),
                            "Discovered pool admin via beacon"
                        );
                        return Ok(beacon);
                    }
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(error = ?e, "Socket error while discovering admin");
                continue;
            }
            Err(_) => {
                // Timeout on this attempt, loop and try again
                continue;
            }
        }
    }
}

/// Request certificate from admin with retry mechanism
async fn request_certificate_with_retry(
    send_socket: &tokio::net::UdpSocket,
    recv_socket: &tokio::net::UdpSocket,
    pool_id: PoolId,
    device_keypair: &DeviceKeyPair,
    pool_root_pubkey: [u8; 32],
    multicast_addr: std::net::SocketAddr,
    total_timeout: tokio::time::Duration,
    retry_interval: tokio::time::Duration,
) -> Result<(PoolMembershipCert, std::net::SocketAddr)> {
    use agent::discovery::BeaconMessage;
    use agent::pki::{CertSigningRequest, MembershipRole};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::time::Instant;

    // Create certificate signing request
    let csr = CertSigningRequest::new(pool_id, device_keypair, MembershipRole::Member);
    let beacon_msg = BeaconMessage::CertRequest(csr);
    let mut packet = Vec::new();
    ciborium::ser::into_writer(&beacon_msg, &mut packet)?;

    let start = Instant::now();
    let my_device_pubkey = device_keypair.public;
    let mut buf = vec![0u8; 2048];
    let mut attempt = 1;

    loop {
        if start.elapsed() > total_timeout {
            anyhow::bail!(
                "Timeout waiting for certificate from pool admin.\n\nPossible issues:\n  • Pool admin stopped responding\n  • Pool ID or root pubkey is incorrect\n  • Admin rejected the certificate request"
            );
        }

        // Broadcast certificate request
        send_socket.send_to(&packet, multicast_addr).await?;
        tracing::info!(attempt = attempt, "Sent certificate request to admin");

        // Wait for response with retry_interval timeout
        match tokio::time::timeout(retry_interval, recv_socket.recv_from(&mut buf)).await {
            Ok(Ok((len, sender_addr))) => {
                // Try to parse as CertResponse
                if let Ok(BeaconMessage::CertResponse(cert)) =
                    ciborium::de::from_reader::<BeaconMessage, _>(&buf[..len])
                {
                    // Check if this cert is for us
                    if cert.device_pubkey == my_device_pubkey && cert.pool_id == pool_id {
                        // Verify cert signature
                        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

                        if cert.verify(&pool_root_pubkey, current_time).is_ok() {
                            tracing::info!("Certificate received and verified from admin");

                            return Ok((cert, sender_addr));
                        } else {
                            tracing::warn!("Received certificate failed signature verification");
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                tracing::warn!(error = ?e, "Socket error receiving certificate");
            }
            Err(_) => {
                // Timeout, retry
                attempt += 1;
            }
        }
    }
}

/// Join an existing pool
async fn cmd_pool_join(
    pool_id_hex: String,
    pool_root_pubkey_hex: String,
    name: Option<String>,
) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Joining Pool".bold().cyan());
    println!("{}", "============".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    // Create device keypair from config
    let device_keypair = DeviceKeyPair::from_private_bytes(config.keypair.to_bytes())?;

    // Parse pool ID
    let pool_id = PoolId::from_hex(&pool_id_hex).context("Invalid pool ID hex")?;

    // Parse pool root pubkey
    let pubkey_bytes =
        hex::decode(&pool_root_pubkey_hex).context("Invalid pool root pubkey hex")?;
    if pubkey_bytes.len() != 32 {
        anyhow::bail!("Pool root pubkey must be 32 bytes");
    }
    let mut pool_root_pubkey = [0u8; 32];
    pool_root_pubkey.copy_from_slice(&pubkey_bytes);

    // Verify pool ID matches pubkey
    let expected_pool_id = PoolId::from_pubkey(&pool_root_pubkey);
    if pool_id != expected_pool_id {
        anyhow::bail!("Pool ID does not match pool root pubkey");
    }

    // Setup multicast sockets for LAN discovery
    use agent::discovery::{BEACON_MULTICAST_ADDR, BEACON_MULTICAST_PORT};
    use std::net::{Ipv4Addr, SocketAddr};
    use tokio::net::UdpSocket;
    use tokio::time::Duration;

    println!("\n{}", "Setting up LAN discovery...".cyan());

    // Create sockets for multicast
    let send_socket = UdpSocket::bind("0.0.0.0:0").await?;
    send_socket.set_multicast_ttl_v4(1)?;

    // Create receive socket with socket2 to set reuse options before binding
    // This allows multiple processes to bind to the same multicast port
    let recv_addr: SocketAddr = format!("0.0.0.0:{}", BEACON_MULTICAST_PORT).parse()?;
    let socket = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::DGRAM,
        Some(socket2::Protocol::UDP),
    )?;
    socket.set_reuse_address(true)?;
    #[cfg(unix)]
    socket.set_reuse_port(true)?; // Allow SO_REUSEPORT on macOS/Linux
    socket.set_nonblocking(true)?;
    socket.bind(&recv_addr.into())?;

    // Convert to tokio UdpSocket
    let recv_socket: std::net::UdpSocket = socket.into();
    let recv_socket = UdpSocket::from_std(recv_socket)?;

    // Disable multicast loopback (don't receive our own cert requests)
    // This prevents the socket from receiving packets it sends to the same group
    // Consistent with BeaconListener and BeaconBroadcaster behavior
    recv_socket.set_multicast_loop_v4(false)?;

    recv_socket.join_multicast_v4(BEACON_MULTICAST_ADDR.parse()?, Ipv4Addr::new(0, 0, 0, 0))?;

    let multicast_addr: SocketAddr =
        format!("{}:{}", BEACON_MULTICAST_ADDR, BEACON_MULTICAST_PORT).parse()?;

    // Phase 1: Discover pool admin via beacons
    println!("\n{}", "🔍 Discovering pool admin on LAN...".cyan());
    println!(
        "{}",
        "   Listening for admin beacons (15s timeout)".dimmed()
    );
    println!("{}", "   Make sure Device 1 (admin) is running!".dimmed());

    let admin_beacon = discover_pool_admin(&recv_socket, pool_id, Duration::from_secs(15)).await?;

    println!(
        "{}",
        format!("✓ Found admin node: {}", admin_beacon.node_id.to_hex()).green()
    );

    // Phase 2: Request certificate with retry mechanism
    println!(
        "\n{}",
        "📝 Requesting certificate from pool admin...".cyan()
    );
    println!("{}", "   Retrying every 2s (30s total timeout)".dimmed());

    let (membership_cert, _admin_addr) = request_certificate_with_retry(
        &send_socket,
        &recv_socket,
        pool_id,
        &device_keypair,
        pool_root_pubkey,
        multicast_addr,
        Duration::from_secs(30), // Total timeout
        Duration::from_secs(2),  // Retry interval
    )
    .await?;

    println!(
        "{}",
        "✓ Received signed certificate from admin!".green().bold()
    );

    // Create pool config
    let pool_name = name.unwrap_or_else(|| "Unnamed Pool".to_string());
    let pool_config = PoolConfig::join_pool(
        pool_id,
        pool_name.clone(),
        pool_root_pubkey,
        membership_cert.clone(),
    )?;

    // Save pool configuration with explicit error handling
    let pool_dir = PoolConfig::pool_dir(&pool_id)?;
    tracing::info!(pool_dir = %pool_dir.display(), "Saving pool configuration");

    pool_config.save(&membership_cert, None).context(format!(
        "Failed to save pool configuration to {}",
        pool_dir.display()
    ))?;

    // Verify files were actually written
    let pool_config_file = pool_dir.join("config.toml");
    let cert_file = pool_dir.join("membership.cert");

    if !pool_config_file.exists() {
        anyhow::bail!(
            "Pool config file was not created at {}",
            pool_config_file.display()
        );
    }
    if !cert_file.exists() {
        anyhow::bail!(
            "Certificate file was not created at {}",
            cert_file.display()
        );
    }

    tracing::info!(
        config_file = %pool_config_file.display(),
        cert_file = %cert_file.display(),
        "Pool configuration files verified"
    );

    println!("\n{}", "✓ Joined pool successfully!".green().bold());
    println!("\n{}", "Pool Details:".bold());
    println!("  Pool ID:           {}", pool_id.to_hex().green());
    println!("  Pool Name:         {}", pool_name);
    println!(
        "  Your Role:         {}",
        format!("{:?}", membership_cert.role).to_lowercase().cyan()
    );
    println!(
        "  Expires:           {} days",
        (membership_cert.expires_at - SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
            / 86400
    );

    println!("\n{}", "Saved to:".dimmed());
    println!("  {}", pool_dir.display());
    println!(
        "  Config:      {}",
        pool_config_file.file_name().unwrap().to_string_lossy()
    );
    println!(
        "  Certificate: {}",
        cert_file.file_name().unwrap().to_string_lossy()
    );

    println!("\n{}", "Next steps:".bold());
    println!("  Run the agent to join the mesh: ./agent start");
    println!("  (The agent will automatically broadcast beacons for discovered pools)");

    println!();
    Ok(())
}

/// List all pools
async fn cmd_pool_list() -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Pools".bold().cyan());
    println!("{}", "=====".cyan());

    // Load device config to get node ID
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh device init' first.")?;

    let device_keypair = DeviceKeyPair::from_private_bytes(config.keypair.to_bytes())?;
    let node_id = device_keypair.node_id();

    // List pools
    let pools = PoolConfig::list_pools()?;

    if pools.is_empty() {
        println!("\n{}", "No pools configured.".yellow());
        println!("\n{}", "Create a new pool:".bold());
        println!("  mesh pool create --name \"My Pool\"");
        println!("\n{}", "Or join an existing pool:".bold());
        println!("  mesh pool join --pool-id <id> --pool-root-pubkey <pubkey>");
        println!();
        return Ok(());
    }

    println!("\n{}", format!("Total pools: {}", pools.len()).bold());
    println!();

    for (pool_id, pool_config, cert) in pools {
        println!("{}", format!("Pool: {}", pool_config.name).bold());
        println!("  Pool ID:         {}", pool_id.to_hex().dimmed());
        println!("  Your Node ID:    {}", node_id.to_hex().dimmed());
        println!(
            "  Role:            {}",
            match pool_config.role {
                MembershipRole::Admin => "admin".green(),
                MembershipRole::Member => "member".cyan(),
            }
        );

        if let Some(days) = pool_config.days_until_expiry() {
            if days > 0 {
                println!("  Expires in:      {} days", days);
            } else {
                println!("  Expires in:      {} (EXPIRED)", "0 days".red());
            }
        } else {
            println!("  Expires in:      {}", "Never".green());
        }

        println!("  Created:         {}", pool_config.created_at.dimmed());

        // Check if cert is valid
        if pool_config.is_cert_valid(&cert) {
            println!("  Status:          {}", "Valid".green());
        } else {
            println!("  Status:          {}", "Invalid/Expired".red());
        }

        println!();
    }

    println!("{}", "Commands:".bold());
    println!("  View peers:      mesh pool peers --pool-id <pool-id>");
    println!("  Start discovery: mesh device start");

    println!();
    Ok(())
}

/// Show discovered peers in a pool
async fn cmd_pool_peers(pool_id_hex: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Pool Peers".bold().cyan());
    println!("{}", "==========".cyan());

    // Parse pool ID
    let pool_id = PoolId::from_hex(&pool_id_hex).context("Invalid pool ID hex")?;

    // Load pool config
    let (pool_config, _) = PoolConfig::load(&pool_id)
        .context("Pool not found. Run 'mesh pool join' or 'mesh pool create' first.")?;

    println!("\n{}", format!("Pool: {}", pool_config.name).bold());
    println!("  Pool ID: {}", pool_id.to_hex().dimmed());

    // Load peer cache
    let peer_cache = PeerCache::load(&pool_id)?;

    let peers = peer_cache.get_peers(&pool_id);

    if peers.is_empty() {
        println!("\n{}", "No peers discovered yet.".yellow());
        println!("\n{}", "Start the agent to begin LAN discovery:".bold());
        println!("  mesh device start");
        println!(
            "\n{}",
            "Make sure other devices are on the same LAN and running the agent.".dimmed()
        );
        println!();
        return Ok(());
    }

    println!("\n{}", format!("Discovered peers: {}", peers.len()).bold());
    println!();

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    for peer in peers {
        let age_secs = now.saturating_sub(peer.last_seen);
        let age_str = if age_secs < 60 {
            format!("{}s ago", age_secs)
        } else if age_secs < 3600 {
            format!("{}m ago", age_secs / 60)
        } else {
            format!("{}h ago", age_secs / 3600)
        };

        let status = if age_secs < 30 {
            "ONLINE".green()
        } else if age_secs < 300 {
            "STALE".yellow()
        } else {
            "OFFLINE".red()
        };

        println!("{}", format!("Peer: {}", peer.node_id.to_hex()).bold());
        println!("  LAN Address:     {}", peer.lan_addr);
        println!("  Discovery:       {:?}", peer.discovery_method);
        println!("  Status:          {}", status);
        println!("  Last Seen:       {}", age_str.dimmed());
        println!();
    }

    println!("{}", "Note: Peers are discovered via LAN beacons.".dimmed());
    println!(
        "{}",
        "Ensure all devices are on the same WiFi/Ethernet network.".dimmed()
    );

    println!();
    Ok(())
}
