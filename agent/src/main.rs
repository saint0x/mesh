//! Mesh AI Agent - Command Line Interface
//!
//! The Mesh AI Agent is a distributed compute sharing daemon that allows devices
//! to contribute spare compute resources for AI workloads (embeddings, OCR, etc.)
//! and distributed tensor-parallel inference.
//!
//! ## Commands
//!
//! ### Device Management
//! - `init` - Initialize device and register with control plane
//! - `start` - Run agent daemon to process jobs
//! - `status` - Show device and network status
//! - `metrics` - Show agent metrics and statistics
//!
//! ### Job Submission
//! - `job` - Submit a job to the network (embeddings, OCR)
//! - `inference` - Submit distributed inference job
//!
//! ### Ring Topology
//! - `join-ring` - Join ring topology for distributed inference
//! - `leave-ring` - Leave ring topology
//! - `ring-status` - Show ring topology and worker position
//! - `pool-status` - Show pool status (all workers)
//!
//! ### Resource Management
//! - `lock-resources` - Lock resources for pool contribution
//! - `unlock-resources` - Request unlock (requires 24h cooldown)
//! - `resource-status` - Show resource lock status
//!
//! ### Model Shards
//! - `shard-status` - Show this worker's shard assignment
//! - `inference-stats` - Show inference statistics

use agent::{
    format_bytes, init_production_logging, init_simple_logging, parse_memory_string,
    DeviceConfig, EmbeddingsExecutor, EmbeddingsInput, EmbeddingsOutput, JobRunner,
    MeshSwarmBuilder, RegistrationClient, ResourceManager,
};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use libp2p::{Multiaddr, PeerId};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Mesh AI Agent - Distributed compute sharing network
#[derive(Parser, Debug)]
#[command(name = "mesh")]
#[command(about = "Mesh AI distributed compute agent", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Initialize device and register with control plane
    Init {
        /// Network ID to join
        #[arg(short, long)]
        network_id: String,

        /// Device name (e.g., "My Laptop")
        #[arg(short = 'd', long, default_value = "My Device")]
        name: String,

        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,
    },

    /// Run agent daemon to process jobs
    Start {
        /// Relay server address (multiaddr format)
        #[arg(short, long, default_value = "/ip4/127.0.0.1/tcp/4001")]
        relay: String,

        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,

        /// Log level (trace, debug, info, warn, error)
        #[arg(short, long, default_value = "info")]
        log_level: String,
    },

    /// Submit a job to the network
    Job {
        /// Job input text
        #[arg(short, long)]
        input: String,

        /// Target peer ID (PeerId format)
        #[arg(short, long)]
        target: String,

        /// Workload type (embeddings, ocr, chat)
        #[arg(short, long, default_value = "embeddings")]
        workload: String,

        /// Job timeout in milliseconds
        #[arg(long, default_value = "5000")]
        timeout_ms: u64,

        /// Relay server address
        #[arg(short, long, default_value = "/ip4/127.0.0.1/tcp/4001")]
        relay: String,

        /// Log level
        #[arg(long, default_value = "info")]
        log_level: String,
    },

    /// Show device and network status
    Status,

    /// Show agent metrics and statistics
    Metrics,

    /// Lock resources for pool contribution
    LockResources {
        /// Amount of memory to lock (e.g., "7GB", "512MB", or bytes)
        #[arg(short, long)]
        memory: String,
    },

    /// Request unlock (requires 24h cooldown)
    UnlockResources,

    /// Show resource lock status
    ResourceStatus,

    /// Show ring topology and worker position
    RingStatus,

    /// Show this worker's shard assignment
    ShardStatus,

    /// Show inference statistics
    InferenceStats,

    /// Show pool status (all workers in the ring)
    PoolStatus {
        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,
    },

    /// Join ring topology for distributed inference
    JoinRing {
        /// Model ID (e.g., "llama-70b")
        #[arg(short, long)]
        model_id: String,

        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,

        /// Relay server address
        #[arg(short, long, default_value = "/ip4/127.0.0.1/tcp/4001")]
        relay: String,

        /// Log level
        #[arg(short, long, default_value = "info")]
        log_level: String,
    },

    /// Leave ring topology
    LeaveRing {
        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,
    },

    /// Submit distributed inference job
    Inference {
        /// Prompt text
        #[arg(short, long)]
        prompt: String,

        /// Model ID
        #[arg(short, long, default_value = "llama-70b")]
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

        /// Control plane URL
        #[arg(short, long = "control-plane", default_value = "http://localhost:8080")]
        control_plane_url: String,

        /// Log level
        #[arg(short, long, default_value = "info")]
        log_level: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init {
            network_id,
            name,
            control_plane_url,
        } => {
            // Simple logging for init command
            init_simple_logging("info")?;
            cmd_init(network_id, name, control_plane_url).await?;
        }

        Commands::Start {
            relay,
            control_plane_url,
            log_level,
        } => {
            // Production logging with file rotation for daemon
            init_production_logging(&log_level, None)?;
            cmd_start(relay, control_plane_url).await?;
        }

        Commands::Job {
            input,
            target,
            workload,
            timeout_ms,
            relay,
            log_level,
        } => {
            init_simple_logging(&log_level)?;
            cmd_job(input, target, workload, timeout_ms, relay).await?;
        }

        Commands::Status => {
            init_simple_logging("info")?;
            cmd_status().await?;
        }

        Commands::Metrics => {
            // No logging for metrics (pure display)
            cmd_metrics().await?;
        }

        Commands::LockResources { memory } => {
            init_simple_logging("info")?;
            cmd_lock_resources(memory).await?;
        }

        Commands::UnlockResources => {
            init_simple_logging("info")?;
            cmd_unlock_resources().await?;
        }

        Commands::ResourceStatus => {
            // No logging for status (pure display)
            cmd_resource_status().await?;
        }

        Commands::RingStatus => {
            // No logging for status (pure display)
            cmd_ring_status().await?;
        }

        Commands::ShardStatus => {
            // No logging for status (pure display)
            cmd_shard_status().await?;
        }

        Commands::InferenceStats => {
            // No logging for stats (pure display)
            cmd_inference_stats().await?;
        }

        Commands::PoolStatus { control_plane_url } => {
            init_simple_logging("info")?;
            cmd_pool_status(control_plane_url).await?;
        }

        Commands::JoinRing {
            model_id,
            control_plane_url,
            relay,
            log_level,
        } => {
            init_simple_logging(&log_level)?;
            cmd_join_ring(model_id, control_plane_url, relay).await?;
        }

        Commands::LeaveRing { control_plane_url } => {
            init_simple_logging("info")?;
            cmd_leave_ring(control_plane_url).await?;
        }

        Commands::Inference {
            prompt,
            model_id,
            max_tokens,
            temperature,
            top_p,
            control_plane_url,
            log_level,
        } => {
            init_simple_logging(&log_level)?;
            cmd_inference(prompt, model_id, max_tokens, temperature, top_p, control_plane_url).await?;
        }
    }

    Ok(())
}

/// Initialize device and register with control plane
async fn cmd_init(network_id: String, name: String, control_plane_url: String) -> Result<()> {
    println!("🔧 Initializing Mesh AI agent...\n");

    // Generate device configuration
    println!("📝 Generating device configuration...");
    let config =
        DeviceConfig::generate(name.clone(), network_id.clone(), control_plane_url.clone());

    println!("   Device ID: {}", config.device_id);
    println!("   Network ID: {}", network_id);
    println!("   Name: {}", name);
    println!("   Tier: {:?}", config.capabilities.tier);
    println!("   CPU Cores: {}", config.capabilities.cpu_cores);
    println!("   RAM: {} MB", config.capabilities.ram_mb);

    // Save configuration
    let config_path = DeviceConfig::default_path()?;
    config.save(&config_path)?;
    println!("\n✓ Configuration saved to: {}", config_path.display());

    // Register with control plane
    println!("\n🌐 Registering with control plane...");
    println!("   URL: {}", control_plane_url);

    let client = RegistrationClient::new(control_plane_url.clone())?;

    match client.register(&config).await {
        Ok(signed_cert) => {
            // Save certificate
            config.save_certificate(&signed_cert)?;

            let cert_path = DeviceConfig::default_certificate_path()?;
            println!("✓ Registration successful!");
            println!("   Certificate saved to: {}", cert_path.display());
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
    println!("  1. Start the agent:  cargo run --bin agent -- start");
    println!(
        "  2. Submit a job:     cargo run --bin agent -- job --input \"Hello\" --target <peer-id>"
    );

    Ok(())
}

/// Run agent daemon
async fn cmd_start(relay: String, _control_plane_url: String) -> Result<()> {
    println!("🚀 Starting Mesh AI agent daemon...\n");

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh init' first.")?;

    info!(
        device_id = %config.device_id,
        network_id = %config.network_id,
        name = %config.name,
        "Loaded device configuration"
    );

    println!("📋 Device: {} ({})", config.name, config.device_id);
    println!("   Network: {}", config.network_id);
    println!("   Tier: {:?}", config.capabilities.tier);

    // Parse relay address
    let relay_addr: Multiaddr = relay.parse().context("Invalid relay address")?;

    println!("\n🌐 Connecting to relay server...");
    println!("   Relay: {}", relay_addr);

    // Create swarm
    let libp2p_keypair = agent::device::keypair::to_libp2p_keypair(&config.keypair);
    let mut swarm = MeshSwarmBuilder::new(libp2p_keypair)
        .with_relay_addr(relay_addr.clone())
        .build()?;

    let local_peer_id = swarm.local_peer_id();
    println!("   Local PeerID: {}", local_peer_id);

    // Connect to relay
    swarm.connect_to_relay()?;

    // Wait for relay connection event
    println!("   Waiting for relay connection...");
    let mut relay_connected = false;
    while !relay_connected {
        if let Some(event) = swarm.next_event().await {
            match event {
                agent::MeshEvent::RelayConnected { .. } => {
                    println!("   ✓ Connected to relay");
                    relay_connected = true;
                }
                agent::MeshEvent::RelayConnectionFailed { error, .. } => {
                    error!(error = %error, "Failed to connect to relay");
                    anyhow::bail!("Relay connection failed: {}", error);
                }
                _ => {}
            }
        }
    }

    // Listen on relay (create reservation)
    let connected_peers = swarm.connected_peers();
    let relay_peer_id = connected_peers
        .first()
        .copied()
        .context("No relay peer connected")?;

    println!("   Creating relay reservation...");
    swarm.listen_on_relay(relay_peer_id)?;

    // Wait for reservation
    let mut reservation_accepted = false;
    while !reservation_accepted {
        if let Some(event) = swarm.next_event().await {
            match event {
                agent::MeshEvent::ReservationAccepted { .. } => {
                    println!("   ✓ Relay reservation accepted");
                    reservation_accepted = true;
                }
                agent::MeshEvent::ReservationDenied { .. } => {
                    error!("Relay reservation denied");
                    anyhow::bail!("Relay reservation denied");
                }
                _ => {}
            }
        }
    }

    // Start heartbeat (in background)
    let heartbeat_config = config.clone();
    tokio::spawn(async move {
        let client = match RegistrationClient::new(heartbeat_config.control_plane_url.clone()) {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to create heartbeat client: {}", e);
                return;
            }
        };
        loop {
            if let Err(e) = client.heartbeat(heartbeat_config.device_id).await {
                error!(error = %e, "Heartbeat failed");
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        }
    });

    // Create executor
    let executor = EmbeddingsExecutor::new()?;
    println!("\n🤖 Executor initialized");
    println!("   Model: {}", executor.model_name());
    println!("   Dimensions: {}", executor.dimensions());

    // Create and run job runner
    println!("\n✅ Agent ready - waiting for jobs...");
    println!("   Press Ctrl+C to stop\n");

    let runner = JobRunner::new(swarm, executor);
    runner.run().await?;

    Ok(())
}

/// Submit a job to the network
async fn cmd_job(
    input: String,
    target: String,
    workload: String,
    timeout_ms: u64,
    relay: String,
) -> Result<()> {
    println!("📤 Submitting job to network...\n");

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh init' first.")?;

    // Parse target peer ID
    let target_peer: PeerId = PeerId::from_str(&target).context("Invalid peer ID format")?;

    println!("📋 Job Details:");
    println!("   Workload: {}", workload);
    println!("   Input: \"{}\"", input);
    println!("   Target: {}", target_peer);
    println!("   Timeout: {}ms", timeout_ms);

    // Parse relay address
    let relay_addr: Multiaddr = relay.parse().context("Invalid relay address")?;

    // Create ephemeral swarm for job submission
    let libp2p_keypair = agent::device::keypair::to_libp2p_keypair(&config.keypair);
    let mut swarm = MeshSwarmBuilder::new(libp2p_keypair)
        .with_relay_addr(relay_addr.clone())
        .build()?;

    println!("\n🌐 Connecting to relay...");
    swarm.connect_to_relay()?;

    // Wait for relay connection
    let mut relay_connected = false;
    while !relay_connected {
        if let Some(event) = swarm.next_event().await {
            if matches!(event, agent::MeshEvent::RelayConnected { .. }) {
                println!("   ✓ Connected to relay");
                relay_connected = true;
            }
        }
    }

    // Create job envelope
    let embeddings_input = EmbeddingsInput::new(input.clone());
    let payload = embeddings_input.to_cbor()?;

    let job = agent::network::JobEnvelope {
        job_id: Uuid::new_v4(),
        network_id: config.network_id.clone(),
        workload_id: workload.clone(),
        payload,
        timeout_ms,
        auth_signature: vec![], // TODO: implement signature
        created_at: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
    };

    println!("   Job ID: {}", job.job_id);

    // Send job
    println!("\n📡 Sending job...");
    swarm.send_job(target_peer, job.clone())?;

    // Wait for result
    println!("   Waiting for result (timeout: {}ms)...", timeout_ms);

    let start = std::time::Instant::now();
    let deadline = std::time::Duration::from_millis(timeout_ms + 5000); // Add 5s buffer

    while start.elapsed() < deadline {
        if let Some(event) = swarm.next_event().await {
            match event {
                agent::MeshEvent::JobCompleted { result, .. } => {
                    if result.job_id == job.job_id {
                        if result.success {
                            println!("\n✅ Job completed successfully!");
                            println!("   Execution time: {}ms", result.execution_time_ms);

                            if let Some(result_bytes) = result.result {
                                match EmbeddingsOutput::from_cbor(&result_bytes) {
                                    Ok(output) => {
                                        let preview: Vec<f32> =
                                            output.embedding.iter().take(5).copied().collect();
                                        println!("   Model: {}", output.model);
                                        println!("   Dimensions: {}", output.dimensions);
                                        println!(
                                            "   Embedding preview: [{:.3}, {:.3}, {:.3}, ...]",
                                            preview.first().unwrap_or(&0.0),
                                            preview.get(1).unwrap_or(&0.0),
                                            preview.get(2).unwrap_or(&0.0),
                                        );
                                    }
                                    Err(e) => {
                                        eprintln!("   Warning: Failed to parse result: {}", e);
                                    }
                                }
                            }
                        } else {
                            println!("\n✗ Job failed!");
                            if let Some(error) = result.error {
                                println!("   Error: {}", error);
                            }
                        }

                        return Ok(());
                    }
                }
                agent::MeshEvent::JobSendFailed { job_id, error, .. } => {
                    if job_id == job.job_id {
                        error!(error = %error, "Job send failed");
                        anyhow::bail!("Failed to send job: {}", error);
                    }
                }
                _ => {}
            }
        }
    }

    eprintln!("\n✗ Job timed out after {}ms", deadline.as_millis());
    anyhow::bail!("Job timed out");
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
                println!("\n⚠️  Certificate: Missing (run 'mesh init' again)");
            }

            println!("\n📡 Control Plane: {}", config.control_plane_url);
        }
        Err(_) => {
            println!("⚠️  Device not initialized");
            println!("\nRun 'mesh init' to set up this device.");
        }
    }

    Ok(())
}

/// Display agent metrics
async fn cmd_metrics() -> Result<()> {
    use colored::Colorize;
    use std::fs;
    use std::path::PathBuf;

    // Stats file path
    let stats_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".meshnet")
        .join("stats.json");

    if !stats_path.exists() {
        println!("{}", "No metrics available".yellow());
        println!("\nThe agent hasn't been started yet, or metrics haven't been saved.");
        println!("Run 'mesh start' to begin collecting metrics.\n");
        return Ok(());
    }

    // Read stats file
    let stats_json = fs::read_to_string(&stats_path).context("Failed to read stats file")?;

    let stats: SavedStats =
        serde_json::from_str(&stats_json).context("Failed to parse stats file")?;

    // Display metrics
    println!("\n{}", "Agent Metrics".bold().cyan());
    println!("{}", "=============".cyan());

    println!("\n{}", "Job Statistics:".bold());
    println!("  Total Jobs:       {}", stats.total_jobs);
    println!(
        "  Completed:        {}",
        stats.completed.to_string().green()
    );
    println!("  Failed:           {}", stats.failed.to_string().red());
    println!("  Active:           {}", stats.active);
    println!("  Success Rate:     {:.1}%", stats.success_rate);

    println!("\n{}", "Performance:".bold());
    println!("  Avg Execution:    {:.2}ms", stats.avg_execution_time_ms);
    println!("  Total CPU Time:   {}ms", stats.total_execution_time_ms);

    println!("\n{}", "System:".bold());
    println!("  Uptime:           {}", stats.uptime);
    println!("  Last Updated:     {}", stats.last_updated);
    println!();

    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SavedStats {
    total_jobs: u64,
    completed: u64,
    failed: u64,
    active: u64,
    success_rate: f64,
    avg_execution_time_ms: f64,
    total_execution_time_ms: u64,
    uptime: String,
    last_updated: String,
}

/// Lock resources for pool contribution
async fn cmd_lock_resources(memory: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Locking Resources".bold().cyan());
    println!("{}", "=================".cyan());

    // Parse memory string
    let bytes = parse_memory_string(&memory)
        .context("Invalid memory format")?;

    // Create and configure resource manager
    let mut manager = ResourceManager::new()
        .context("Failed to initialize resource manager")?;

    manager.load_config()
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
    manager.set_allocation(bytes)
        .context("Failed to set allocation")?;

    println!("\n{}", "Allocation:".bold());
    println!("  Requested:     {}", format_bytes(manager.user_allocated()));
    println!("  With buffer:   {} (7% safety margin)", format_bytes(manager.locked_memory()));
    println!("  Total system:  {}", format_bytes(manager.total_memory()));

    // Lock memory
    println!("\n{}", "Locking memory...".bold());

    match manager.lock_memory() {
        Ok(()) => {
            println!("\n{}", "Memory locked successfully!".green().bold());
            println!("  Lock timestamp: {:?}", manager.lock_timestamp());
            println!("  Cooldown period: 24 hours");
            println!("\n{}", "Note: Memory will remain locked for 24 hours.".yellow());
            println!("{}", "Use 'mesh resource-status' to check lock status.".yellow());
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
    let mut manager = ResourceManager::new()
        .context("Failed to initialize resource manager")?;

    manager.load_config()
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
            println!("\n{}", "The 24-hour cooldown period has not elapsed.".yellow());
            println!("{}", "This prevents frequent resource changes in the pool.".yellow());
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
    let mut manager = ResourceManager::new()
        .context("Failed to initialize resource manager")?;

    manager.load_config()
        .context("Failed to load resource config")?;

    println!("\n{}", "System Memory:".bold());
    println!("  Total:         {}", format_bytes(manager.total_memory()));

    println!("\n{}", "Allocation:".bold());
    println!("  User request:  {}", format_bytes(manager.user_allocated()));
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
            println!("Run 'mesh init' to set up this device.\n");
            return Ok(());
        }
    };

    println!("\n{}", "Worker Identity:".bold());
    println!("  Device ID:     {}", config.device_id);
    println!("  Network ID:    {}", config.network_id);

    // Try to load ring position from saved state
    let ring_state_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".meshnet")
        .join("ring_state.json");

    if ring_state_path.exists() {
        match std::fs::read_to_string(&ring_state_path) {
            Ok(data) => {
                if let Ok(state) = serde_json::from_str::<serde_json::Value>(&data) {
                    println!("\n{}", "Ring Position:".bold());
                    if let Some(pos) = state.get("position") {
                        println!("  Position:      {}", pos.to_string().green());
                    }
                    if let Some(total) = state.get("total_workers") {
                        println!("  Total Workers: {}", total);
                    }
                    if let Some(left) = state.get("left_neighbor") {
                        println!("  Left Neighbor: {}", left.as_str().unwrap_or("unknown"));
                    }
                    if let Some(right) = state.get("right_neighbor") {
                        println!("  Right Neighbor: {}", right.as_str().unwrap_or("unknown"));
                    }
                } else {
                    println!("\n{}", "Not currently in a ring".yellow());
                }
            }
            Err(_) => {
                println!("\n{}", "Not currently in a ring".yellow());
            }
        }
    } else {
        println!("\n{}", "Ring Status:".bold());
        println!("  Status:        {}", "NOT JOINED".yellow());
        println!("\n{}", "To join a ring, start the agent with 'mesh start'".dimmed());
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
            println!("\n{}", format!("Failed to load shard registry: {}", e).red());
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
        println!("{}", "Shards are assigned when you join a ring topology.".dimmed());
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
        println!("{}", "Statistics are recorded during inference jobs.".dimmed());
        println!();
        return Ok(());
    }

    let stats_json = std::fs::read_to_string(&stats_path)
        .context("Failed to read inference stats file")?;

    let stats: serde_json::Value = serde_json::from_str(&stats_json)
        .context("Failed to parse inference stats")?;

    println!("\n{}", "Job Statistics:".bold());
    if let Some(completed) = stats.get("jobs_completed") {
        println!("  Completed:       {}", completed.to_string().green());
    }
    if let Some(failed) = stats.get("jobs_failed") {
        println!("  Failed:          {}", failed.to_string().red());
    }
    if let Some(rate) = stats.get("success_rate") {
        println!("  Success Rate:    {:.1}%", rate.as_f64().unwrap_or(0.0) * 100.0);
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
        println!("  Avg Latency:      {:.2}ms", latency.as_f64().unwrap_or(0.0));
    }
    if let Some(layers) = stats.get("total_layers_processed") {
        println!("  Layers Processed: {}", layers);
    }

    println!("\n{}", "Fault Tolerance:".bold());
    if let Some(ckpts) = stats.get("checkpoints_created") {
        println!("  Checkpoints:      {}", ckpts);
    }
    if let Some(recoveries) = stats.get("checkpoint_recoveries") {
        println!("  Recoveries:       {}", recoveries);
    }

    println!("\n{}", "System:".bold());
    if let Some(uptime) = stats.get("uptime") {
        println!("  Uptime:           {}", uptime.as_str().unwrap_or("unknown"));
    }
    if let Some(updated) = stats.get("last_updated") {
        println!("  Last Updated:     {}", updated.as_str().unwrap_or("unknown"));
    }

    println!();
    Ok(())
}

/// Show pool status (all workers from control plane)
async fn cmd_pool_status(control_plane_url: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Pool Status".bold().cyan());
    println!("{}", "===========".cyan());

    // Load device configuration to get network ID
    let config_path = DeviceConfig::default_path()?;
    let config = match DeviceConfig::load(&config_path) {
        Ok(c) => c,
        Err(_) => {
            println!("\n{}", "Device not initialized".yellow());
            println!("Run 'mesh init' to set up this device.\n");
            return Ok(());
        }
    };

    println!("\n{}", "Fetching pool status...".dimmed());

    // Fetch ring topology from control plane
    let client = reqwest::Client::new();
    let url = format!(
        "{}/api/ring/{}",
        control_plane_url.trim_end_matches('/'),
        config.network_id
    );

    match client.get(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(data) = response.json::<serde_json::Value>().await {
                    println!("\n{}", "Ring Topology:".bold());

                    if let Some(workers) = data.get("workers").and_then(|w| w.as_array()) {
                        println!("  Total Workers: {}", workers.len().to_string().green());
                        println!();

                        for (i, worker) in workers.iter().enumerate() {
                            let device_id = worker
                                .get("device_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let position = worker
                                .get("ring_position")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let memory = worker
                                .get("locked_memory")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let status = worker
                                .get("status")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");

                            let is_self = device_id == config.device_id.to_string();
                            let prefix = if is_self { "→ " } else { "  " };

                            let status_colored = match status {
                                "online" => "ONLINE".green().to_string(),
                                "offline" => "OFFLINE".red().to_string(),
                                _ => status.yellow().to_string(),
                            };

                            println!(
                                "{}Worker {} (pos {}): {} - {} RAM",
                                prefix,
                                i,
                                position,
                                status_colored,
                                format_bytes(memory)
                            );
                        }
                    } else {
                        println!("  {}", "No workers in ring".yellow());
                    }

                    // Show shard distribution if available
                    if let Some(total_columns) = data.get("total_columns").and_then(|v| v.as_u64()) {
                        println!("\n{}", "Shard Distribution:".bold());
                        println!("  Total Columns: {}", total_columns);
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
async fn cmd_join_ring(model_id: String, control_plane_url: String, _relay: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Joining Ring Topology".bold().cyan());
    println!("{}", "=====================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh init' first.")?;

    println!("\n{}", "Device Identity:".bold());
    println!("  Device ID:     {}", config.device_id);
    println!("  Model ID:      {}", model_id);

    // Connect to control plane
    println!("\n{}", "Requesting ring join...".dimmed());
    let client = reqwest::Client::new();
    let url = format!(
        "{}/api/ring/join",
        control_plane_url.trim_end_matches('/')
    );

    #[derive(serde::Serialize)]
    struct JoinRingRequest {
        device_id: String,
        model_id: String,
    }

    let request = JoinRingRequest {
        device_id: config.device_id.to_string(),
        model_id: model_id.clone(),
    };

    match client.post(&url).json(&request).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(data) = response.json::<serde_json::Value>().await {
                    println!("\n{}", "✓ Joined ring successfully!".green().bold());

                    if let Some(position) = data.get("position") {
                        println!("  Position:        {}", position.to_string().green());
                    }
                    if let Some(total) = data.get("total_workers") {
                        println!("  Total Workers:   {}", total);
                    }
                    if let Some(col_start) = data.get("column_start") {
                        if let Some(col_end) = data.get("column_end") {
                            println!("  Column Range:    {} - {}", col_start, col_end);
                        }
                    }

                    // Save ring state locally
                    let ring_state_path = dirs::home_dir()
                        .unwrap_or_else(|| std::path::PathBuf::from("."))
                        .join(".meshnet")
                        .join("ring_state.json");

                    if let Some(parent) = ring_state_path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }

                    std::fs::write(&ring_state_path, serde_json::to_string_pretty(&data)?)?;
                    info!("Ring state saved to: {}", ring_state_path.display());

                    println!("\n{}", "Next steps:".bold());
                    println!("  1. Start the agent:     cargo run --bin agent -- start");
                    println!("  2. Submit inference:    cargo run --bin agent -- inference --prompt \"Hello\"");
                }
            } else {
                println!("  {}", format!("Failed to join ring (HTTP {})", response.status()).red());
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

/// Leave ring topology
async fn cmd_leave_ring(control_plane_url: String) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Leaving Ring Topology".bold().cyan());
    println!("{}", "=====================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh init' first.")?;

    println!("\n{}", "Requesting ring leave...".dimmed());
    let client = reqwest::Client::new();
    let url = format!(
        "{}/api/ring/leave/{}",
        control_plane_url.trim_end_matches('/'),
        config.device_id
    );

    match client.delete(&url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                println!("\n{}", "✓ Left ring successfully!".green().bold());

                // Remove local ring state
                let ring_state_path = dirs::home_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join(".meshnet")
                    .join("ring_state.json");

                if ring_state_path.exists() {
                    std::fs::remove_file(&ring_state_path)?;
                    info!("Removed ring state file");
                }
            } else {
                println!("  {}", format!("Failed to leave ring (HTTP {})", response.status()).red());
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
    control_plane_url: String,
) -> Result<()> {
    use colored::Colorize;

    println!("\n{}", "Submitting Inference Job".bold().cyan());
    println!("{}", "========================".cyan());

    // Load device configuration
    let config_path = DeviceConfig::default_path()?;
    let config = DeviceConfig::load(&config_path)
        .context("Failed to load device config. Run 'mesh init' first.")?;

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
        model_id: String,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    }

    let request = InferenceJobRequest {
        device_id: config.device_id.to_string(),
        model_id: model_id.clone(),
        prompt: prompt.clone(),
        max_tokens,
        temperature,
        top_p,
    };

    // Submit to control plane
    println!("\n{}", "Submitting job...".dimmed());
    let client = reqwest::Client::new();
    let url = format!(
        "{}/api/inference/submit",
        control_plane_url.trim_end_matches('/')
    );

    match client.post(&url).json(&request).send().await {
        Ok(response) => {
            if response.status().is_success() {
                if let Ok(result) = response.json::<serde_json::Value>().await {
                    println!("\n{}", "✓ Inference completed!".green().bold());

                    if let Some(job_id) = result.get("job_id") {
                        println!("  Job ID:          {}", job_id.as_str().unwrap_or("unknown"));
                    }

                    if let Some(tokens_generated) = result.get("completion_tokens") {
                        println!("  Tokens Generated: {}", tokens_generated);
                    }

                    if let Some(exec_time) = result.get("execution_time_ms") {
                        println!("  Execution Time:   {}ms", exec_time);
                    }

                    if let Some(completion) = result.get("completion") {
                        println!("\n{}", "Completion:".bold());
                        println!("{}", completion.as_str().unwrap_or(""));
                    } else if let Some(output) = result.get("output") {
                        println!("\n{}", "Output:".bold());
                        println!("{}", output.as_str().unwrap_or(""));
                    }

                    // Show that this used mock weights
                    println!("\n{}", "Note: This inference used mock Xavier-initialized weights.".yellow());
                    println!("{}", "Output is deterministic but not semantically coherent.".yellow());
                    println!("{}", "Swap in real safetensors weights for production use.".yellow());
                } else {
                    println!("  {}", "Failed to parse response".red());
                }
            } else {
                println!("  {}", format!("Inference failed (HTTP {})", response.status()).red());
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
