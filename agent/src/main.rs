//! Mesh AI Agent - Command Line Interface
//!
//! The Mesh AI Agent is a distributed compute sharing daemon that allows devices
//! to contribute spare compute resources for AI workloads (embeddings, OCR, etc.)
//! and earn credits in the process.
//!
//! ## Commands
//!
//! - `init` - Initialize device and register with control plane
//! - `start` - Run agent daemon to process jobs
//! - `job` - Submit a job to the network
//! - `status` - Show device and network status

use agent::{
    DeviceConfig, EmbeddingsExecutor, EmbeddingsInput, JobRunner, MeshSwarmBuilder,
    RegistrationClient,
};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use libp2p::{Multiaddr, PeerId};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
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
            // Simple logging for init
            init_logging("info")?;
            cmd_init(network_id, name, control_plane_url).await?;
        }

        Commands::Start {
            relay,
            control_plane_url,
            log_level,
        } => {
            init_logging(&log_level)?;
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
            init_logging(&log_level)?;
            cmd_job(input, target, workload, timeout_ms, relay).await?;
        }

        Commands::Status => {
            init_logging("info")?;
            cmd_status().await?;
        }
    }

    Ok(())
}

/// Initialize logging with the specified level
fn init_logging(level: &str) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(env_filter)
        .init();

    Ok(())
}

/// Initialize device and register with control plane
async fn cmd_init(network_id: String, name: String, control_plane_url: String) -> Result<()> {
    println!("🔧 Initializing Mesh AI agent...\n");

    // Generate device configuration
    println!("📝 Generating device configuration...");
    let config = DeviceConfig::generate(name.clone(), network_id.clone(), control_plane_url.clone());

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
            eprintln!("   Make sure the control plane is running at {}", control_plane_url);
            return Err(e.into());
        }
    }

    println!("\n✅ Device initialized successfully!");
    println!("\nNext steps:");
    println!("  1. Start the agent:  cargo run --bin agent -- start");
    println!("  2. Submit a job:     cargo run --bin agent -- job --input \"Hello\" --target <peer-id>");

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
    let relay_addr: Multiaddr = relay
        .parse()
        .context("Invalid relay address")?;

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
    let target_peer: PeerId = PeerId::from_str(&target)
        .context("Invalid peer ID format")?;

    println!("📋 Job Details:");
    println!("   Workload: {}", workload);
    println!("   Input: \"{}\"", input);
    println!("   Target: {}", target_peer);
    println!("   Timeout: {}ms", timeout_ms);

    // Parse relay address
    let relay_addr: Multiaddr = relay
        .parse()
        .context("Invalid relay address")?;

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
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs(),
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
                                            preview.get(0).unwrap_or(&0.0),
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

use agent::EmbeddingsOutput;
