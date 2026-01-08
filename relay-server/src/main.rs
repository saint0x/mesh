use clap::Parser;
use futures::StreamExt;
use tokio::signal;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod auth;
mod config;
mod errors;
mod events;
mod relay;

use config::Config;
use errors::Result;

/// Mesh Relay Server - libp2p Circuit Relay v2
#[derive(Parser, Debug)]
#[command(name = "relay-server")]
#[command(about = "Mesh Relay Server for NAT traversal and device connectivity")]
#[command(version)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "~/.meshnet/relay.toml")]
    config: String,

    /// Override TCP port (overrides config file)
    #[arg(short, long)]
    port: Option<u16>,

    /// Override log level (trace, debug, info, warn, error)
    #[arg(short, long)]
    log_level: Option<String>,

    /// Generate default config and exit
    #[arg(long)]
    generate_config: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle config generation
    if cli.generate_config {
        let config = Config::default();
        let path = shellexpand::tilde(&cli.config);
        let path = std::path::Path::new(path.as_ref());

        config.save(path)?;
        println!("Generated default configuration at: {}", path.display());
        println!("\nEdit the configuration file and then start the relay server with:");
        println!("  relay-server --config {}", cli.config);
        return Ok(());
    }

    // Load configuration
    let config_path = shellexpand::tilde(&cli.config);
    let config_path = std::path::Path::new(config_path.as_ref());

    let mut config = if config_path.exists() {
        Config::load(config_path)?
    } else {
        // Auto-generate default config on first run
        let config = Config::default();
        config.save(config_path)?;

        println!("First run detected - created default configuration at: {}", config_path.display());
        println!("Edit {} to customize relay settings\n", config_path.display());

        config
    };

    // Override port if specified via CLI
    if let Some(port) = cli.port {
        config.network.tcp_listen_addr = format!("/ip4/0.0.0.0/tcp/{}", port);
        config.network.quic_listen_addr = format!("/ip4/0.0.0.0/udp/{}/quic-v1", port);
        tracing::info!(port = port, "Port overridden via CLI argument");
    }

    // Setup logging
    setup_logging(&config, cli.log_level.as_deref())?;

    tracing::info!("Starting Mesh Relay Server");
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "Version");

    // Log configuration
    tracing::info!(
        max_reservations = config.relay.max_reservations,
        max_per_peer = config.relay.max_reservations_per_peer,
        max_circuits = config.relay.max_circuits_per_peer,
        circuit_duration_secs = config.relay.max_circuit_duration_secs,
        circuit_bytes = config.relay.max_circuit_bytes,
        "Relay configuration"
    );

    tracing::info!(
        tcp = %config.network.tcp_listen_addr,
        quic = %config.network.quic_listen_addr,
        "Network configuration"
    );

    if config.auth.auth_enabled {
        tracing::warn!("Token authentication enabled");
    } else {
        tracing::info!("Token authentication disabled (public relay)");
    }

    // Build relay swarm
    let mut swarm = relay::build_swarm(&config).await?;

    // Parse listen addresses
    let tcp_addr: libp2p::Multiaddr = config.network.tcp_listen_addr.parse()
        .map_err(|e| errors::RelayError::Config(format!("Invalid TCP address: {}", e)))?;

    let quic_addr: libp2p::Multiaddr = config.network.quic_listen_addr.parse()
        .map_err(|e| errors::RelayError::Config(format!("Invalid QUIC address: {}", e)))?;

    // Start listening
    swarm.listen_on(tcp_addr)
        .map_err(|e| errors::RelayError::Transport(format!("Failed to listen on TCP: {}", e)))?;

    // QUIC is optional - warn if it fails but don't exit
    if let Err(e) = swarm.listen_on(quic_addr) {
        tracing::warn!(error = %e, "Failed to listen on QUIC, continuing with TCP only");
    }

    tracing::info!("Relay server started successfully");
    println!("\nMesh Relay Server is running!");
    println!("Press Ctrl+C to stop\n");

    // Main event loop
    loop {
        tokio::select! {
            // Handle graceful shutdown
            _ = signal::ctrl_c() => {
                tracing::info!("Received shutdown signal (Ctrl+C)");
                println!("\nShutting down relay server...");
                break;
            }

            // Handle swarm events
            event = swarm.select_next_some() => {
                events::handle_swarm_event(event).await;
            }
        }
    }

    tracing::info!("Relay server stopped");
    Ok(())
}

/// Setup logging based on configuration
fn setup_logging(config: &Config, log_level_override: Option<&str>) -> Result<()> {
    let log_level = log_level_override.unwrap_or(&config.logging.level);

    let env_filter = EnvFilter::try_new(log_level)
        .unwrap_or_else(|_| EnvFilter::new("info"));

    if config.logging.log_format == "json" {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().pretty())
            .init();
    }

    Ok(())
}
