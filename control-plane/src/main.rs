use clap::Parser;
use control_plane::{api, services, AppState, Database};
use services::certificate::ControlPlaneKeypair;
use std::sync::Arc;
use tokio::signal;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

/// Mesh Control Plane - Network orchestration and coordination
#[derive(Parser, Debug)]
#[command(name = "control-plane")]
#[command(about = "Mesh Control Plane for network coordination")]
#[command(version)]
struct Cli {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Override log level (trace, debug, info, warn, error)
    #[arg(short, long)]
    log_level: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize tracing
    let log_level = match cli.log_level.as_deref() {
        Some("trace") => Level::TRACE,
        Some("debug") => Level::DEBUG,
        Some("warn") => Level::WARN,
        Some("error") => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Starting Meshnet Control Plane");

    // Initialize database
    let db_path = Database::default_path()?;
    info!(path = %db_path.display(), "Using database");

    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| format!("Invalid database path (contains invalid UTF-8): {}", db_path.display()))?;
    let db = Database::new(db_path_str)?;

    // Run migrations
    info!("Running database migrations");
    db.migrate()?;

    // Load or generate control plane keypair
    info!("Loading control plane keypair");
    let keypair = Arc::new(ControlPlaneKeypair::load_or_generate()?);

    // Create application state
    let state = AppState::new(db.clone(), keypair);

    // Create API router
    let app = api::create_router(state);

    // Spawn presence monitor task
    info!("Starting presence monitor");
    tokio::spawn(async move {
        services::presence_monitor(db).await;
    });

    // Try binding to requested port, with fallback to find available port
    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => {
            info!(address = %addr, "Control plane listening");
            listener
        }
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            // Port in use, try to find an available port
            info!(port = cli.port, "Port already in use, searching for available port...");

            let mut found = None;
            for try_port in (cli.port + 1)..(cli.port + 100) {
                let try_addr = format!("0.0.0.0:{}", try_port);
                if let Ok(listener) = tokio::net::TcpListener::bind(&try_addr).await {
                    info!(address = %try_addr, "Control plane listening on alternative port");
                    println!("\n⚠️  Port {} was in use. Using port {} instead.", cli.port, try_port);
                    println!("Update agent commands to use: --control-plane http://localhost:{}\n", try_port);
                    found = Some(listener);
                    break;
                }
            }

            found.ok_or_else(|| {
                format!("Could not find available port in range {}-{}. Please stop conflicting services.",
                    cli.port, cli.port + 100)
            })?
        }
        Err(e) => return Err(e.into()),
    };

    // Start server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Control plane shut down");

    Ok(())
}

/// Wait for shutdown signal (Ctrl+C)
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = signal::ctrl_c().await {
            tracing::error!("Failed to install Ctrl+C handler: {}", e);
            // Sleep forever since we can't listen for signals
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(e) => {
                tracing::error!("Failed to install SIGTERM handler: {}", e);
                // Sleep forever since we can't listen for signals
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("Received terminate signal");
        },
    }
}
