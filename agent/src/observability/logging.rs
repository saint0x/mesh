use std::path::PathBuf;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize production logging with file rotation
///
/// This sets up dual output:
/// - Daily rotating log files in ~/.meshnet/logs/agent.log
/// - Stdout with pretty formatting
/// - Configurable via RUST_LOG environment variable
pub fn init_production_logging(level: &str, log_dir: Option<PathBuf>) -> anyhow::Result<()> {
    let log_dir = log_dir.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".meshnet")
            .join("logs")
    });

    // Create log directory
    std::fs::create_dir_all(&log_dir)?;

    // Environment filter (RUST_LOG overrides level)
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    // File appender with daily rotation
    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "agent.log");

    // Build subscriber with both file and stdout
    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_writer(file_appender)
                .with_ansi(false) // No colors in log files
                .with_target(true)
                .with_line_number(true),
        )
        .with(
            fmt::layer()
                .with_writer(std::io::stdout)
                .with_target(false)
                .with_line_number(false),
        )
        .try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;

    tracing::info!(
        log_dir = %log_dir.display(),
        level = %level,
        "Production logging initialized"
    );

    Ok(())
}

/// Initialize simple logging for CLI commands (stdout only)
pub fn init_simple_logging(level: &str) -> anyhow::Result<()> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_writer(std::io::stdout)
                .with_target(false)
                .with_line_number(false),
        )
        .try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;

    Ok(())
}
