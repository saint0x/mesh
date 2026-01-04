# Contributing to Mesh

Thank you for your interest in contributing to Mesh! This document provides guidelines and instructions for development.

## Development Setup

### Prerequisites

- **Rust 1.88+** (latest stable recommended)
- **PostgreSQL** (via Docker, or native install)
- **Docker** and Docker Compose (for database)
- **Git**

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/meshai/mesh.git
   cd mesh
   ```

2. **Install Rust dependencies:**
   ```bash
   cargo build --workspace
   ```

3. **Install SQLx CLI:**
   ```bash
   cargo install sqlx-cli --no-default-features --features postgres
   ```

4. **Start the database:**
   ```bash
   cd control-plane
   docker-compose up -d
   ```

5. **Run migrations:**
   ```bash
   cd control-plane
   cp .env.example .env
   sqlx migrate run
   ```

6. **Run tests:**
   ```bash
   cargo test --workspace
   ```

## Project Structure

```
mesh/
├── agent/              # Desktop/mobile agent (library)
│   ├── src/
│   │   ├── device/     # Device configuration and identity
│   │   ├── network/    # Network layer (libp2p)
│   │   ├── executor/   # Job execution
│   │   └── errors.rs   # Error types
│   └── Cargo.toml
├── relay-server/       # Relay gateway (binary)
│   ├── src/
│   │   └── main.rs
│   └── Cargo.toml
├── control-plane/      # Control plane API (binary)
│   ├── src/
│   │   ├── db/         # Database models and queries
│   │   ├── routes/     # HTTP routes
│   │   └── main.rs
│   ├── migrations/     # SQLx migrations
│   ├── docker-compose.yml
│   └── Cargo.toml
└── reference/          # Reference VPN implementation (study material)
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow our [Coding Standards](#coding-standards) below.

### 3. Run Tests and Checks

Before committing, ensure all checks pass:

```bash
# Format code
cargo fmt --all

# Run lints
cargo clippy --workspace --all-targets -- -D warnings

# Run tests
cargo test --workspace

# Build release (optional)
cargo build --workspace --release
```

### 4. Commit Changes

Use conventional commit messages:

```bash
git commit -m "feat(agent): add device capabilities detection"
git commit -m "fix(relay): handle connection timeout gracefully"
git commit -m "docs: update CONTRIBUTING.md with testing guidelines"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Coding Standards

### Error Handling

**✅ DO:**
- Use `Result<T>` for all fallible operations
- Define domain-specific error types with `thiserror`
- Add context when propagating errors
- Log errors with structured fields

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),
}

pub fn load_config(path: &Path) -> Result<DeviceConfig, AgentError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| {
            tracing::error!(path = %path.display(), error = %e, "Failed to read config");
            AgentError::Io(e)
        })?;

    Ok(toml::from_str(&content)?)
}
```

**❌ DON'T:**
- Use `unwrap()` or `expect()` in production code
- Swallow errors silently
- Use generic error messages

### Logging

Use `tracing` for structured logging:

```rust
use tracing::{info, warn, error, debug, instrument};

#[instrument(skip(executor))]
async fn execute_job(job: &JobEnvelope, executor: &Executor) -> Result<JobResult> {
    info!(
        job_id = %job.job_id,
        workload = %job.workload_id,
        "Starting job execution"
    );

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

### Testing

**Unit Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_save_load_roundtrip() {
        let config = DeviceConfig::generate(
            "test-device".to_string(),
            "test-network".to_string(),
            "http://localhost:8080".to_string(),
        );

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("device.toml");

        config.save(&path).expect("save should succeed");
        let loaded = DeviceConfig::load(&path).expect("load should succeed");

        // Verify keypair bytes are identical
        assert_eq!(config.keypair.to_bytes(), loaded.keypair.to_bytes());
        assert_eq!(config.device_id, loaded.device_id);
    }
}
```

**Integration Tests:**
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_database_connection() {
        let database_url = std::env::var("DATABASE_URL")
            .expect("DATABASE_URL must be set for integration tests");

        let pool = sqlx::postgres::PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to database");

        // Test query
        let row: (i64,) = sqlx::query_as("SELECT 1")
            .fetch_one(&pool)
            .await
            .expect("Query failed");

        assert_eq!(row.0, 1);
    }
}
```

### Code Style

- **Format:** Run `cargo fmt` before committing
- **Lints:** Fix all `cargo clippy` warnings
- **Naming:**
  - Types: `PascalCase` (e.g., `DeviceConfig`)
  - Functions/variables: `snake_case` (e.g., `load_config`)
  - Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Comments:** Use `///` for public API docs, `//` for inline comments
- **Line length:** Prefer <100 characters, max 120

### Documentation

Public APIs must have documentation:

```rust
/// Device configuration containing identity and network settings.
///
/// This struct is serialized to TOML and saved at `~/.meshnet/device.toml`.
/// The Ed25519 keypair is encoded using multibase Base58BTC format.
///
/// # Examples
///
/// ```
/// use mesh_agent::DeviceConfig;
///
/// let config = DeviceConfig::generate(
///     "my-device".to_string(),
///     "my-network".to_string(),
///     "https://control.example.com".to_string(),
/// );
///
/// config.save(&config.default_path()?)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    // ...
}
```

## Database Migrations

### Creating Migrations

```bash
cd control-plane
sqlx migrate add my_migration_name
```

Edit the generated SQL file in `control-plane/migrations/`.

### Running Migrations

```bash
cd control-plane
sqlx migrate run
```

### Reverting Migrations

```bash
sqlx migrate revert
```

### SQLx Compile-Time Verification

We use `sqlx::query!` for compile-time query verification:

```rust
let device = sqlx::query_as!(
    Device,
    r#"
    SELECT device_id, network_id, name, public_key, capabilities,
           certificate, created_at, last_seen, status
    FROM devices
    WHERE device_id = $1
    "#,
    device_id
)
.fetch_one(&pool)
.await?;
```

To prepare query metadata (for CI without database):

```bash
cd control-plane
cargo sqlx prepare
```

## Testing

### Running Tests

```bash
# All tests
cargo test --workspace

# Specific package
cargo test -p agent
cargo test -p control-plane

# Specific test
cargo test test_config_save_load_roundtrip

# With output
cargo test -- --nocapture
```

### Integration Tests

Integration tests require the database to be running:

```bash
cd control-plane
docker-compose up -d
cargo test -p control-plane
```

## Continuous Integration

Our CI pipeline runs on every push and PR:

1. **Build** on Linux, macOS, Windows
2. **Tests** on all platforms
3. **Clippy** lints (with `-D warnings`)
4. **Format** check (`cargo fmt --check`)

Ensure all checks pass locally before pushing.

## Getting Help

- **Documentation:** See [README.md](README.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Issues:** Check [GitHub Issues](https://github.com/meshai/mesh/issues)
- **Discussions:** Use [GitHub Discussions](https://github.com/meshai/mesh/discussions)

## License

By contributing to Mesh, you agree that your contributions will be licensed under the MIT License.
