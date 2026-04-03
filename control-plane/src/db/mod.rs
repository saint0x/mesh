use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::OptionalExtension;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub mod models;

/// Database-related errors
#[derive(Error, Debug)]
pub enum DbError {
    #[error("Database error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("Connection pool error: {0}")]
    Pool(#[from] r2d2::Error),

    #[error("Database not found: {0}")]
    NotFound(String),

    #[error("Database configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, DbError>;

/// Database connection pool
#[derive(Clone)]
pub struct Database {
    pool: Pool<SqliteConnectionManager>,
}

impl Database {
    /// Create a new database connection pool
    pub fn new(database_path: &str) -> Result<Self> {
        tracing::info!(path = %database_path, "Connecting to database");

        // Create directory if needed (skip for :memory: databases)
        if database_path != ":memory:" {
            if let Some(parent) = std::path::Path::new(database_path).parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    DbError::Config(format!("Failed to create database directory: {}", e))
                })?;
            }
        }

        // For in-memory databases, use shared cache to ensure all connections see the same data
        // If the path contains "?mode=memory", it's a unique in-memory DB, use as-is
        let is_memory = database_path == ":memory:" || database_path.contains("?mode=memory");
        let connection_string = if database_path == ":memory:" {
            format!(
                "file:meshnet-{}?mode=memory&cache=shared",
                uuid::Uuid::new_v4()
            )
        } else {
            database_path.to_string()
        };

        // Create connection manager
        let manager = SqliteConnectionManager::file(connection_string).with_init(move |conn| {
            // Enable foreign keys
            conn.execute_batch("PRAGMA foreign_keys = ON;")?;
            // Enable WAL mode for better concurrency (not applicable to :memory:)
            if !is_memory {
                conn.execute_batch("PRAGMA journal_mode = WAL;")?;
            }
            Ok(())
        });

        // Create connection pool
        let pool = Pool::builder().max_size(10).build(manager)?;

        tracing::info!("Database connected successfully");

        Ok(Self { pool })
    }

    /// Run database migrations
    pub fn migrate(&self) -> Result<()> {
        tracing::info!("Running database migrations");

        let conn = self.pool.get()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )?;

        // Resolve migrations relative to the installed binary or crate root so startup
        // does not depend on the caller's current working directory.
        let migrations_dir = locate_migrations_dir()?;

        // Get all .sql files
        let mut migration_files: Vec<_> = std::fs::read_dir(migrations_dir)
            .map_err(|e| DbError::Config(format!("Failed to read migrations directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().and_then(|s| s.to_str()) == Some("sql"))
            .collect();

        // Sort by filename (assumes numeric prefix like 001_, 002_, etc.)
        migration_files.sort_by_key(|entry| entry.path());

        // Execute each migration
        for entry in migration_files {
            let path = entry.path();
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    DbError::Config(format!("Invalid migration filename: {}", path.display()))
                })?
                .to_string();

            let already_applied: Option<String> = conn
                .query_row(
                    "SELECT filename FROM schema_migrations WHERE filename = ?1",
                    [&filename],
                    |row| row.get(0),
                )
                .optional()?;
            if already_applied.is_some() {
                tracing::debug!(file = %filename, "Skipping previously applied migration");
                continue;
            }

            if migration_is_already_effective(&conn, &filename)? {
                tracing::info!(
                    file = %filename,
                    "Backfilling previously applied migration from existing schema"
                );
                conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (?1)",
                    [&filename],
                )?;
                continue;
            }

            tracing::info!(file = %path.display(), "Applying migration");

            let sql = std::fs::read_to_string(&path)
                .map_err(|e| DbError::Config(format!("Failed to read migration file: {}", e)))?;

            let tx = conn.unchecked_transaction()?;
            tx.execute_batch(&sql)?;
            tx.execute(
                "INSERT INTO schema_migrations (filename) VALUES (?1)",
                [&filename],
            )?;
            tx.commit()?;
        }

        tracing::info!("Migrations completed successfully");
        Ok(())
    }

    /// Get a connection from the pool
    pub fn get_conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        Ok(self.pool.get()?)
    }

    /// Get default database path: `~/.meshnet/control-plane.db`
    pub fn default_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| DbError::Config("Could not determine home directory".to_string()))?;

        let db_dir = home.join(".meshnet");
        std::fs::create_dir_all(&db_dir)
            .map_err(|e| DbError::Config(format!("Failed to create database directory: {}", e)))?;

        Ok(db_dir.join("control-plane.db"))
    }
}

fn locate_migrations_dir() -> Result<PathBuf> {
    if let Ok(explicit_dir) = std::env::var("MESHNET_MIGRATIONS_DIR") {
        let path = PathBuf::from(explicit_dir);
        if path.exists() {
            return Ok(path);
        }
    }

    let mut candidates = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        for ancestor in exe_path.ancestors() {
            candidates.push(ancestor.join("migrations"));
            candidates.push(ancestor.join("control-plane").join("migrations"));
        }
    }

    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations"));

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(DbError::Config(
        "Migrations directory not found near the executable or crate root".to_string(),
    ))
}

fn migration_is_already_effective(conn: &rusqlite::Connection, filename: &str) -> Result<bool> {
    Ok(match filename {
        "001_create_networks.sql" => table_exists(conn, "networks")?,
        "002_create_devices.sql" => table_exists(conn, "devices")?,
        "003_create_ledger_events.sql" => table_exists(conn, "ledger_events")?,
        "004_add_ring_topology.sql" => {
            table_exists(conn, "resource_locks")?
                && table_exists(conn, "pools")?
                && column_exists(conn, "devices", "ring_position")?
                && column_exists(conn, "devices", "left_neighbor_id")?
                && column_exists(conn, "devices", "right_neighbor_id")?
                && column_exists(conn, "devices", "shard_column_start")?
                && column_exists(conn, "devices", "shard_column_end")?
                && column_exists(conn, "devices", "contributed_memory")?
                && column_exists(conn, "devices", "lock_status")?
                && column_exists(conn, "devices", "lock_timestamp")?
                && column_exists(conn, "devices", "unlock_requested_at")?
        }
        "005_create_inference_dispatch.sql" => {
            table_exists(conn, "inference_jobs")?
                && table_exists(conn, "inference_job_assignments")?
        }
        "006_add_device_connectivity_state.sql" => {
            column_exists(conn, "devices", "connectivity_state")?
        }
        "007_add_device_peer_metadata.sql" => {
            column_exists(conn, "devices", "peer_id")?
                && column_exists(conn, "devices", "listen_addrs")?
        }
        "008_add_device_direct_candidates.sql" => {
            column_exists(conn, "devices", "direct_candidates")?
        }
        "009_add_inference_assignment_metrics.sql" => {
            column_exists(conn, "inference_job_assignments", "execution_time_ms")?
        }
        "010_add_ring_model_id.sql" => column_exists(conn, "devices", "shard_model_id")?,
        _ => false,
    })
}

fn table_exists(conn: &rusqlite::Connection, table_name: &str) -> Result<bool> {
    let table = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [table_name],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(table.is_some())
}

fn column_exists(conn: &rusqlite::Connection, table_name: &str, column_name: &str) -> Result<bool> {
    let pragma = format!("PRAGMA table_info({})", table_name);
    let mut stmt = conn.prepare(&pragma)?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let existing: String = row.get(1)?;
        if existing == column_name {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
static TEST_DB_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique in-memory database for testing
/// Each call returns a new isolated database
#[cfg(test)]
pub(crate) fn create_test_db() -> Database {
    let id = TEST_DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let db_name = format!("file:testdb{}?mode=memory&cache=shared", id);
    let db = Database::new(&db_name).expect("Failed to create test database");
    db.migrate().expect("Failed to run migrations");
    db
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_in_memory_db() {
        let db = Database::new(":memory:").expect("Failed to create in-memory database");

        // Verify we can execute a simple query
        let conn = db.get_conn().expect("Failed to get connection");
        let result: i64 = conn
            .query_row("SELECT 1", [], |row| row.get(0))
            .expect("Failed to execute query");

        assert_eq!(result, 1);
    }

    #[test]
    fn test_run_migrations() {
        let db = Database::new(":memory:").expect("Failed to create database");
        db.migrate().expect("Failed to run migrations");

        // Verify tables were created
        let conn = db.get_conn().expect("Failed to get connection");
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .expect("Failed to prepare statement");

        let tables: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .expect("Failed to query tables")
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("Failed to collect results");

        assert!(tables.contains(&"networks".to_string()));
        assert!(tables.contains(&"devices".to_string()));
        assert!(tables.contains(&"ledger_events".to_string()));
        assert!(tables.contains(&"schema_migrations".to_string()));
    }

    #[test]
    fn test_run_migrations_is_idempotent() {
        let temp_db = NamedTempFile::new().expect("Failed to create temp database file");
        let db = Database::new(
            temp_db
                .path()
                .to_str()
                .expect("Temp database path should be valid UTF-8"),
        )
        .expect("Failed to create database");
        db.migrate().expect("Failed to run first migration pass");
        db.migrate().expect("Failed to run second migration pass");

        let conn = db.get_conn().expect("Failed to get connection");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM schema_migrations", [], |row| {
                row.get(0)
            })
            .expect("Failed to count applied migrations");
        assert!(count >= 1);
    }

    #[test]
    fn test_run_migrations_backfills_legacy_schema_history() {
        let temp_db = NamedTempFile::new().expect("Failed to create temp database file");
        let db = Database::new(
            temp_db
                .path()
                .to_str()
                .expect("Temp database path should be valid UTF-8"),
        )
        .expect("Failed to create database");

        let conn = db.get_conn().expect("Failed to get connection");
        conn.execute_batch(
            r#"
            CREATE TABLE networks (
                network_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_user_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                settings TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE devices (
                device_id TEXT PRIMARY KEY,
                network_id TEXT NOT NULL,
                name TEXT NOT NULL,
                public_key BLOB NOT NULL,
                capabilities TEXT NOT NULL,
                certificate BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen TEXT,
                status TEXT NOT NULL DEFAULT 'offline',
                ring_position INTEGER,
                left_neighbor_id TEXT,
                right_neighbor_id TEXT,
                shard_column_start INTEGER,
                shard_column_end INTEGER,
                contributed_memory INTEGER,
                lock_status TEXT DEFAULT 'unlocked',
                lock_timestamp TEXT,
                unlock_requested_at TEXT
            );
            CREATE TABLE ledger_events (
                event_id TEXT PRIMARY KEY,
                network_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                job_id TEXT,
                device_id TEXT,
                user_id TEXT,
                credits_amount REAL,
                metadata TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE resource_locks (
                device_id TEXT PRIMARY KEY,
                memory_bytes INTEGER NOT NULL,
                lock_timestamp TEXT NOT NULL,
                cooldown_hours INTEGER NOT NULL DEFAULT 24,
                unlock_requested_at TEXT,
                status TEXT NOT NULL DEFAULT 'locked'
            );
            CREATE TABLE pools (
                pool_id TEXT PRIMARY KEY,
                network_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                ring_stable BOOLEAN DEFAULT FALSE,
                total_workers INTEGER,
                active_workers INTEGER,
                status TEXT DEFAULT 'initializing',
                last_checkpoint_token INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            "#,
        )
        .expect("Failed to seed legacy schema");
        drop(conn);

        db.migrate()
            .expect("Failed to backfill migrations on legacy schema");

        let conn = db.get_conn().expect("Failed to get connection");
        let recorded: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM schema_migrations WHERE filename IN (?1, ?2, ?3, ?4)",
                [
                    "001_create_networks.sql",
                    "002_create_devices.sql",
                    "003_create_ledger_events.sql",
                    "004_add_ring_topology.sql",
                ],
                |row| row.get(0),
            )
            .expect("Failed to count backfilled migrations");
        assert_eq!(recorded, 4);
    }

    #[test]
    fn test_locate_migrations_dir_finds_real_directory() {
        let path = locate_migrations_dir().expect("Should find migrations directory");
        assert!(path.ends_with("migrations"));
        assert!(path.join("001_create_networks.sql").exists());
    }

    #[test]
    fn test_foreign_keys_enabled() {
        let db = Database::new(":memory:").expect("Failed to create database");
        let conn = db.get_conn().expect("Failed to get connection");

        let result: i64 = conn
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .expect("Failed to check foreign keys");

        assert_eq!(result, 1, "Foreign keys should be enabled");
    }

    #[test]
    fn test_wal_mode_enabled() {
        let db = Database::new(":memory:").expect("Failed to create database");
        let conn = db.get_conn().expect("Failed to get connection");

        let result: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .expect("Failed to check journal mode");

        // In-memory DBs use "memory" mode, file DBs would use "wal"
        assert!(
            result == "memory" || result == "wal",
            "Journal mode should be memory (for :memory:) or wal (for file)"
        );
    }

    #[test]
    fn test_default_path() {
        let path = Database::default_path().expect("Failed to get default path");
        assert!(path.to_string_lossy().contains(".meshnet"));
        assert!(path.to_string_lossy().ends_with("control-plane.db"));
    }
}
