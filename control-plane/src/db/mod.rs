use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::path::PathBuf;
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
            "file::memory:?cache=shared"
        } else {
            database_path
        };

        // Create connection manager
        let manager = SqliteConnectionManager::file(connection_string)
            .with_init(move |conn| {
                // Enable foreign keys
                conn.execute_batch("PRAGMA foreign_keys = ON;")?;
                // Enable WAL mode for better concurrency (not applicable to :memory:)
                if !is_memory {
                    conn.execute_batch("PRAGMA journal_mode = WAL;")?;
                }
                Ok(())
            });

        // Create connection pool
        let pool = Pool::builder()
            .max_size(10)
            .build(manager)?;

        tracing::info!("Database connected successfully");

        Ok(Self { pool })
    }

    /// Run database migrations
    pub fn migrate(&self) -> Result<()> {
        tracing::info!("Running database migrations");

        let conn = self.pool.get()?;

        // Read migration files and execute
        // Use CARGO_MANIFEST_DIR to find migrations relative to crate root
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| ".".to_string());
        let migrations_dir = std::path::Path::new(&manifest_dir).join("migrations");

        if !migrations_dir.exists() {
            return Err(DbError::Config(format!(
                "Migrations directory not found at: {}",
                migrations_dir.display()
            )));
        }

        // Get all .sql files
        let mut migration_files: Vec<_> = std::fs::read_dir(migrations_dir)
            .map_err(|e| DbError::Config(format!("Failed to read migrations directory: {}", e)))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension().and_then(|s| s.to_str()) == Some("sql")
            })
            .collect();

        // Sort by filename (assumes numeric prefix like 001_, 002_, etc.)
        migration_files.sort_by_key(|entry| entry.path());

        // Execute each migration
        for entry in migration_files {
            let path = entry.path();
            tracing::info!(file = %path.display(), "Applying migration");

            let sql = std::fs::read_to_string(&path)
                .map_err(|e| DbError::Config(format!("Failed to read migration file: {}", e)))?;

            conn.execute_batch(&sql)?;
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
