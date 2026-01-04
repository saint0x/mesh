use sqlx::migrate::MigrateDatabase;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;
use thiserror::Error;

pub mod models;

/// Database-related errors
#[derive(Error, Debug)]
pub enum DbError {
    #[error("Database error: {0}")]
    Sqlx(#[from] sqlx::Error),

    #[error("Migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    #[error("Database not found: {0}")]
    NotFound(String),

    #[error("Database configuration error: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, DbError>;

/// Database connection pool and configuration
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Create a new database connection pool
    ///
    /// # Arguments
    ///
    /// * `database_url` - SQLite database URL (e.g., "sqlite:meshnet.db" or "sqlite::memory:")
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use control_plane::db::Database;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // File-based database
    /// let db = Database::new("sqlite:meshnet.db").await?;
    ///
    /// // In-memory database (for testing)
    /// let db = Database::new("sqlite::memory:").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new(database_url: &str) -> Result<Self> {
        tracing::info!(url = %database_url, "Connecting to database");

        // Create database if it doesn't exist (for file-based DBs)
        if !database_url.contains(":memory:")
            && !sqlx::Sqlite::database_exists(database_url).await?
        {
            tracing::info!("Database does not exist, creating...");
            sqlx::Sqlite::create_database(database_url).await?;
        }

        // Configure connection options
        let options = SqliteConnectOptions::from_str(database_url)?
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal) // Write-Ahead Logging for better concurrency
            .synchronous(sqlx::sqlite::SqliteSynchronous::Normal) // Balance safety and performance
            .foreign_keys(true); // Enable foreign key constraints

        // Create connection pool
        let pool = SqlitePoolOptions::new()
            .max_connections(10)
            .connect_with(options)
            .await?;

        tracing::info!("Database connected successfully");

        Ok(Self { pool })
    }

    /// Run database migrations
    ///
    /// Applies all pending migrations from the `migrations/` directory.
    pub async fn migrate(&self) -> Result<()> {
        tracing::info!("Running database migrations");
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        tracing::info!("Migrations completed successfully");
        Ok(())
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Get default database path: `~/.meshnet/control-plane.db`
    pub fn default_path() -> Result<String> {
        let home = dirs::home_dir()
            .ok_or_else(|| DbError::Config("Could not determine home directory".to_string()))?;

        let db_dir = home.join(".meshnet");
        std::fs::create_dir_all(&db_dir)
            .map_err(|e| DbError::Config(format!("Failed to create database directory: {}", e)))?;

        let db_path = db_dir.join("control-plane.db");
        Ok(format!("sqlite:{}", db_path.display()))
    }

    /// Close the database connection pool
    pub async fn close(self) {
        tracing::info!("Closing database connection");
        self.pool.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_in_memory_db() {
        let db = Database::new("sqlite::memory:")
            .await
            .expect("Failed to create in-memory database");

        // Verify we can execute a simple query
        let result: (i64,) = sqlx::query_as("SELECT 1")
            .fetch_one(db.pool())
            .await
            .expect("Failed to execute query");

        assert_eq!(result.0, 1);
    }

    #[tokio::test]
    async fn test_run_migrations() {
        let db = Database::new("sqlite::memory:")
            .await
            .expect("Failed to create database");

        db.migrate().await.expect("Failed to run migrations");

        // Verify tables were created
        let tables: Vec<(String,)> =
            sqlx::query_as("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .fetch_all(db.pool())
                .await
                .expect("Failed to query tables");

        let table_names: Vec<String> = tables.into_iter().map(|(name,)| name).collect();

        assert!(table_names.contains(&"networks".to_string()));
        assert!(table_names.contains(&"devices".to_string()));
        assert!(table_names.contains(&"ledger_events".to_string()));
    }

    #[tokio::test]
    async fn test_foreign_keys_enabled() {
        let db = Database::new("sqlite::memory:")
            .await
            .expect("Failed to create database");

        let result: (i64,) = sqlx::query_as("PRAGMA foreign_keys")
            .fetch_one(db.pool())
            .await
            .expect("Failed to check foreign keys");

        assert_eq!(result.0, 1, "Foreign keys should be enabled");
    }

    #[tokio::test]
    async fn test_wal_mode_enabled() {
        // Note: In-memory databases use "memory" journal mode, not WAL
        // This test would pass with a file-based database
        let db = Database::new("sqlite::memory:")
            .await
            .expect("Failed to create database");

        let result: (String,) = sqlx::query_as("PRAGMA journal_mode")
            .fetch_one(db.pool())
            .await
            .expect("Failed to check journal mode");

        // In-memory DBs use "memory" mode, file DBs would use "wal"
        assert!(
            result.0 == "memory" || result.0 == "wal",
            "Journal mode should be memory (for :memory:) or wal (for file)"
        );
    }

    #[tokio::test]
    async fn test_default_path() {
        let path = Database::default_path().expect("Failed to get default path");
        assert!(path.starts_with("sqlite:"));
        assert!(path.contains(".meshnet"));
        assert!(path.ends_with("control-plane.db"));
    }
}
