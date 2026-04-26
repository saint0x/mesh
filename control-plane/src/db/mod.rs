use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::OptionalExtension;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub mod models;

use crate::api::types::{
    DecodeQueueEntryStatus, ExecutionPhase, InferenceSessionCheckpointStatus,
    JobSchedulerStatusResponse, KvReplicaResidencyStatus, KvResidencySummary, KvTransferPolicy,
    NetworkSchedulerStatusResponse, RegroupEventStatus, SchedulerJobSummary,
    ServingGroupLeaseStatus, ServingGroupMemberStatus, ServingGroupStatus,
};

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

    #[error("Database data error: {0}")]
    Data(String),
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
        "011_add_assignment_capacity_and_shards.sql" => {
            column_exists(conn, "inference_job_assignments", "shard_column_start")?
                && column_exists(conn, "inference_job_assignments", "shard_column_end")?
                && column_exists(conn, "inference_job_assignments", "assigned_capacity_units")?
                && column_exists(conn, "inference_job_assignments", "execution_provider")?
        }
        "012_add_credit_reservation_and_allowance.sql" => {
            column_exists(conn, "inference_jobs", "reserved_credits")?
                && column_exists(conn, "inference_jobs", "settled_credits")?
                && column_exists(conn, "inference_jobs", "released_credits")?
                && column_exists(conn, "inference_jobs", "available_completion_tokens")?
                && column_exists(conn, "inference_jobs", "model_size_factor")?
        }
        "013_add_realtime_accounting_progress.sql" => {
            column_exists(conn, "inference_jobs", "accounted_completion_tokens")?
                && column_exists(conn, "inference_jobs", "prompt_credits_accounted")?
                && column_exists(
                    conn,
                    "inference_job_assignments",
                    "reported_completion_tokens",
                )?
        }
        "014_add_execution_plan_to_inference_jobs.sql" => {
            column_exists(conn, "inference_jobs", "execution_plan_json")?
        }
        "015_add_phase_timing_to_inference_jobs.sql" => {
            column_exists(conn, "inference_jobs", "time_to_first_token_ms")?
                && column_exists(conn, "inference_jobs", "prefill_completed_at")?
        }
        "016_add_active_segment_to_inference_jobs.sql" => {
            column_exists(conn, "inference_jobs", "active_segment_id")?
        }
        "017_add_assignment_segment_lifecycle.sql" => {
            column_exists(conn, "inference_job_assignments", "active_segment_id")?
                && column_exists(
                    conn,
                    "inference_job_assignments",
                    "last_completed_segment_id",
                )?
                && column_exists(conn, "inference_job_assignments", "segment_completed_at")?
        }
        "018_create_inference_sessions.sql" => table_exists(conn, "inference_sessions")?,
        "019_create_inference_session_replicas.sql" => {
            table_exists(conn, "inference_session_replicas")?
        }
        "020_create_inference_session_checkpoints.sql" => {
            table_exists(conn, "inference_session_checkpoints")?
        }
        "021_create_inference_serving_groups.sql" => {
            table_exists(conn, "inference_serving_groups")?
        }
        "022_create_inference_decode_queue.sql" => table_exists(conn, "inference_decode_queue")?,
        "023_add_scheduler_visibility_metadata.sql" => {
            column_exists(conn, "inference_decode_queue", "blocked_reason")?
                && column_exists(conn, "inference_decode_queue", "blocked_since")?
                && column_exists(conn, "inference_decode_queue", "block_detail")?
                && column_exists(conn, "inference_serving_groups", "lease_owner_device_id")?
                && column_exists(conn, "inference_serving_groups", "lease_expires_at")?
        }
        "024_create_inference_regroup_events.sql" => {
            table_exists(conn, "inference_regroup_events")?
        }
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

#[derive(Clone)]
struct ServingGroupRow {
    group_id: String,
    session_id: String,
    job_id: String,
    network_id: String,
    model_id: String,
    phase: ExecutionPhase,
    lease_owner_device_id: Option<String>,
    lease_expires_at: Option<String>,
    member: ServingGroupMemberStatus,
}

#[derive(Clone)]
struct KvSessionRow {
    session_id: String,
    job_id: String,
    network_id: String,
    model_id: String,
    status: String,
    active_segment_id: Option<String>,
    kv_owner_device_id: String,
    kv_transfer_policy: KvTransferPolicy,
    kv_sequence_position: Option<u32>,
    kv_checkpoint_device_id: Option<String>,
    kv_checkpoint_created_at: Option<String>,
    updated_at: String,
    latest_checkpoint: Option<InferenceSessionCheckpointStatus>,
}

impl Database {
    pub fn load_network_scheduler_status(
        &self,
        network_id: &str,
    ) -> Result<NetworkSchedulerStatusResponse> {
        let conn = self.get_conn()?;
        require_network_exists(&conn, network_id)?;

        Ok(NetworkSchedulerStatusResponse {
            success: true,
            network_id: network_id.to_string(),
            jobs: load_scheduler_job_summaries(&conn, network_id, None)?,
            decode_queue: load_decode_queue_entries(&conn, network_id, None)?,
            serving_groups: load_serving_group_snapshots(&conn, network_id, None)?,
            kv_residency: load_kv_residency_summaries(&conn, network_id, None)?,
            regroup_events: load_regroup_events(&conn, network_id, None)?,
        })
    }

    pub fn load_job_scheduler_status(&self, job_id: &str) -> Result<JobSchedulerStatusResponse> {
        let conn = self.get_conn()?;
        let (network_id, model_id, status, active_segment_id, completion_tokens, updated_at) =
            conn.query_row(
                r#"
                SELECT network_id, model_id, status, active_segment_id, completion_tokens, updated_at
                FROM inference_jobs
                WHERE job_id = ?
                "#,
                [job_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, i64>(4)? as u32,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| DbError::NotFound(format!("Inference job not found: {}", job_id)))?;

        Ok(JobSchedulerStatusResponse {
            success: true,
            job_id: job_id.to_string(),
            network_id: network_id.clone(),
            model_id,
            status,
            active_segment_id,
            completion_tokens,
            updated_at,
            decode_queue: load_decode_queue_entries(&conn, &network_id, Some(job_id))?,
            serving_groups: load_serving_group_snapshots(&conn, &network_id, Some(job_id))?,
            kv_residency: load_kv_residency_summaries(&conn, &network_id, Some(job_id))?,
            regroup_events: load_regroup_events(&conn, &network_id, Some(job_id))?,
        })
    }
}

fn require_network_exists(conn: &rusqlite::Connection, network_id: &str) -> Result<()> {
    let exists = conn
        .query_row(
            "SELECT network_id FROM networks WHERE network_id = ?",
            [network_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if exists.is_some() {
        Ok(())
    } else {
        Err(DbError::NotFound(format!(
            "Network not found: {}",
            network_id
        )))
    }
}

fn load_scheduler_job_summaries(
    conn: &rusqlite::Connection,
    network_id: &str,
    job_id: Option<&str>,
) -> Result<Vec<SchedulerJobSummary>> {
    if let Some(job_id) = job_id {
        let mut stmt = conn.prepare(
            r#"
            SELECT job_id, model_id, status, active_segment_id, completion_tokens, updated_at
            FROM inference_jobs
            WHERE network_id = ?1 AND job_id = ?2
            ORDER BY updated_at DESC, job_id DESC
            "#,
        )?;
        let rows = stmt.query_map([network_id, job_id], |row| {
            Ok(SchedulerJobSummary {
                job_id: row.get(0)?,
                model_id: row.get(1)?,
                status: row.get(2)?,
                active_segment_id: row.get(3)?,
                completion_tokens: row.get::<_, i64>(4)? as u32,
                updated_at: row.get(5)?,
            })
        })?;
        collect_rows(rows)
    } else {
        let mut stmt = conn.prepare(
            r#"
            SELECT job_id, model_id, status, active_segment_id, completion_tokens, updated_at
            FROM inference_jobs
            WHERE network_id = ?1
            ORDER BY updated_at DESC, job_id DESC
            "#,
        )?;
        let rows = stmt.query_map([network_id], |row| {
            Ok(SchedulerJobSummary {
                job_id: row.get(0)?,
                model_id: row.get(1)?,
                status: row.get(2)?,
                active_segment_id: row.get(3)?,
                completion_tokens: row.get::<_, i64>(4)? as u32,
                updated_at: row.get(5)?,
            })
        })?;
        collect_rows(rows)
    }
}

fn load_decode_queue_entries(
    conn: &rusqlite::Connection,
    network_id: &str,
    job_id: Option<&str>,
) -> Result<Vec<DecodeQueueEntryStatus>> {
    if let Some(job_id) = job_id {
        let mut stmt = conn.prepare(
            r#"
            SELECT session_id, job_id, network_id, segment_id, group_id, status,
                   blocked_reason, blocked_since, block_detail, ready_at,
                   lease_owner_device_id, lease_expires_at, last_error, updated_at
            FROM inference_decode_queue
            WHERE network_id = ?1 AND job_id = ?2
            ORDER BY updated_at ASC, session_id ASC
            "#,
        )?;
        let rows = stmt.query_map([network_id, job_id], |row| {
            Ok(DecodeQueueEntryStatus {
                session_id: row.get(0)?,
                job_id: row.get(1)?,
                network_id: row.get(2)?,
                segment_id: row.get(3)?,
                group_id: row.get(4)?,
                status: row.get(5)?,
                blocked_reason: row.get(6)?,
                blocked_since: row.get(7)?,
                block_detail: row.get(8)?,
                ready_at: row.get(9)?,
                lease_owner_device_id: row.get(10)?,
                lease_expires_at: row.get(11)?,
                last_error: row.get(12)?,
                updated_at: row.get(13)?,
            })
        })?;
        collect_rows(rows)
    } else {
        let mut stmt = conn.prepare(
            r#"
            SELECT session_id, job_id, network_id, segment_id, group_id, status,
                   blocked_reason, blocked_since, block_detail, ready_at,
                   lease_owner_device_id, lease_expires_at, last_error, updated_at
            FROM inference_decode_queue
            WHERE network_id = ?1
            ORDER BY updated_at ASC, session_id ASC
            "#,
        )?;
        let rows = stmt.query_map([network_id], |row| {
            Ok(DecodeQueueEntryStatus {
                session_id: row.get(0)?,
                job_id: row.get(1)?,
                network_id: row.get(2)?,
                segment_id: row.get(3)?,
                group_id: row.get(4)?,
                status: row.get(5)?,
                blocked_reason: row.get(6)?,
                blocked_since: row.get(7)?,
                block_detail: row.get(8)?,
                ready_at: row.get(9)?,
                lease_owner_device_id: row.get(10)?,
                lease_expires_at: row.get(11)?,
                last_error: row.get(12)?,
                updated_at: row.get(13)?,
            })
        })?;
        collect_rows(rows)
    }
}

fn load_serving_group_snapshots(
    conn: &rusqlite::Connection,
    network_id: &str,
    job_id: Option<&str>,
) -> Result<Vec<ServingGroupStatus>> {
    let rows = if let Some(job_id) = job_id {
        let mut stmt = conn.prepare(
            r#"
            SELECT sg.group_id, sg.session_id, sg.job_id, sg.network_id, sg.model_id, sg.phase,
                   sg.lease_owner_device_id, sg.lease_expires_at, sg.device_id, sg.ring_position,
                   sg.shard_column_start, sg.shard_column_end, sg.assigned_capacity_units,
                   sg.execution_provider, sg.status, a.status, a.lease_expires_at,
                   a.active_segment_id, sg.last_error
            FROM inference_serving_groups sg
            LEFT JOIN inference_job_assignments a
              ON a.job_id = sg.job_id
             AND a.device_id = sg.device_id
            WHERE sg.network_id = ?1 AND sg.job_id = ?2
            ORDER BY sg.session_id ASC, sg.phase ASC, sg.group_id ASC, sg.ring_position ASC
            "#,
        )?;
        let rows = stmt.query_map([network_id, job_id], |row| {
            Ok(ServingGroupRow {
                group_id: row.get(0)?,
                session_id: row.get(1)?,
                job_id: row.get(2)?,
                network_id: row.get(3)?,
                model_id: row.get(4)?,
                phase: parse_execution_phase(row.get::<_, String>(5)?.as_str())
                    .map_err(to_sql_err)?,
                lease_owner_device_id: row.get(6)?,
                lease_expires_at: row.get(7)?,
                member: ServingGroupMemberStatus {
                    device_id: row.get(8)?,
                    ring_position: row.get::<_, i64>(9)? as u32,
                    shard_column_start: row.get::<_, i64>(10)? as u32,
                    shard_column_end: row.get::<_, i64>(11)? as u32,
                    assigned_capacity_units: row.get::<_, i64>(12)? as u32,
                    execution_provider: row.get(13)?,
                    status: row.get(14)?,
                    assignment_status: row.get(15)?,
                    assignment_lease_expires_at: row.get(16)?,
                    active_segment_id: row.get(17)?,
                    last_error: row.get(18)?,
                },
            })
        })?;
        collect_rows(rows)?
    } else {
        let mut stmt = conn.prepare(
            r#"
            SELECT sg.group_id, sg.session_id, sg.job_id, sg.network_id, sg.model_id, sg.phase,
                   sg.lease_owner_device_id, sg.lease_expires_at, sg.device_id, sg.ring_position,
                   sg.shard_column_start, sg.shard_column_end, sg.assigned_capacity_units,
                   sg.execution_provider, sg.status, a.status, a.lease_expires_at,
                   a.active_segment_id, sg.last_error
            FROM inference_serving_groups sg
            LEFT JOIN inference_job_assignments a
              ON a.job_id = sg.job_id
             AND a.device_id = sg.device_id
            WHERE sg.network_id = ?1
            ORDER BY sg.session_id ASC, sg.phase ASC, sg.group_id ASC, sg.ring_position ASC
            "#,
        )?;
        let rows = stmt.query_map([network_id], |row| {
            Ok(ServingGroupRow {
                group_id: row.get(0)?,
                session_id: row.get(1)?,
                job_id: row.get(2)?,
                network_id: row.get(3)?,
                model_id: row.get(4)?,
                phase: parse_execution_phase(row.get::<_, String>(5)?.as_str())
                    .map_err(to_sql_err)?,
                lease_owner_device_id: row.get(6)?,
                lease_expires_at: row.get(7)?,
                member: ServingGroupMemberStatus {
                    device_id: row.get(8)?,
                    ring_position: row.get::<_, i64>(9)? as u32,
                    shard_column_start: row.get::<_, i64>(10)? as u32,
                    shard_column_end: row.get::<_, i64>(11)? as u32,
                    assigned_capacity_units: row.get::<_, i64>(12)? as u32,
                    execution_provider: row.get(13)?,
                    status: row.get(14)?,
                    assignment_status: row.get(15)?,
                    assignment_lease_expires_at: row.get(16)?,
                    active_segment_id: row.get(17)?,
                    last_error: row.get(18)?,
                },
            })
        })?;
        collect_rows(rows)?
    };
    let mut groups = BTreeMap::<(String, String), ServingGroupStatus>::new();
    for row in rows {
        let key = (row.session_id.clone(), row.group_id.clone());
        let entry = groups.entry(key).or_insert_with(|| ServingGroupStatus {
            group_id: row.group_id.clone(),
            session_id: row.session_id.clone(),
            job_id: row.job_id.clone(),
            network_id: row.network_id.clone(),
            model_id: row.model_id.clone(),
            phase: row.phase,
            member_count: 0,
            lease: row
                .lease_owner_device_id
                .as_ref()
                .or(row.lease_expires_at.as_ref())
                .map(|_| ServingGroupLeaseStatus {
                    lease_kind: if matches!(row.phase, ExecutionPhase::Decode) {
                        "decode_queue".to_string()
                    } else {
                        "group".to_string()
                    },
                    owner_device_id: row.lease_owner_device_id.clone(),
                    lease_expires_at: row.lease_expires_at.clone(),
                    status: if row.lease_owner_device_id.is_some() {
                        "leased".to_string()
                    } else {
                        "pending".to_string()
                    },
                }),
            members: Vec::new(),
        });
        entry.member_count += 1;
        if entry.lease.is_none() {
            entry.lease = row
                .lease_owner_device_id
                .as_ref()
                .or(row.lease_expires_at.as_ref())
                .map(|_| ServingGroupLeaseStatus {
                    lease_kind: if matches!(row.phase, ExecutionPhase::Decode) {
                        "decode_queue".to_string()
                    } else {
                        "group".to_string()
                    },
                    owner_device_id: row.lease_owner_device_id.clone(),
                    lease_expires_at: row.lease_expires_at.clone(),
                    status: if row.lease_owner_device_id.is_some() {
                        "leased".to_string()
                    } else {
                        "pending".to_string()
                    },
                });
        }
        entry.members.push(row.member);
    }

    Ok(groups.into_values().collect())
}

fn load_kv_residency_summaries(
    conn: &rusqlite::Connection,
    network_id: &str,
    job_id: Option<&str>,
) -> Result<Vec<KvResidencySummary>> {
    let rows = if let Some(job_id) = job_id {
        let mut stmt = conn.prepare(
            r#"
            SELECT session_id, job_id, network_id, model_id, status, active_segment_id,
                   kv_owner_device_id, kv_transfer_policy, kv_sequence_position,
                   kv_checkpoint_device_id, kv_checkpoint_created_at, updated_at
            FROM inference_sessions
            WHERE network_id = ?1 AND job_id = ?2
            ORDER BY updated_at DESC, session_id DESC
            "#,
        )?;
        let rows = stmt.query_map([network_id, job_id], |row| {
            Ok(KvSessionRow {
                session_id: row.get(0)?,
                job_id: row.get(1)?,
                network_id: row.get(2)?,
                model_id: row.get(3)?,
                status: row.get(4)?,
                active_segment_id: row.get(5)?,
                kv_owner_device_id: row.get(6)?,
                kv_transfer_policy: parse_kv_transfer_policy(row.get::<_, String>(7)?.as_str())
                    .map_err(to_sql_err)?,
                kv_sequence_position: row.get::<_, Option<i64>>(8)?.map(|v| v as u32),
                kv_checkpoint_device_id: row.get(9)?,
                kv_checkpoint_created_at: row.get(10)?,
                updated_at: row.get(11)?,
                latest_checkpoint: None,
            })
        })?;
        collect_rows(rows)?
    } else {
        let mut stmt = conn.prepare(
            r#"
            SELECT session_id, job_id, network_id, model_id, status, active_segment_id,
                   kv_owner_device_id, kv_transfer_policy, kv_sequence_position,
                   kv_checkpoint_device_id, kv_checkpoint_created_at, updated_at
            FROM inference_sessions
            WHERE network_id = ?1
            ORDER BY updated_at DESC, session_id DESC
            "#,
        )?;
        let rows = stmt.query_map([network_id], |row| {
            Ok(KvSessionRow {
                session_id: row.get(0)?,
                job_id: row.get(1)?,
                network_id: row.get(2)?,
                model_id: row.get(3)?,
                status: row.get(4)?,
                active_segment_id: row.get(5)?,
                kv_owner_device_id: row.get(6)?,
                kv_transfer_policy: parse_kv_transfer_policy(row.get::<_, String>(7)?.as_str())
                    .map_err(to_sql_err)?,
                kv_sequence_position: row.get::<_, Option<i64>>(8)?.map(|v| v as u32),
                kv_checkpoint_device_id: row.get(9)?,
                kv_checkpoint_created_at: row.get(10)?,
                updated_at: row.get(11)?,
                latest_checkpoint: None,
            })
        })?;
        collect_rows(rows)?
    };
    let mut sessions = rows;
    for session in &mut sessions {
        session.latest_checkpoint = load_latest_checkpoint_for_session(conn, &session.session_id)?;
    }

    let mut output = Vec::with_capacity(sessions.len());
    for session in sessions {
        output.push(KvResidencySummary {
            session_id: session.session_id.clone(),
            job_id: session.job_id.clone(),
            network_id: session.network_id.clone(),
            model_id: session.model_id.clone(),
            status: session.status.clone(),
            active_segment_id: session.active_segment_id.clone(),
            kv_owner_device_id: session.kv_owner_device_id.clone(),
            kv_transfer_policy: session.kv_transfer_policy,
            kv_sequence_position: session.kv_sequence_position,
            kv_checkpoint_device_id: session.kv_checkpoint_device_id.clone(),
            kv_checkpoint_created_at: session.kv_checkpoint_created_at.clone(),
            latest_checkpoint: session.latest_checkpoint.clone(),
            replicas: load_kv_replica_rows(conn, &session.session_id)?,
            updated_at: session.updated_at.clone(),
        });
    }

    Ok(output)
}

fn load_kv_replica_rows(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> Result<Vec<KvReplicaResidencyStatus>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT device_id, status, kv_sequence_position, checkpoint_created_at, updated_at, last_error
        FROM inference_session_replicas
        WHERE session_id = ?
        ORDER BY device_id ASC
        "#,
    )?;
    let rows = stmt.query_map([session_id], |row| {
        Ok(KvReplicaResidencyStatus {
            device_id: row.get(0)?,
            status: row.get(1)?,
            kv_sequence_position: row.get::<_, Option<i64>>(2)?.map(|v| v as u32),
            checkpoint_created_at: row.get(3)?,
            updated_at: row.get(4)?,
            last_error: row.get(5)?,
        })
    })?;
    collect_rows(rows)
}

fn load_latest_checkpoint_for_session(
    conn: &rusqlite::Connection,
    session_id: &str,
) -> Result<Option<InferenceSessionCheckpointStatus>> {
    conn.query_row(
        r#"
        SELECT checkpoint_id, source_device_id, source_segment_id, phase,
               kv_sequence_position, size_bytes, checkpoint_sha256, created_at
        FROM inference_session_checkpoints
        WHERE session_id = ?
        ORDER BY created_at DESC, checkpoint_id DESC
        LIMIT 1
        "#,
        [session_id],
        |row| {
            Ok(InferenceSessionCheckpointStatus {
                checkpoint_id: row.get(0)?,
                source_device_id: row.get(1)?,
                source_segment_id: row.get(2)?,
                phase: parse_execution_phase(row.get::<_, String>(3)?.as_str())
                    .map_err(to_sql_err)?,
                kv_sequence_position: row.get::<_, i64>(4)? as u32,
                size_bytes: row.get::<_, i64>(5)? as u64,
                sha256: row.get(6)?,
                created_at: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(DbError::from)
}

fn load_regroup_events(
    conn: &rusqlite::Connection,
    network_id: &str,
    job_id: Option<&str>,
) -> Result<Vec<RegroupEventStatus>> {
    let rows = if let Some(job_id) = job_id {
        let mut stmt = conn.prepare(
            r#"
            SELECT event_id, session_id, job_id, network_id, model_id, phase, group_id, device_id,
                   event_kind, reason, previous_status, new_status, segment_id, observed_at
            FROM inference_regroup_events
            WHERE network_id = ?1 AND job_id = ?2
            ORDER BY observed_at DESC, event_id DESC
            LIMIT 100
            "#,
        )?;
        let rows = stmt.query_map([network_id, job_id], |row| {
            Ok(RegroupEventStatus {
                event_id: row.get(0)?,
                session_id: row.get(1)?,
                job_id: row.get(2)?,
                network_id: row.get(3)?,
                model_id: row.get(4)?,
                phase: parse_execution_phase(row.get::<_, String>(5)?.as_str())
                    .map_err(to_sql_err)?,
                group_id: row.get(6)?,
                device_id: row.get(7)?,
                event_kind: row.get(8)?,
                reason: row.get(9)?,
                previous_status: row.get(10)?,
                new_status: row.get(11)?,
                segment_id: row.get(12)?,
                observed_at: row.get(13)?,
            })
        })?;
        collect_rows(rows)
    } else {
        let mut stmt = conn.prepare(
            r#"
            SELECT event_id, session_id, job_id, network_id, model_id, phase, group_id, device_id,
                   event_kind, reason, previous_status, new_status, segment_id, observed_at
            FROM inference_regroup_events
            WHERE network_id = ?1
            ORDER BY observed_at DESC, event_id DESC
            LIMIT 100
            "#,
        )?;
        let rows = stmt.query_map([network_id], |row| {
            Ok(RegroupEventStatus {
                event_id: row.get(0)?,
                session_id: row.get(1)?,
                job_id: row.get(2)?,
                network_id: row.get(3)?,
                model_id: row.get(4)?,
                phase: parse_execution_phase(row.get::<_, String>(5)?.as_str())
                    .map_err(to_sql_err)?,
                group_id: row.get(6)?,
                device_id: row.get(7)?,
                event_kind: row.get(8)?,
                reason: row.get(9)?,
                previous_status: row.get(10)?,
                new_status: row.get(11)?,
                segment_id: row.get(12)?,
                observed_at: row.get(13)?,
            })
        })?;
        collect_rows(rows)
    };
    rows
}

fn collect_rows<T>(
    rows: impl Iterator<Item = std::result::Result<T, rusqlite::Error>>,
) -> Result<Vec<T>> {
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(DbError::from)
}

fn parse_execution_phase(value: &str) -> Result<ExecutionPhase> {
    match value {
        "prefill" => Ok(ExecutionPhase::Prefill),
        "decode" => Ok(ExecutionPhase::Decode),
        other => Err(DbError::Data(format!(
            "Invalid execution phase in persisted state: {}",
            other
        ))),
    }
}

fn parse_kv_transfer_policy(value: &str) -> Result<KvTransferPolicy> {
    let normalized = if value.starts_with('"') {
        value.to_string()
    } else {
        format!("\"{}\"", value)
    };
    serde_json::from_str(&normalized).map_err(|e| {
        DbError::Data(format!(
            "Invalid KV transfer policy in persisted state ({}): {}",
            value, e
        ))
    })
}

fn to_sql_err(err: DbError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        0,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::other(err.to_string())),
    )
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

    fn seed_scheduler_visibility_fixture(db: &Database) {
        let conn = db.get_conn().expect("Failed to get connection");
        conn.execute(
            "INSERT INTO networks (network_id, name, owner_user_id, settings) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params!["net-1", "Test Network", "owner-1", "{}"],
        )
        .expect("Failed to insert network");

        for (device_id, ring_position, key_seed) in
            [("worker-a", 0_i64, 7_u8), ("worker-b", 1_i64, 8_u8)]
        {
            conn.execute(
                r#"
                INSERT INTO devices (
                    device_id, network_id, name, public_key, peer_id, capabilities,
                    status, ring_position, shard_model_id, shard_column_start, shard_column_end,
                    contributed_memory
                ) VALUES (?1, 'net-1', ?2, ?3, ?4, ?5, 'online', ?6, 'model-1', ?7, ?8, 4096)
                "#,
                rusqlite::params![
                    device_id,
                    format!("Device {}", device_id),
                    vec![key_seed; 32],
                    format!("peer-{}", device_id),
                    "{}",
                    ring_position,
                    ring_position * 64,
                    (ring_position + 1) * 64
                ],
            )
            .expect("Failed to insert device");
        }

        conn.execute(
            r#"
            INSERT INTO inference_jobs (
                job_id, network_id, submitted_by_device_id, model_id, prompt, prompt_tokens,
                max_tokens, temperature, top_p, status, ring_worker_count, completion_tokens,
                available_completion_tokens, execution_plan_json, active_segment_id, updated_at
            ) VALUES (
                'job-1', 'net-1', 'worker-a', 'model-1', 'hello', '[1,2,3]',
                32, 0.7, 0.9, 'running', 2, 3, 32, NULL, 'segment-decode', '2026-04-25T12:00:00Z'
            )
            "#,
            [],
        )
        .expect("Failed to insert job");

        conn.execute(
            r#"
            INSERT INTO inference_sessions (
                session_id, job_id, network_id, model_id, status, active_segment_id,
                kv_owner_device_id, kv_transfer_policy, kv_sequence_position,
                kv_checkpoint_device_id, kv_checkpoint_created_at, updated_at
            ) VALUES (
                'session-1', 'job-1', 'net-1', 'model-1', 'decode_pending_transfer',
                'segment-decode', 'worker-a', 'export_on_handoff', 3,
                'worker-a', '2026-04-25T12:00:05Z', '2026-04-25T12:00:06Z'
            )
            "#,
            [],
        )
        .expect("Failed to insert session");

        for (device_id, status, seq) in [
            ("worker-a", "decode_active", Some(3_i64)),
            ("worker-b", "decode_pending_transfer", Some(2_i64)),
        ] {
            conn.execute(
                r#"
                INSERT INTO inference_session_replicas (
                    session_id, device_id, job_id, status, active_segment_id,
                    kv_sequence_position, checkpoint_created_at, updated_at, last_error
                ) VALUES (?1, ?2, 'job-1', ?3, 'segment-decode', ?4, '2026-04-25T12:00:05Z', '2026-04-25T12:00:06Z', NULL)
                "#,
                rusqlite::params!["session-1", device_id, status, seq],
            )
            .expect("Failed to insert session replica");
        }

        conn.execute(
            r#"
            INSERT INTO inference_session_checkpoints (
                checkpoint_id, session_id, job_id, source_device_id, source_segment_id, phase,
                kv_sequence_position, size_bytes, checkpoint_sha256, checkpoint_bytes, created_at, updated_at
            ) VALUES (
                'ckpt-1', 'session-1', 'job-1', 'worker-a', 'segment-prefill', 'prefill',
                3, 128, 'deadbeef', X'ABCD', '2026-04-25T12:00:05Z', '2026-04-25T12:00:05Z'
            )
            "#,
            [],
        )
        .expect("Failed to insert session checkpoint");

        for (device_id, ring_position, start, end, assign_status, lease_expires_at) in [
            (
                "worker-a",
                0_i64,
                0_i64,
                64_i64,
                "acknowledged",
                Some("2026-04-25T12:01:00Z"),
            ),
            ("worker-b", 1_i64, 64_i64, 128_i64, "waiting", None),
        ] {
            conn.execute(
                r#"
                INSERT INTO inference_job_assignments (
                    assignment_id, job_id, network_id, device_id, ring_position, status, lease_expires_at,
                    assigned_at, shard_column_start, shard_column_end, assigned_capacity_units,
                    execution_provider, active_segment_id
                ) VALUES (?1, 'job-1', 'net-1', ?2, ?3, ?4, ?5, '2026-04-25T12:00:00Z', ?6, ?7, 16, 'metal', 'segment-decode')
                "#,
                rusqlite::params![
                    format!("assign-{}", device_id),
                    device_id,
                    ring_position,
                    assign_status,
                    lease_expires_at,
                    start,
                    end
                ],
            )
            .expect("Failed to insert assignment");
        }

        for (device_id, ring_position, start, end, status) in [
            ("worker-a", 0_i64, 0_i64, 64_i64, "decode_member"),
            (
                "worker-b",
                1_i64,
                64_i64,
                128_i64,
                "decode_pending_transfer",
            ),
        ] {
            conn.execute(
                r#"
                INSERT INTO inference_serving_groups (
                    group_id, session_id, job_id, network_id, model_id, phase, device_id,
                    ring_position, shard_column_start, shard_column_end, assigned_capacity_units,
                    execution_provider, status, last_error, updated_at
                ) VALUES (
                    'group-decode', 'session-1', 'job-1', 'net-1', 'model-1', 'decode', ?1,
                    ?2, ?3, ?4, 16, 'metal', ?5, NULL, '2026-04-25T12:00:06Z'
                )
                "#,
                rusqlite::params![device_id, ring_position, start, end, status],
            )
            .expect("Failed to insert serving group member");
        }

        conn.execute(
            r#"
            INSERT INTO inference_decode_queue (
                session_id, job_id, network_id, segment_id, group_id, status, ready_at,
                lease_owner_device_id, lease_expires_at, last_error, updated_at
            ) VALUES (
                'session-1', 'job-1', 'net-1', 'segment-decode', 'group-decode', 'blocked_on_prefill',
                NULL, NULL, NULL, NULL, '2026-04-25T12:00:06Z'
            )
            "#,
            [],
        )
        .expect("Failed to insert decode queue row");
    }

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

    #[test]
    fn test_network_scheduler_status_exposes_queue_groups_and_kv_residency() {
        let db = create_test_db();
        seed_scheduler_visibility_fixture(&db);

        let status = db
            .load_network_scheduler_status("net-1")
            .expect("Failed to load network scheduler status");

        assert_eq!(status.jobs.len(), 1);
        assert_eq!(status.decode_queue.len(), 1);
        assert_eq!(
            status.decode_queue[0].blocked_reason.as_deref(),
            Some("prefill_incomplete")
        );
        assert_eq!(status.serving_groups.len(), 1);
        assert_eq!(status.serving_groups[0].member_count, 2);
        assert_eq!(
            status.serving_groups[0]
                .lease
                .as_ref()
                .and_then(|lease| lease.owner_device_id.as_deref()),
            None
        );
        assert_eq!(status.kv_residency.len(), 1);
        assert_eq!(status.kv_residency[0].replicas.len(), 2);
        assert_eq!(
            status.kv_residency[0]
                .latest_checkpoint
                .as_ref()
                .map(|checkpoint| checkpoint.kv_sequence_position),
            Some(3)
        );
        assert!(!status.regroup_events.is_empty());
    }

    #[test]
    fn test_job_scheduler_status_tracks_regroup_and_leases() {
        let db = create_test_db();
        seed_scheduler_visibility_fixture(&db);

        let conn = db.get_conn().expect("Failed to get connection");
        conn.execute(
            r#"
            UPDATE inference_decode_queue
            SET status = 'leased',
                lease_owner_device_id = 'worker-a',
                lease_expires_at = '2026-04-25T12:02:00Z',
                updated_at = '2026-04-25T12:01:30Z'
            WHERE session_id = 'session-1'
            "#,
            [],
        )
        .expect("Failed to update decode queue lease");
        drop(conn);

        let status = db
            .load_job_scheduler_status("job-1")
            .expect("Failed to load job scheduler status");

        assert_eq!(status.decode_queue[0].status, "leased");
        assert_eq!(
            status.decode_queue[0].lease_owner_device_id.as_deref(),
            Some("worker-a")
        );
        assert_eq!(
            status.serving_groups[0]
                .lease
                .as_ref()
                .and_then(|lease| lease.owner_device_id.as_deref()),
            Some("worker-a")
        );
        assert!(status
            .regroup_events
            .iter()
            .any(|event| event.event_kind == "queue_lease_changed"));
    }
}
