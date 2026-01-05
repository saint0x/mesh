use crate::api::error::{ApiError, ApiResult};
use crate::db::Database;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};

/// Total number of columns in the model shard space (fixed at 8192)
const TOTAL_SHARD_COLUMNS: u32 = 8192;

/// Unique identifier for a device in the ring
pub type DeviceId = String;

/// Model shard assignment for a worker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelShard {
    pub model_id: String,
    pub column_range: (u32, u32),
    pub estimated_memory: u64,
}

/// Ring position assignment for a worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingPosition {
    pub position: u32,
    pub shard: ModelShard,
    pub left_neighbor: DeviceId,
    pub right_neighbor: DeviceId,
}

/// Worker in the ring topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub device_id: DeviceId,
    pub network_id: String,
    pub contributed_memory: u64,
    pub ring_position: Option<u32>,
    pub status: String,
}

/// Pool status for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatus {
    pub pool_id: String,
    pub model_id: String,
    pub total_workers: u32,
    pub active_workers: u32,
    pub ring_stable: bool,
    pub status: String,
}

/// Ring topology information for a network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingTopology {
    pub workers: Vec<WorkerTopologyInfo>,
    pub ring_stable: bool,
}

/// Worker topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTopologyInfo {
    pub device_id: DeviceId,
    pub position: u32,
    pub shard: ModelShard,
    pub left_neighbor: DeviceId,
    pub right_neighbor: DeviceId,
}

/// Ring Topology Manager for distributed worker coordination
///
/// Manages the ring topology of workers in a distributed inference pool.
/// Workers are assigned sequential positions in the ring, with each worker
/// responsible for a contiguous range of model columns (shards).
pub struct RingTopologyManager {
    db: Arc<Database>,
    /// In-memory cache of workers by device_id
    workers: RwLock<HashMap<DeviceId, Worker>>,
    /// Ordered list of device IDs in ring order
    ring_sequence: RwLock<Vec<DeviceId>>,
    /// Default model ID for shard assignment
    default_model_id: String,
}

impl RingTopologyManager {
    /// Create a new RingTopologyManager
    pub fn new(db: Arc<Database>, default_model_id: String) -> Self {
        Self {
            db,
            workers: RwLock::new(HashMap::new()),
            ring_sequence: RwLock::new(Vec::new()),
            default_model_id,
        }
    }

    /// Load existing ring topology from database
    pub fn load_from_db(&self, network_id: &str) -> ApiResult<()> {
        let conn = self.db.get_conn()?;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT device_id, network_id, contributed_memory, ring_position, status
                FROM devices
                WHERE network_id = ? AND ring_position IS NOT NULL
                ORDER BY ring_position
                "#,
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let workers_iter = stmt
            .query_map(params![network_id], |row| {
                Ok(Worker {
                    device_id: row.get(0)?,
                    network_id: row.get(1)?,
                    contributed_memory: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as u64,
                    ring_position: row.get(3)?,
                    status: row.get(4)?,
                })
            })
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let mut workers_map = self.workers.write().map_err(|_| {
            ApiError::Internal("Failed to acquire workers write lock".to_string())
        })?;
        let mut ring_seq = self.ring_sequence.write().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence write lock".to_string())
        })?;

        workers_map.clear();
        ring_seq.clear();

        for worker_result in workers_iter {
            let worker =
                worker_result.map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
            ring_seq.push(worker.device_id.clone());
            workers_map.insert(worker.device_id.clone(), worker);
        }

        info!(
            network_id = %network_id,
            worker_count = ring_seq.len(),
            "Loaded ring topology from database"
        );

        Ok(())
    }

    /// Add a worker to the ring topology
    ///
    /// Assigns a sequential ring position, calculates shard column range,
    /// updates all neighbors, and persists to database atomically.
    pub fn add_worker(&self, worker: Worker) -> ApiResult<RingPosition> {
        // Validate worker
        if worker.device_id.is_empty() {
            return Err(ApiError::BadRequest("device_id cannot be empty".to_string()));
        }
        if worker.network_id.is_empty() {
            return Err(ApiError::BadRequest("network_id cannot be empty".to_string()));
        }

        let conn = self.db.get_conn()?;

        // Verify device exists in database
        let device_exists: Option<String> = conn
            .query_row(
                "SELECT device_id FROM devices WHERE device_id = ?",
                params![&worker.device_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        if device_exists.is_none() {
            return Err(ApiError::NotFound(format!(
                "Device {} not found",
                worker.device_id
            )));
        }

        // Acquire write locks
        let mut workers_map = self.workers.write().map_err(|_| {
            ApiError::Internal("Failed to acquire workers write lock".to_string())
        })?;
        let mut ring_seq = self.ring_sequence.write().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence write lock".to_string())
        })?;

        // Check if worker already in ring
        if workers_map.contains_key(&worker.device_id) {
            return Err(ApiError::Conflict(format!(
                "Worker {} already in ring",
                worker.device_id
            )));
        }

        // Assign ring position (next sequential position)
        let new_position = ring_seq.len() as u32;
        let total_workers = ring_seq.len() + 1;

        // Calculate shard assignment
        let shard = self.assign_shard(new_position, total_workers as u32);

        // Calculate neighbors (will be updated for all workers)
        let (left_neighbor, right_neighbor) = if total_workers == 1 {
            // Single worker: neighbors are itself
            (worker.device_id.clone(), worker.device_id.clone())
        } else {
            let left_pos = if new_position == 0 {
                total_workers as u32 - 1
            } else {
                new_position - 1
            };
            let right_pos = (new_position + 1) % total_workers as u32;

            let left = ring_seq.get(left_pos as usize).cloned().unwrap_or_default();
            let right = if right_pos == new_position {
                worker.device_id.clone()
            } else {
                ring_seq.get(right_pos as usize).cloned().unwrap_or_default()
            };
            (left, right)
        };

        // Update worker with position
        let mut updated_worker = worker.clone();
        updated_worker.ring_position = Some(new_position);

        // Add to in-memory structures
        ring_seq.push(worker.device_id.clone());
        workers_map.insert(worker.device_id.clone(), updated_worker);

        // Persist to database atomically

        // Begin transaction
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        // Update the new worker
        let update_result = conn.execute(
            r#"
            UPDATE devices SET
                ring_position = ?,
                shard_column_start = ?,
                shard_column_end = ?,
                contributed_memory = ?,
                left_neighbor_id = ?,
                right_neighbor_id = ?
            WHERE device_id = ?
            "#,
            params![
                new_position,
                shard.column_range.0,
                shard.column_range.1,
                worker.contributed_memory as i64,
                &left_neighbor,
                &right_neighbor,
                &worker.device_id
            ],
        );

        if let Err(e) = update_result {
            conn.execute("ROLLBACK", []).ok();
            return Err(ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))));
        }

        // Update ring connections for all workers
        if let Err(e) = self.update_ring_connections_internal(&conn, &ring_seq, total_workers as u32) {
            conn.execute("ROLLBACK", []).ok();
            return Err(e);
        }

        // Commit transaction
        conn.execute("COMMIT", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        info!(
            device_id = %worker.device_id,
            position = new_position,
            total_workers = total_workers,
            shard_start = shard.column_range.0,
            shard_end = shard.column_range.1,
            "Worker added to ring"
        );

        Ok(RingPosition {
            position: new_position,
            shard,
            left_neighbor,
            right_neighbor,
        })
    }

    /// Calculate shard assignment for a given position
    ///
    /// Formula: columns_per_worker = 8192 / total_workers
    /// Worker N gets columns: [N * columns_per_worker, (N+1) * columns_per_worker)
    pub fn assign_shard(&self, position: u32, total_workers: u32) -> ModelShard {
        if total_workers == 0 {
            return ModelShard {
                model_id: self.default_model_id.clone(),
                column_range: (0, 0),
                estimated_memory: 0,
            };
        }

        let columns_per_worker = TOTAL_SHARD_COLUMNS / total_workers;
        let remainder = TOTAL_SHARD_COLUMNS % total_workers;

        // Distribute remainder columns to the first 'remainder' workers
        let start = if position < remainder {
            position * (columns_per_worker + 1)
        } else {
            remainder * (columns_per_worker + 1) + (position - remainder) * columns_per_worker
        };

        let end = if position < remainder {
            start + columns_per_worker + 1
        } else {
            start + columns_per_worker
        };

        // Estimate memory based on column count (rough estimate: 1MB per column)
        let estimated_memory = (end - start) as u64 * 1_000_000;

        ModelShard {
            model_id: self.default_model_id.clone(),
            column_range: (start, end),
            estimated_memory,
        }
    }

    /// Update ring connections for all workers
    ///
    /// Updates left_neighbor and right_neighbor for all workers in the ring.
    /// Handles wraparound: Worker 0's left = Worker N-1
    pub fn update_ring_connections(&self, network_id: &str) -> ApiResult<()> {
        let conn = self.db.get_conn()?;

        let ring_seq = self.ring_sequence.read().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence read lock".to_string())
        })?;

        let total_workers = ring_seq.len() as u32;

        // Begin transaction
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        if let Err(e) = self.update_ring_connections_internal(&conn, &ring_seq, total_workers) {
            conn.execute("ROLLBACK", []).ok();
            return Err(e);
        }

        // Commit transaction
        conn.execute("COMMIT", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        info!(
            network_id = %network_id,
            total_workers = total_workers,
            "Ring connections updated"
        );

        Ok(())
    }

    /// Internal method to update ring connections within a transaction
    fn update_ring_connections_internal(
        &self,
        conn: &rusqlite::Connection,
        ring_seq: &[DeviceId],
        total_workers: u32,
    ) -> ApiResult<()> {
        if total_workers == 0 {
            return Ok(());
        }

        for (pos, device_id) in ring_seq.iter().enumerate() {
            let position = pos as u32;

            let (left_neighbor, right_neighbor) = if total_workers == 1 {
                (device_id.clone(), device_id.clone())
            } else {
                let left_pos = if position == 0 {
                    total_workers - 1
                } else {
                    position - 1
                };
                let right_pos = (position + 1) % total_workers;

                let left = ring_seq.get(left_pos as usize).cloned().unwrap_or_default();
                let right = ring_seq.get(right_pos as usize).cloned().unwrap_or_default();
                (left, right)
            };

            // Recalculate shard for updated total
            let shard = self.assign_shard(position, total_workers);

            conn.execute(
                r#"
                UPDATE devices SET
                    ring_position = ?,
                    shard_column_start = ?,
                    shard_column_end = ?,
                    left_neighbor_id = ?,
                    right_neighbor_id = ?
                WHERE device_id = ?
                "#,
                params![
                    position,
                    shard.column_range.0,
                    shard.column_range.1,
                    &left_neighbor,
                    &right_neighbor,
                    device_id
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
        }

        Ok(())
    }

    /// Handle worker failure (leave ring)
    ///
    /// Marks worker as offline, removes from ring, and triggers redistribution.
    pub fn handle_worker_failure(&self, failed_worker_id: DeviceId) -> ApiResult<()> {
        if failed_worker_id.is_empty() {
            return Err(ApiError::BadRequest("device_id cannot be empty".to_string()));
        }

        let conn = self.db.get_conn()?;

        // Acquire write locks
        let mut workers_map = self.workers.write().map_err(|_| {
            ApiError::Internal("Failed to acquire workers write lock".to_string())
        })?;
        let mut ring_seq = self.ring_sequence.write().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence write lock".to_string())
        })?;

        // Check if worker exists in ring
        if !workers_map.contains_key(&failed_worker_id) {
            // Worker not in ring, just mark as offline in DB
            let rows_affected = conn
                .execute(
                    "UPDATE devices SET status = 'offline' WHERE device_id = ?",
                    params![&failed_worker_id],
                )
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

            if rows_affected == 0 {
                return Err(ApiError::NotFound(format!(
                    "Device {} not found",
                    failed_worker_id
                )));
            }
            return Ok(());
        }

        // Remove from in-memory structures
        workers_map.remove(&failed_worker_id);
        ring_seq.retain(|id| id != &failed_worker_id);

        let remaining_workers = ring_seq.len() as u32;

        // Begin transaction
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        // Mark failed worker as offline and clear ring position
        let update_result = conn.execute(
            r#"
            UPDATE devices SET
                status = 'offline',
                ring_position = NULL,
                left_neighbor_id = NULL,
                right_neighbor_id = NULL,
                shard_column_start = NULL,
                shard_column_end = NULL
            WHERE device_id = ?
            "#,
            params![&failed_worker_id],
        );

        if let Err(e) = update_result {
            conn.execute("ROLLBACK", []).ok();
            return Err(ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))));
        }

        // Update ring connections for remaining workers
        if let Err(e) = self.update_ring_connections_internal(&conn, &ring_seq, remaining_workers) {
            conn.execute("ROLLBACK", []).ok();
            return Err(e);
        }

        // Commit transaction
        conn.execute("COMMIT", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        warn!(
            device_id = %failed_worker_id,
            remaining_workers = remaining_workers,
            "Worker removed from ring due to failure"
        );

        // Log redistribution trigger (actual redistribution in Phase 2)
        info!(
            remaining_workers = remaining_workers,
            "Shard redistribution triggered (logging only for Phase 1)"
        );

        Ok(())
    }

    /// Get ring topology for a network
    pub fn get_topology(&self, network_id: &str) -> ApiResult<RingTopology> {
        let conn = self.db.get_conn()?;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT device_id, ring_position, shard_column_start, shard_column_end,
                       left_neighbor_id, right_neighbor_id
                FROM devices
                WHERE network_id = ? AND ring_position IS NOT NULL
                ORDER BY ring_position
                "#,
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let workers_iter = stmt
            .query_map(params![network_id], |row| {
                let device_id: String = row.get(0)?;
                let position: u32 = row.get(1)?;
                let shard_start: u32 = row.get(2)?;
                let shard_end: u32 = row.get(3)?;
                let left_neighbor: String = row.get::<_, Option<String>>(4)?.unwrap_or_default();
                let right_neighbor: String = row.get::<_, Option<String>>(5)?.unwrap_or_default();

                Ok(WorkerTopologyInfo {
                    device_id,
                    position,
                    shard: ModelShard {
                        model_id: String::new(), // Will be filled in
                        column_range: (shard_start, shard_end),
                        estimated_memory: (shard_end - shard_start) as u64 * 1_000_000,
                    },
                    left_neighbor,
                    right_neighbor,
                })
            })
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let mut workers = Vec::new();
        for worker_result in workers_iter {
            let mut worker =
                worker_result.map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
            worker.shard.model_id = self.default_model_id.clone();
            workers.push(worker);
        }

        // Ring is stable if there's at least one worker
        let ring_stable = !workers.is_empty();

        Ok(RingTopology {
            workers,
            ring_stable,
        })
    }

    /// Get worker count in ring
    pub fn worker_count(&self) -> usize {
        self.ring_sequence
            .read()
            .map(|seq| seq.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::device_service;

    fn create_test_ring_manager() -> (RingTopologyManager, Database) {
        let db = create_test_db();
        let manager = RingTopologyManager::new(Arc::new(db.clone()), "test-model".to_string());
        (manager, db)
    }

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            cpu_cores: 8,
            ram_mb: 16384,
            os: "macos".into(),
            arch: "aarch64".into(),
            has_gpu: false,
            tier: Tier::Tier2,
        }
    }

    fn register_test_device(db: &Database, device_id: &str, network_id: &str) {
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        // Generate unique public key based on device_id hash
        let mut public_key = [0u8; 32];
        let hash = device_id.as_bytes();
        for (i, byte) in hash.iter().cycle().take(32).enumerate() {
            public_key[i] = *byte;
        }
        device_service::register_device(
            db,
            &keypair,
            device_id.to_string(),
            network_id.to_string(),
            "Test Device".to_string(),
            public_key.to_vec(),
            test_capabilities(),
        )
        .unwrap();
    }

    #[test]
    fn test_assign_shard_single_worker() {
        let (manager, _db) = create_test_ring_manager();
        let shard = manager.assign_shard(0, 1);

        assert_eq!(shard.column_range, (0, TOTAL_SHARD_COLUMNS));
        assert_eq!(shard.model_id, "test-model");
    }

    #[test]
    fn test_assign_shard_two_workers() {
        let (manager, _db) = create_test_ring_manager();

        let shard0 = manager.assign_shard(0, 2);
        let shard1 = manager.assign_shard(1, 2);

        assert_eq!(shard0.column_range, (0, 4096));
        assert_eq!(shard1.column_range, (4096, 8192));
    }

    #[test]
    fn test_assign_shard_three_workers() {
        let (manager, _db) = create_test_ring_manager();

        let shard0 = manager.assign_shard(0, 3);
        let shard1 = manager.assign_shard(1, 3);
        let shard2 = manager.assign_shard(2, 3);

        // 8192 / 3 = 2730 remainder 2
        // Worker 0: 0-2731 (2731 columns)
        // Worker 1: 2731-5462 (2731 columns)
        // Worker 2: 5462-8192 (2730 columns)
        assert_eq!(shard0.column_range.0, 0);
        assert_eq!(shard2.column_range.1, TOTAL_SHARD_COLUMNS);

        // Verify no gaps
        assert_eq!(shard0.column_range.1, shard1.column_range.0);
        assert_eq!(shard1.column_range.1, shard2.column_range.0);
    }

    #[test]
    fn test_assign_shard_no_overlap() {
        let (manager, _db) = create_test_ring_manager();

        for total in 1..=20 {
            let mut ranges: Vec<(u32, u32)> = Vec::new();
            for pos in 0..total {
                let shard = manager.assign_shard(pos, total);
                ranges.push(shard.column_range);
            }

            // Verify no overlaps
            for i in 0..ranges.len() {
                for j in (i + 1)..ranges.len() {
                    let (start_i, end_i) = ranges[i];
                    let (start_j, end_j) = ranges[j];
                    assert!(
                        end_i <= start_j || end_j <= start_i,
                        "Overlapping ranges at total={}: {:?} and {:?}",
                        total, ranges[i], ranges[j]
                    );
                }
            }

            // Verify full coverage
            assert_eq!(ranges[0].0, 0, "First shard should start at 0");
            assert_eq!(
                ranges.last().unwrap().1,
                TOTAL_SHARD_COLUMNS,
                "Last shard should end at {}",
                TOTAL_SHARD_COLUMNS
            );
        }
    }

    #[test]
    fn test_add_single_worker() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";
        let device_id = "device-1";

        register_test_device(&db, device_id, network_id);

        let worker = Worker {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
            ring_position: None,
            status: "online".to_string(),
        };

        let position = manager.add_worker(worker).unwrap();

        assert_eq!(position.position, 0);
        assert_eq!(position.left_neighbor, device_id);
        assert_eq!(position.right_neighbor, device_id);
        assert_eq!(position.shard.column_range, (0, TOTAL_SHARD_COLUMNS));
    }

    #[test]
    fn test_add_multiple_workers() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        // Register 3 devices
        for i in 0..3 {
            register_test_device(&db, &format!("device-{}", i), network_id);
        }

        // Add to ring
        let mut positions = Vec::new();
        for i in 0..3 {
            let worker = Worker {
                device_id: format!("device-{}", i),
                network_id: network_id.to_string(),
                contributed_memory: 8_000_000_000,
                ring_position: None,
                status: "online".to_string(),
            };
            positions.push(manager.add_worker(worker).unwrap());
        }

        // Verify positions
        assert_eq!(positions[0].position, 0);
        assert_eq!(positions[1].position, 1);
        assert_eq!(positions[2].position, 2);

        // Verify final ring connections
        // After all workers added, ring is: 0 -> 1 -> 2 -> 0
        // Device 0: left=2, right=1
        // Device 1: left=0, right=2
        // Device 2: left=1, right=0
        let topology = manager.get_topology(network_id).unwrap();
        assert_eq!(topology.workers.len(), 3);

        let w0 = topology.workers.iter().find(|w| w.position == 0).unwrap();
        let w1 = topology.workers.iter().find(|w| w.position == 1).unwrap();
        let w2 = topology.workers.iter().find(|w| w.position == 2).unwrap();

        assert_eq!(w0.left_neighbor, "device-2");
        assert_eq!(w0.right_neighbor, "device-1");
        assert_eq!(w1.left_neighbor, "device-0");
        assert_eq!(w1.right_neighbor, "device-2");
        assert_eq!(w2.left_neighbor, "device-1");
        assert_eq!(w2.right_neighbor, "device-0");
    }

    #[test]
    fn test_add_worker_nonexistent_device() {
        let (manager, _db) = create_test_ring_manager();

        let worker = Worker {
            device_id: "nonexistent".to_string(),
            network_id: "test-network".to_string(),
            contributed_memory: 8_000_000_000,
            ring_position: None,
            status: "online".to_string(),
        };

        let result = manager.add_worker(worker);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    #[test]
    fn test_add_duplicate_worker() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";
        let device_id = "device-dup";

        register_test_device(&db, device_id, network_id);

        let worker = Worker {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
            ring_position: None,
            status: "online".to_string(),
        };

        // First add should succeed
        manager.add_worker(worker.clone()).unwrap();

        // Second add should fail
        let result = manager.add_worker(worker);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::Conflict(_)));
    }

    #[test]
    fn test_handle_worker_failure() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        // Register and add 3 devices
        for i in 0..3 {
            register_test_device(&db, &format!("device-{}", i), network_id);
            let worker = Worker {
                device_id: format!("device-{}", i),
                network_id: network_id.to_string(),
                contributed_memory: 8_000_000_000,
                ring_position: None,
                status: "online".to_string(),
            };
            manager.add_worker(worker).unwrap();
        }

        // Remove middle worker
        manager.handle_worker_failure("device-1".to_string()).unwrap();

        // Verify remaining ring
        let topology = manager.get_topology(network_id).unwrap();
        assert_eq!(topology.workers.len(), 2);

        // Check positions are 0 and 1 (renumbered)
        let positions: Vec<u32> = topology.workers.iter().map(|w| w.position).collect();
        assert!(positions.contains(&0));
        assert!(positions.contains(&1));
    }

    #[test]
    fn test_handle_worker_failure_nonexistent() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        // Register but don't add to ring
        register_test_device(&db, "device-1", network_id);

        // Should succeed (just marks as offline)
        let result = manager.handle_worker_failure("device-1".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn test_handle_worker_failure_not_found() {
        let (manager, _db) = create_test_ring_manager();

        let result = manager.handle_worker_failure("nonexistent".to_string());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    #[test]
    fn test_get_topology_empty() {
        let (manager, _db) = create_test_ring_manager();

        let topology = manager.get_topology("test-network").unwrap();
        assert!(topology.workers.is_empty());
        assert!(!topology.ring_stable);
    }

    #[test]
    fn test_ring_stability() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        register_test_device(&db, "device-1", network_id);

        // Empty ring should not be stable
        let topology = manager.get_topology(network_id).unwrap();
        assert!(!topology.ring_stable);

        // Add worker
        let worker = Worker {
            device_id: "device-1".to_string(),
            network_id: network_id.to_string(),
            contributed_memory: 8_000_000_000,
            ring_position: None,
            status: "online".to_string(),
        };
        manager.add_worker(worker).unwrap();

        // Ring with worker should be stable
        let topology = manager.get_topology(network_id).unwrap();
        assert!(topology.ring_stable);
    }

    #[test]
    fn test_all_workers_leave() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        // Register and add 2 devices
        for i in 0..2 {
            register_test_device(&db, &format!("device-{}", i), network_id);
            let worker = Worker {
                device_id: format!("device-{}", i),
                network_id: network_id.to_string(),
                contributed_memory: 8_000_000_000,
                ring_position: None,
                status: "online".to_string(),
            };
            manager.add_worker(worker).unwrap();
        }

        // Remove all workers
        manager.handle_worker_failure("device-0".to_string()).unwrap();
        manager.handle_worker_failure("device-1".to_string()).unwrap();

        // Verify ring is empty
        let topology = manager.get_topology(network_id).unwrap();
        assert!(topology.workers.is_empty());
        assert!(!topology.ring_stable);
    }

    #[test]
    fn test_ten_workers_ring() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        // Register and add 10 devices
        for i in 0..10 {
            register_test_device(&db, &format!("device-{}", i), network_id);
            let worker = Worker {
                device_id: format!("device-{}", i),
                network_id: network_id.to_string(),
                contributed_memory: 8_000_000_000,
                ring_position: None,
                status: "online".to_string(),
            };
            manager.add_worker(worker).unwrap();
        }

        // Verify ring correctness
        let topology = manager.get_topology(network_id).unwrap();
        assert_eq!(topology.workers.len(), 10);
        assert!(topology.ring_stable);

        // Verify positions are 0-9
        let mut positions: Vec<u32> = topology.workers.iter().map(|w| w.position).collect();
        positions.sort();
        assert_eq!(positions, (0..10).collect::<Vec<_>>());

        // Verify neighbor correctness for each worker
        for worker in &topology.workers {
            let left_pos = if worker.position == 0 { 9 } else { worker.position - 1 };
            let right_pos = (worker.position + 1) % 10;

            let expected_left = format!("device-{}", left_pos);
            let expected_right = format!("device-{}", right_pos);

            assert_eq!(worker.left_neighbor, expected_left);
            assert_eq!(worker.right_neighbor, expected_right);
        }

        // Verify shard coverage
        let mut all_ranges: Vec<(u32, u32)> = topology.workers.iter()
            .map(|w| w.shard.column_range)
            .collect();
        all_ranges.sort_by_key(|r| r.0);

        // First should start at 0, last should end at 8192
        assert_eq!(all_ranges[0].0, 0);
        assert_eq!(all_ranges[9].1, TOTAL_SHARD_COLUMNS);

        // No gaps between ranges
        for i in 0..9 {
            assert_eq!(all_ranges[i].1, all_ranges[i + 1].0);
        }
    }
}
