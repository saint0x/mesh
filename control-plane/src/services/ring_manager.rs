use crate::api::error::{ApiError, ApiResult};
use crate::connectivity::{
    ConnectivityPath, ConnectivityStatus, DeviceConnectivityState, DirectCandidateScope,
    DirectPeerCandidate, InferenceSchedulingPolicy,
};
use crate::db::Database;
use crate::device::{DeviceCapabilities, Tier};
use crate::model_assets;
use crate::services::network_service;
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

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
    pub model_id: String,
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
    pub peer_punch_plans: Vec<PeerPunchPlan>,
}

/// Worker topology information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerTopologyInfo {
    pub device_id: DeviceId,
    pub peer_id: String,
    pub position: u32,
    pub status: String,
    pub contributed_memory: u64,
    pub shard: ModelShard,
    pub left_neighbor: DeviceId,
    pub right_neighbor: DeviceId,
    pub connectivity_state: Option<DeviceConnectivityState>,
    pub listen_addrs: Vec<String>,
    pub direct_candidates: Vec<DirectPeerCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PunchPathReason {
    RelayPath,
    DegradedConnectivity,
    PrivateReachabilityOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PunchPathStrategy {
    SimultaneousDial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerPunchPlan {
    pub source_device_id: String,
    pub target_device_id: String,
    pub target_peer_id: String,
    pub strategy: PunchPathStrategy,
    pub reason: PunchPathReason,
    pub relay_rendezvous_required: bool,
    pub attempt_window_ms: u64,
    pub issued_at_ms: u64,
    pub target_candidates: Vec<DirectPeerCandidate>,
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
    /// Serializes topology mutations so ring position updates remain atomic under contention.
    mutation_lock: Mutex<()>,
}

#[derive(Debug, Clone)]
struct WorkerCapacityProfile {
    tier: Tier,
    contributed_memory: u64,
    fallback_memory_mb: u64,
}

fn tier_capacity_units(policy: &InferenceSchedulingPolicy, tier: Tier) -> u32 {
    match tier {
        Tier::Tier0 => policy.tier_capacity_units.tier0,
        Tier::Tier1 => policy.tier_capacity_units.tier1,
        Tier::Tier2 => policy.tier_capacity_units.tier2,
        Tier::Tier3 => policy.tier_capacity_units.tier3,
        Tier::Tier4 => policy.tier_capacity_units.tier4,
    }
}

fn effective_capacity_weight(
    profile: &WorkerCapacityProfile,
    scheduling_policy: &InferenceSchedulingPolicy,
) -> u128 {
    let memory_mb = if profile.contributed_memory > 0 {
        (profile.contributed_memory / (1024 * 1024)).max(1)
    } else {
        profile.fallback_memory_mb.max(1)
    };
    u128::from(tier_capacity_units(scheduling_policy, profile.tier).max(1)) * u128::from(memory_mb)
}

fn allocate_weighted_column_ranges(
    total_columns: u32,
    profiles: &[WorkerCapacityProfile],
    scheduling_policy: &InferenceSchedulingPolicy,
) -> Vec<(u32, u32)> {
    if profiles.is_empty() {
        return Vec::new();
    }

    let weights: Vec<u128> = profiles
        .iter()
        .map(|profile| effective_capacity_weight(profile, scheduling_policy).max(1))
        .collect();
    let total_weight: u128 = weights.iter().sum::<u128>().max(1);

    let mut widths: Vec<u32> = weights
        .iter()
        .map(|weight| ((u128::from(total_columns) * *weight) / total_weight) as u32)
        .collect();
    let assigned_columns: u32 = widths.iter().sum();
    let mut remaining_columns = total_columns.saturating_sub(assigned_columns);

    let mut remainders: Vec<(usize, u128)> = weights
        .iter()
        .enumerate()
        .map(|(index, weight)| (index, (u128::from(total_columns) * *weight) % total_weight))
        .collect();
    remainders.sort_by(
        |(left_index, left_remainder), (right_index, right_remainder)| {
            right_remainder
                .cmp(left_remainder)
                .then_with(|| left_index.cmp(right_index))
        },
    );

    let mut cursor = 0usize;
    while remaining_columns > 0 {
        widths[remainders[cursor].0] += 1;
        remaining_columns -= 1;
        cursor = (cursor + 1) % remainders.len();
    }

    let mut start = 0u32;
    widths
        .into_iter()
        .map(|width| {
            let end = start + width;
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

impl RingTopologyManager {
    /// Create a new RingTopologyManager
    pub fn new(db: Arc<Database>) -> Self {
        Self {
            db,
            workers: RwLock::new(HashMap::new()),
            ring_sequence: RwLock::new(Vec::new()),
            mutation_lock: Mutex::new(()),
        }
    }

    /// Load existing ring topology from database
    pub fn load_from_db(&self, network_id: &str) -> ApiResult<()> {
        let conn = self.db.get_conn()?;

        let mut stmt = conn
            .prepare(
                r#"
                SELECT device_id, network_id, shard_model_id, contributed_memory, ring_position, status
                FROM devices
                WHERE network_id = ? AND ring_position IS NOT NULL AND shard_model_id IS NOT NULL
                ORDER BY ring_position
                "#,
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let workers_iter = stmt
            .query_map(params![network_id], |row| {
                Ok(Worker {
                    device_id: row.get(0)?,
                    network_id: row.get(1)?,
                    model_id: row.get(2)?,
                    contributed_memory: row.get::<_, Option<i64>>(3)?.unwrap_or(0) as u64,
                    ring_position: row.get(4)?,
                    status: row.get(5)?,
                })
            })
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let mut workers_map = self
            .workers
            .write()
            .map_err(|_| ApiError::Internal("Failed to acquire workers write lock".to_string()))?;
        let mut ring_seq = self.ring_sequence.write().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence write lock".to_string())
        })?;

        workers_map.clear();
        ring_seq.clear();

        for worker_result in workers_iter {
            let worker = worker_result
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
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
    /// Assigns a sequential ring position, recalculates weighted shard column ranges,
    /// updates all neighbors, and persists to database atomically.
    pub fn add_worker(&self, worker: Worker) -> ApiResult<RingPosition> {
        let _mutation_guard = self
            .mutation_lock
            .lock()
            .map_err(|_| ApiError::Internal("Failed to acquire ring mutation lock".to_string()))?;

        if worker.device_id.is_empty() {
            return Err(ApiError::BadRequest(
                "device_id cannot be empty".to_string(),
            ));
        }
        if worker.network_id.is_empty() {
            return Err(ApiError::BadRequest(
                "network_id cannot be empty".to_string(),
            ));
        }

        let conn = self.db.get_conn()?;

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

        let mut workers_map = self
            .workers
            .write()
            .map_err(|_| ApiError::Internal("Failed to acquire workers write lock".to_string()))?;
        let mut ring_seq = self.ring_sequence.write().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence write lock".to_string())
        })?;

        if workers_map.contains_key(&worker.device_id) {
            return Err(ApiError::Conflict(format!(
                "Worker {} already in ring",
                worker.device_id
            )));
        }

        let new_position = ring_seq.len() as u32;
        let total_workers = ring_seq.len() + 1;

        let mut updated_worker = worker.clone();
        updated_worker.ring_position = Some(new_position);
        ring_seq.push(worker.device_id.clone());
        workers_map.insert(worker.device_id.clone(), updated_worker);

        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let update_result = conn.execute(
            r#"
            UPDATE devices SET
                ring_position = ?,
                shard_model_id = ?,
                shard_column_start = 0,
                shard_column_end = 0,
                contributed_memory = ?,
                left_neighbor_id = NULL,
                right_neighbor_id = NULL
            WHERE device_id = ?
            "#,
            params![
                new_position,
                &worker.model_id,
                worker.contributed_memory as i64,
                &worker.device_id
            ],
        );

        if let Err(e) = update_result {
            conn.execute("ROLLBACK", []).ok();
            return Err(ApiError::Database(Box::new(crate::db::DbError::Rusqlite(
                e,
            ))));
        }

        let assigned_shards = match self.update_ring_connections_internal(&conn, &ring_seq) {
            Ok(assigned_shards) => assigned_shards,
            Err(e) => {
                conn.execute("ROLLBACK", []).ok();
                return Err(e);
            }
        };

        conn.execute("COMMIT", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let shard = assigned_shards
            .get(&worker.device_id)
            .cloned()
            .ok_or_else(|| {
                ApiError::Internal("Failed to resolve shard after ring update".to_string())
            })?;

        let (left_neighbor, right_neighbor): (String, String) = conn
            .query_row(
                "SELECT left_neighbor_id, right_neighbor_id FROM devices WHERE device_id = ?",
                params![&worker.device_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
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

    fn load_worker_capacity_profiles(
        &self,
        conn: &rusqlite::Connection,
        ring_seq: &[DeviceId],
    ) -> ApiResult<Vec<WorkerCapacityProfile>> {
        let mut profiles = Vec::with_capacity(ring_seq.len());
        for device_id in ring_seq {
            let (capabilities_json, contributed_memory): (String, Option<i64>) = conn
                .query_row(
                    "SELECT capabilities, contributed_memory FROM devices WHERE device_id = ?",
                    params![device_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
            let capabilities: DeviceCapabilities = serde_json::from_str(&capabilities_json)
                .map_err(|e| {
                    ApiError::Internal(format!("Failed to parse device capabilities: {}", e))
                })?;
            profiles.push(WorkerCapacityProfile {
                tier: capabilities.tier,
                contributed_memory: contributed_memory.unwrap_or_default().max(0) as u64,
                fallback_memory_mb: capabilities.ram_mb as u64
                    + capabilities.gpu_vram_mb.unwrap_or_default() as u64,
            });
        }
        Ok(profiles)
    }

    fn assign_shards_for_ring(
        &self,
        conn: &rusqlite::Connection,
        ring_seq: &[DeviceId],
    ) -> ApiResult<Vec<ModelShard>> {
        if ring_seq.is_empty() {
            return Ok(Vec::new());
        }

        let (network_id, model_id): (String, String) = conn
            .query_row(
                "SELECT network_id, shard_model_id FROM devices WHERE device_id = ? AND shard_model_id IS NOT NULL",
                params![&ring_seq[0]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        for device_id in ring_seq.iter().skip(1) {
            let candidate_model_id: String = conn
                .query_row(
                    "SELECT shard_model_id FROM devices WHERE device_id = ? AND shard_model_id IS NOT NULL",
                    params![device_id],
                    |row| row.get(0),
                )
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
            if candidate_model_id != model_id {
                return Err(ApiError::Conflict(
                    "Ring contains mixed model IDs; weighted shard assignment requires a single model ring"
                        .to_string(),
                ));
            }
        }

        let scheduling_policy =
            network_service::load_network_settings(&self.db, &network_id)?.scheduling_policy;
        let profiles = self.load_worker_capacity_profiles(conn, ring_seq)?;
        let manifest = model_assets::load_model_manifest(&model_id)?;
        let ranges = allocate_weighted_column_ranges(
            manifest.tensor_parallelism_dim,
            &profiles,
            &scheduling_policy,
        );

        Ok(profiles
            .into_iter()
            .zip(ranges.into_iter())
            .map(|(_profile, column_range)| ModelShard {
                model_id: model_id.clone(),
                estimated_memory: ((manifest.total_model_bytes as u128)
                    * u128::from(column_range.1.saturating_sub(column_range.0))
                    / u128::from(manifest.tensor_parallelism_dim.max(1)))
                    as u64,
                column_range,
            })
            .collect())
    }

    #[cfg(test)]
    fn assign_shard(
        &self,
        model_id: &str,
        position: u32,
        total_workers: u32,
    ) -> ApiResult<ModelShard> {
        let manifest = model_assets::load_model_manifest(model_id)?;
        let profiles = (0..total_workers)
            .map(|_| WorkerCapacityProfile {
                tier: Tier::Tier1,
                contributed_memory: 1,
                fallback_memory_mb: 1,
            })
            .collect::<Vec<_>>();
        let ranges = allocate_weighted_column_ranges(
            manifest.tensor_parallelism_dim,
            &profiles,
            &InferenceSchedulingPolicy::default(),
        );
        let column_range = ranges
            .get(position as usize)
            .copied()
            .ok_or_else(|| ApiError::BadRequest("invalid test shard position".to_string()))?;
        Ok(ModelShard {
            model_id: model_id.to_string(),
            estimated_memory: ((manifest.total_model_bytes as u128)
                * u128::from(column_range.1.saturating_sub(column_range.0))
                / u128::from(manifest.tensor_parallelism_dim.max(1)))
                as u64,
            column_range,
        })
    }

    /// Update ring connections for all workers
    ///
    /// Updates left_neighbor and right_neighbor for all workers in the ring.
    /// Handles wraparound: Worker 0's left = Worker N-1
    pub fn update_ring_connections(&self, network_id: &str) -> ApiResult<()> {
        let _mutation_guard = self
            .mutation_lock
            .lock()
            .map_err(|_| ApiError::Internal("Failed to acquire ring mutation lock".to_string()))?;

        let conn = self.db.get_conn()?;

        let ring_seq = self.ring_sequence.read().map_err(|_| {
            ApiError::Internal("Failed to acquire ring_sequence read lock".to_string())
        })?;

        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        if let Err(e) = self.update_ring_connections_internal(&conn, &ring_seq) {
            conn.execute("ROLLBACK", []).ok();
            return Err(e);
        }

        conn.execute("COMMIT", [])
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        info!(
            network_id = %network_id,
            total_workers = ring_seq.len(),
            "Ring connections updated"
        );

        Ok(())
    }

    /// Internal method to update ring connections within a transaction
    fn update_ring_connections_internal(
        &self,
        conn: &rusqlite::Connection,
        ring_seq: &[DeviceId],
    ) -> ApiResult<HashMap<DeviceId, ModelShard>> {
        let total_workers = ring_seq.len() as u32;
        if total_workers == 0 {
            return Ok(HashMap::new());
        }

        let assigned_shards = self.assign_shards_for_ring(conn, ring_seq)?;
        let mut shard_map = HashMap::with_capacity(assigned_shards.len());

        for ((pos, device_id), shard) in
            ring_seq.iter().enumerate().zip(assigned_shards.into_iter())
        {
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
                let right = ring_seq
                    .get(right_pos as usize)
                    .cloned()
                    .unwrap_or_default();
                (left, right)
            };

            conn.execute(
                r#"
                UPDATE devices SET
                    ring_position = ?,
                    shard_model_id = ?,
                    shard_column_start = ?,
                    shard_column_end = ?,
                    left_neighbor_id = ?,
                    right_neighbor_id = ?
                WHERE device_id = ?
                "#,
                params![
                    position,
                    &shard.model_id,
                    shard.column_range.0,
                    shard.column_range.1,
                    &left_neighbor,
                    &right_neighbor,
                    device_id
                ],
            )
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

            shard_map.insert(device_id.clone(), shard);
        }

        Ok(shard_map)
    }

    /// Handle worker failure (leave ring)
    ///
    /// Marks worker as offline, removes from ring, and triggers redistribution.
    pub fn handle_worker_failure(&self, failed_worker_id: DeviceId) -> ApiResult<()> {
        let _mutation_guard = self
            .mutation_lock
            .lock()
            .map_err(|_| ApiError::Internal("Failed to acquire ring mutation lock".to_string()))?;

        if failed_worker_id.is_empty() {
            return Err(ApiError::BadRequest(
                "device_id cannot be empty".to_string(),
            ));
        }

        let conn = self.db.get_conn()?;

        // Acquire write locks
        let mut workers_map = self
            .workers
            .write()
            .map_err(|_| ApiError::Internal("Failed to acquire workers write lock".to_string()))?;
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
                shard_model_id = NULL,
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
            return Err(ApiError::Database(Box::new(crate::db::DbError::Rusqlite(
                e,
            ))));
        }

        // Update ring connections for remaining workers
        if let Err(e) = self.update_ring_connections_internal(&conn, &ring_seq) {
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
                SELECT device_id, ring_position, shard_model_id, shard_column_start, shard_column_end,
                       left_neighbor_id, right_neighbor_id, status, contributed_memory, connectivity_state,
                       peer_id, listen_addrs, direct_candidates
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
                let shard_model_id = row.get::<_, String>(2)?;
                let shard_start: u32 = row.get(3)?;
                let shard_end: u32 = row.get(4)?;
                let left_neighbor: String = row.get::<_, Option<String>>(5)?.unwrap_or_default();
                let right_neighbor: String = row.get::<_, Option<String>>(6)?.unwrap_or_default();
                let status: String = row.get(7)?;
                let contributed_memory = row.get::<_, Option<i64>>(8)?.unwrap_or(0) as u64;
                let connectivity_state = row
                    .get::<_, Option<String>>(9)?
                    .map(|json| serde_json::from_str(&json))
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            9,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?;
                let peer_id: String = row.get(10)?;
                let listen_addrs = row
                    .get::<_, Option<String>>(11)?
                    .map(|json| serde_json::from_str(&json))
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            11,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?
                    .unwrap_or_default();
                let direct_candidates = row
                    .get::<_, Option<String>>(12)?
                    .map(|json| serde_json::from_str(&json))
                    .transpose()
                    .map_err(|e| {
                        rusqlite::Error::FromSqlConversionFailure(
                            12,
                            rusqlite::types::Type::Text,
                            Box::new(e),
                        )
                    })?
                    .unwrap_or_default();

                Ok(WorkerTopologyInfo {
                    device_id,
                    peer_id,
                    position,
                    status,
                    contributed_memory,
                    shard: ModelShard {
                        model_id: shard_model_id,
                        column_range: (shard_start, shard_end),
                        estimated_memory: contributed_memory,
                    },
                    left_neighbor,
                    right_neighbor,
                    connectivity_state,
                    listen_addrs,
                    direct_candidates,
                })
            })
            .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

        let mut workers = Vec::new();
        for worker_result in workers_iter {
            let worker = worker_result
                .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;
            workers.push(worker);
        }

        // Ring is stable if there's at least one worker
        let ring_stable = !workers.is_empty();

        Ok(RingTopology {
            peer_punch_plans: build_peer_punch_plans(&workers),
            workers,
            ring_stable,
        })
    }

    /// Get worker count in ring
    pub fn worker_count(&self) -> usize {
        self.ring_sequence.read().map(|seq| seq.len()).unwrap_or(0)
    }
}

fn build_peer_punch_plans(workers: &[WorkerTopologyInfo]) -> Vec<PeerPunchPlan> {
    let issued_at_ms = current_epoch_ms();
    let mut plans = Vec::new();

    for source in workers.iter().filter(|worker| worker.status == "online") {
        for target in workers.iter().filter(|worker| worker.status == "online") {
            if source.device_id == target.device_id
                || source.direct_candidates.is_empty()
                || target.direct_candidates.is_empty()
            {
                continue;
            }

            let Some(reason) = classify_punch_reason(source, target) else {
                continue;
            };

            plans.push(PeerPunchPlan {
                source_device_id: source.device_id.clone(),
                target_device_id: target.device_id.clone(),
                target_peer_id: target.peer_id.clone(),
                strategy: PunchPathStrategy::SimultaneousDial,
                reason,
                relay_rendezvous_required: uses_relay_path(source) || uses_relay_path(target),
                attempt_window_ms: 5_000,
                issued_at_ms,
                target_candidates: target.direct_candidates.iter().take(6).cloned().collect(),
            });
        }
    }

    plans
}

fn classify_punch_reason(
    source: &WorkerTopologyInfo,
    target: &WorkerTopologyInfo,
) -> Option<PunchPathReason> {
    if uses_relay_path(source) || uses_relay_path(target) {
        return Some(PunchPathReason::RelayPath);
    }

    if has_degraded_connectivity(source) || has_degraded_connectivity(target) {
        return Some(PunchPathReason::DegradedConnectivity);
    }

    if !has_publicish_candidate(source) || !has_publicish_candidate(target) {
        return Some(PunchPathReason::PrivateReachabilityOnly);
    }

    None
}

fn uses_relay_path(worker: &WorkerTopologyInfo) -> bool {
    worker
        .connectivity_state
        .as_ref()
        .map(|state| state.active_path == ConnectivityPath::Relayed)
        .unwrap_or(false)
}

fn has_degraded_connectivity(worker: &WorkerTopologyInfo) -> bool {
    worker
        .connectivity_state
        .as_ref()
        .map(|state| {
            matches!(
                state.status,
                ConnectivityStatus::Unknown
                    | ConnectivityStatus::Degraded
                    | ConnectivityStatus::Disconnected
            )
        })
        .unwrap_or(true)
}

fn has_publicish_candidate(worker: &WorkerTopologyInfo) -> bool {
    worker.direct_candidates.iter().any(|candidate| {
        matches!(
            candidate.scope,
            DirectCandidateScope::Public | DirectCandidateScope::Dns
        )
    })
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::{
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath,
        InferenceSchedulingPolicy, NetworkConnectivity,
    };
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::provider::{ExecutionProviderInfo, ExecutionProviderKind};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::device_service;

    fn ensure_test_model_assets() {
        crate::model_assets::testsupport::ensure_test_model("test-model", 8192);
    }

    fn create_test_ring_manager() -> (RingTopologyManager, Database) {
        ensure_test_model_assets();
        let db = create_test_db();
        let manager = RingTopologyManager::new(Arc::new(db.clone()));
        (manager, db)
    }

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            tier: Tier::Tier2,
            cpu_cores: 8,
            ram_mb: 16384,
            gpu_present: false,
            gpu_vram_mb: None,
            os: "macos".into(),
            arch: "aarch64".into(),
            execution_providers: vec![
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cpu,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Metal,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cuda,
                    available: false,
                    reason: Some("cuda provider is only available on Linux builds".into()),
                },
            ],
            default_execution_provider: ExecutionProviderKind::Metal,
        }
    }

    fn test_connectivity() -> NetworkConnectivity {
        NetworkConnectivity {
            preferred_path: ConnectivityPath::Relayed,
            attachments: vec![ConnectivityAttachment {
                kind: ConnectivityAttachmentKind::Libp2pRelay,
                endpoint: "/dns4/relay.mesh.example/tcp/4001".to_string(),
                priority: 0,
            }],
        }
    }

    fn register_test_device(db: &Database, device_id: &str, network_id: &str) {
        let _ = crate::services::network_service::create_network(
            db,
            network_id.to_string(),
            network_id.to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
        );
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
            format!("test-peer-manager-{}", device_id),
            test_capabilities(),
        )
        .unwrap();
    }

    #[test]
    fn test_assign_shard_single_worker() {
        let (manager, _db) = create_test_ring_manager();
        let shard = manager.assign_shard("test-model", 0, 1).unwrap();

        assert_eq!(shard.column_range, (0, 8192));
        assert_eq!(shard.model_id, "test-model");
    }

    #[test]
    fn test_assign_shard_two_workers() {
        let (manager, _db) = create_test_ring_manager();

        let shard0 = manager.assign_shard("test-model", 0, 2).unwrap();
        let shard1 = manager.assign_shard("test-model", 1, 2).unwrap();

        assert_eq!(shard0.column_range, (0, 4096));
        assert_eq!(shard1.column_range, (4096, 8192));
    }

    #[test]
    fn test_assign_shard_three_workers() {
        let (manager, _db) = create_test_ring_manager();

        let shard0 = manager.assign_shard("test-model", 0, 3).unwrap();
        let shard1 = manager.assign_shard("test-model", 1, 3).unwrap();
        let shard2 = manager.assign_shard("test-model", 2, 3).unwrap();

        // 8192 / 3 = 2730 remainder 2
        // Worker 0: 0-2731 (2731 columns)
        // Worker 1: 2731-5462 (2731 columns)
        // Worker 2: 5462-8192 (2730 columns)
        assert_eq!(shard0.column_range.0, 0);
        assert_eq!(shard2.column_range.1, 8192);

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
                let shard = manager.assign_shard("test-model", pos, total).unwrap();
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
                        total,
                        ranges[i],
                        ranges[j]
                    );
                }
            }

            // Verify full coverage
            assert_eq!(ranges[0].0, 0, "First shard should start at 0");
            assert_eq!(
                ranges.last().unwrap().1,
                8192,
                "Last shard should end at {}",
                8192
            );
        }
    }

    #[test]
    fn test_allocate_weighted_column_ranges_biases_capacity() {
        let ranges = allocate_weighted_column_ranges(
            8192,
            &[
                WorkerCapacityProfile {
                    tier: Tier::Tier4,
                    contributed_memory: 16 * 1024 * 1024 * 1024,
                    fallback_memory_mb: 16384,
                },
                WorkerCapacityProfile {
                    tier: Tier::Tier0,
                    contributed_memory: 2 * 1024 * 1024 * 1024,
                    fallback_memory_mb: 2048,
                },
            ],
            &InferenceSchedulingPolicy::default(),
        );

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges[1].1, 8192);
        assert_eq!(ranges[0].1, ranges[1].0);

        let first_width = ranges[0].1 - ranges[0].0;
        let second_width = ranges[1].1 - ranges[1].0;
        assert!(first_width > second_width);
    }

    #[test]
    fn test_allocate_weighted_column_ranges_covers_full_tensor_parallel_space() {
        let ranges = allocate_weighted_column_ranges(
            8192,
            &[
                WorkerCapacityProfile {
                    tier: Tier::Tier3,
                    contributed_memory: 12 * 1024 * 1024 * 1024,
                    fallback_memory_mb: 12288,
                },
                WorkerCapacityProfile {
                    tier: Tier::Tier2,
                    contributed_memory: 8 * 1024 * 1024 * 1024,
                    fallback_memory_mb: 8192,
                },
                WorkerCapacityProfile {
                    tier: Tier::Tier1,
                    contributed_memory: 4 * 1024 * 1024 * 1024,
                    fallback_memory_mb: 4096,
                },
            ],
            &InferenceSchedulingPolicy::default(),
        );

        assert_eq!(ranges.first().copied(), Some((0, ranges[0].1)));
        assert_eq!(ranges.last().map(|range| range.1), Some(8192));

        for window in ranges.windows(2) {
            assert_eq!(window[0].1, window[1].0);
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
            model_id: "test-model".to_string(),
            contributed_memory: 8_000_000_000,
            ring_position: None,
            status: "online".to_string(),
        };

        let position = manager.add_worker(worker).unwrap();

        assert_eq!(position.position, 0);
        assert_eq!(position.left_neighbor, device_id);
        assert_eq!(position.right_neighbor, device_id);
        assert_eq!(position.shard.column_range, (0, 8192));
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
                model_id: "test-model".to_string(),
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
            model_id: "test-model".to_string(),
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
            model_id: "test-model".to_string(),
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
                model_id: "test-model".to_string(),
                contributed_memory: 8_000_000_000,
                ring_position: None,
                status: "online".to_string(),
            };
            manager.add_worker(worker).unwrap();
        }

        // Remove middle worker
        manager
            .handle_worker_failure("device-1".to_string())
            .unwrap();

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
            model_id: "test-model".to_string(),
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
                model_id: "test-model".to_string(),
                contributed_memory: 8_000_000_000,
                ring_position: None,
                status: "online".to_string(),
            };
            manager.add_worker(worker).unwrap();
        }

        // Remove all workers
        manager
            .handle_worker_failure("device-0".to_string())
            .unwrap();
        manager
            .handle_worker_failure("device-1".to_string())
            .unwrap();

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
                model_id: "test-model".to_string(),
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
            let left_pos = if worker.position == 0 {
                9
            } else {
                worker.position - 1
            };
            let right_pos = (worker.position + 1) % 10;

            let expected_left = format!("device-{}", left_pos);
            let expected_right = format!("device-{}", right_pos);

            assert_eq!(worker.left_neighbor, expected_left);
            assert_eq!(worker.right_neighbor, expected_right);
        }

        // Verify shard coverage
        let mut all_ranges: Vec<(u32, u32)> = topology
            .workers
            .iter()
            .map(|w| w.shard.column_range)
            .collect();
        all_ranges.sort_by_key(|r| r.0);

        // First should start at 0, last should end at 8192
        assert_eq!(all_ranges[0].0, 0);
        assert_eq!(all_ranges[9].1, 8192);

        // No gaps between ranges
        for i in 0..9 {
            assert_eq!(all_ranges[i].1, all_ranges[i + 1].0);
        }
    }

    #[test]
    fn test_get_topology_generates_peer_punch_plans_for_relayed_workers() {
        let (manager, db) = create_test_ring_manager();
        let network_id = "test-network";

        for i in 0..3 {
            let device_id = format!("device-{}", i);
            register_test_device(&db, &device_id, network_id);
            device_service::update_heartbeat(
                &db,
                device_id.clone(),
                DeviceConnectivityState {
                    active_path: ConnectivityPath::Relayed,
                    active_endpoint: Some("/dns4/relay.mesh.example/tcp/4001".to_string()),
                    status: ConnectivityStatus::Connected,
                },
                vec![format!(
                    "/ip4/10.0.0.{}/tcp/4100/p2p/test-peer-manager-{}",
                    i + 2,
                    device_id
                )],
                vec![DirectPeerCandidate {
                    endpoint: format!(
                        "/ip4/10.0.0.{}/tcp/4100/p2p/test-peer-manager-{}",
                        i + 2,
                        device_id
                    ),
                    transport: crate::connectivity::DirectCandidateTransport::Tcp,
                    scope: crate::connectivity::DirectCandidateScope::Private,
                    source: crate::connectivity::DirectCandidateSource::LocalListen,
                    priority: 21,
                    last_updated_ms: 1_700_000_000_000,
                }],
            )
            .unwrap();

            manager
                .add_worker(Worker {
                    device_id,
                    network_id: network_id.to_string(),
                    model_id: "test-model".to_string(),
                    contributed_memory: 8_000_000_000,
                    ring_position: None,
                    status: "online".to_string(),
                })
                .unwrap();
        }

        let topology = manager.get_topology(network_id).unwrap();

        assert_eq!(topology.workers.len(), 3);
        assert_eq!(topology.peer_punch_plans.len(), 6);
        assert!(topology
            .peer_punch_plans
            .iter()
            .all(|plan| matches!(plan.reason, PunchPathReason::RelayPath)));
        assert!(topology
            .peer_punch_plans
            .iter()
            .all(|plan| plan.relay_rendezvous_required));
    }
}
