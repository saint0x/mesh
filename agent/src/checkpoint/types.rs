//! Checkpoint types and structures
//!
//! This module defines the types used for checkpointing inference state.

use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::time::Duration;
use uuid::Uuid;

use crate::inference::kv_cache::{KVCacheBlob, KVCacheEncoding, KVCacheSnapshot, KVSequenceState};
use crate::inference::ExecutionPhase;

/// Configuration for checkpoint management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Directory for storing checkpoints
    pub checkpoint_dir: std::path::PathBuf,

    /// Maximum number of checkpoints to keep per job
    pub max_checkpoints_per_job: u32,

    /// Maximum total checkpoint storage in bytes
    pub max_total_storage_bytes: u64,

    /// Checkpoint retention period
    pub retention_period: Duration,

    /// Whether to replicate checkpoints to peers
    pub replicate_to_peers: bool,

    /// Number of peers to replicate to (if replication enabled)
    pub replication_factor: u32,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: dirs::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(".meshnet")
                .join("checkpoints"),
            max_checkpoints_per_job: 5,
            max_total_storage_bytes: 10 * 1024 * 1024 * 1024, // 10 GB
            retention_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            replicate_to_peers: false,
            replication_factor: 2,
        }
    }
}

/// Metadata about a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Unique checkpoint identifier
    pub checkpoint_id: Uuid,

    /// Job ID this checkpoint belongs to
    pub job_id: Uuid,

    /// Token index at checkpoint time
    pub token_index: u32,

    /// Timestamp when checkpoint was created (Unix seconds)
    pub created_at: u64,

    /// Size of checkpoint data in bytes
    pub size_bytes: u64,

    /// Layer that was being processed
    pub current_layer: u32,

    /// Model ID
    pub model_id: String,

    /// Worker that created this checkpoint
    pub worker_id: String,

    /// Hash of checkpoint data for verification
    pub data_hash: String,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata
    pub fn new(
        job_id: Uuid,
        token_index: u32,
        current_layer: u32,
        model_id: String,
        worker_id: String,
        size_bytes: u64,
        data_hash: String,
    ) -> Self {
        Self {
            checkpoint_id: Uuid::new_v4(),
            job_id,
            token_index,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            size_bytes,
            current_layer,
            model_id,
            worker_id,
            data_hash,
        }
    }

    /// Get age of checkpoint
    pub fn age(&self) -> Duration {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Duration::from_secs(now.saturating_sub(self.created_at))
    }
}

/// Where the authoritative KV state currently resides.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KVCacheResidency {
    /// Inline with a durable checkpoint on local disk.
    CheckpointStore,
    /// Bundled for transfer between workers.
    TransferBundle,
    /// Described by a remote reference and fetched on demand.
    RemoteReference,
}

/// Future-facing payload location for KV handoff.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KVPayloadRef {
    /// Full inline payload materialized with the checkpoint bytes.
    Inline {
        encoding: KVCacheEncoding,
        size_bytes: u64,
    },
    /// Externalized payload to be fetched through another transport later.
    External {
        encoding: KVCacheEncoding,
        location: String,
        size_bytes: Option<u64>,
    },
}

/// Metadata describing a KV handoff without forcing local materialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KVCacheHandoff {
    /// Current durable or transferable residency of the KV state.
    pub residency: KVCacheResidency,
    /// Worker that should be treated as the owner after restore/import.
    pub owner_worker_id: String,
    /// Worker that originally exported the current snapshot lineage.
    pub source_worker_id: String,
    /// Sequence coverage represented by this handoff.
    pub sequence: KVSequenceState,
    /// Payload reference for the cache state.
    pub payload_ref: KVPayloadRef,
    /// Inline payload materialization when available locally.
    pub payload: Option<KVCacheBlob>,
}

impl KVCacheHandoff {
    pub fn checkpoint_resident(snapshot: KVCacheSnapshot, owner_worker_id: String) -> Self {
        let payload_ref = KVPayloadRef::Inline {
            encoding: snapshot.blob.encoding,
            size_bytes: snapshot.blob.size_bytes(),
        };
        Self {
            residency: KVCacheResidency::CheckpointStore,
            owner_worker_id: owner_worker_id.clone(),
            source_worker_id: owner_worker_id,
            sequence: snapshot.sequence,
            payload_ref,
            payload: Some(snapshot.blob),
        }
    }

    pub fn transfer_bundle(&self) -> Self {
        let mut cloned = self.clone();
        cloned.residency = KVCacheResidency::TransferBundle;
        cloned
    }

    pub fn imported_for(&self, owner_worker_id: String) -> Self {
        let mut cloned = self.clone();
        cloned.residency = KVCacheResidency::CheckpointStore;
        cloned.owner_worker_id = owner_worker_id;
        cloned
    }

    pub fn validate(&self) -> Result<(), String> {
        self.sequence.validate().map_err(|e| e.to_string())?;
        match (&self.payload_ref, &self.payload) {
            (
                KVPayloadRef::Inline {
                    encoding,
                    size_bytes,
                },
                Some(blob),
            ) => {
                if blob.encoding != *encoding {
                    return Err(format!(
                        "KV handoff encoding mismatch: payload_ref {:?} vs payload {:?}",
                        encoding, blob.encoding
                    ));
                }
                if blob.size_bytes() != *size_bytes {
                    return Err(format!(
                        "KV handoff payload size mismatch: payload_ref {} vs payload {}",
                        size_bytes,
                        blob.size_bytes()
                    ));
                }
                let snapshot = KVCacheSnapshot {
                    sequence: self.sequence,
                    blob: blob.clone(),
                };
                snapshot.validate().map_err(|e| e.to_string())?;
            }
            (KVPayloadRef::Inline { .. }, None) => {
                return Err("KV handoff is missing inline payload bytes".to_string());
            }
            (KVPayloadRef::External { .. }, _) => {}
        }
        Ok(())
    }

    pub fn materialize_snapshot(&self) -> Result<Option<KVCacheSnapshot>, String> {
        self.validate()?;
        match (&self.payload_ref, &self.payload) {
            (KVPayloadRef::Inline { .. }, Some(blob)) => Ok(Some(KVCacheSnapshot {
                sequence: self.sequence,
                blob: blob.clone(),
            })),
            (KVPayloadRef::External { .. }, _) => Ok(None),
            (KVPayloadRef::Inline { .. }, None) => {
                Err("KV handoff is missing inline payload bytes".to_string())
            }
        }
    }
}

/// A checkpoint containing inference state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint metadata
    pub metadata: CheckpointMetadata,

    /// Inference request (original)
    pub request: CheckpointedRequest,

    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,

    /// Generation config (for resumption)
    pub config: CheckpointedConfig,

    /// KV cache handoff for inference recovery.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_handoff: Option<KVCacheHandoff>,

    /// Legacy inline KV payload for backwards-compatible deserialization only.
    #[serde(default, skip_serializing)]
    pub kv_cache_state: Option<Vec<u8>>,

    /// Legacy sequence position for backwards-compatible deserialization only.
    #[serde(default, skip_serializing)]
    pub sequence_position: Option<u32>,

    /// RNG state for reproducibility
    pub rng_state: Option<Vec<u8>>,
}

/// Minimal request info needed for checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointedRequest {
    /// Job ID
    pub job_id: Uuid,

    /// Session ID within the serving engine plan.
    pub session_id: Uuid,

    /// Network ID
    pub network_id: String,

    /// Model ID
    pub model_id: String,

    /// Original prompt tokens
    pub prompt_tokens: Vec<u32>,

    /// Executor ID
    pub executor_id: String,

    /// Execution phase active when the snapshot was taken.
    pub phase: ExecutionPhase,
}

/// Generation config stored in checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointedConfig {
    /// Max tokens to generate
    pub max_tokens: u32,

    /// Temperature
    pub temperature: f32,

    /// Top-p
    pub top_p: f32,

    /// Stop sequences
    pub stop_sequences: Vec<String>,

    /// Whether the request streams tokens
    pub stream: bool,

    /// Checkpoint interval
    pub checkpoint_interval: u32,

    /// Total model layers required to resume execution
    pub total_layers: u32,
}

impl Checkpoint {
    /// Calculate checkpoint data hash using BLAKE3
    pub fn compute_hash(&self) -> String {
        let bytes = self.to_cbor().unwrap_or_default();
        hex::encode({
            let mut hasher = sha2::Sha256::new();
            hasher.update(bytes);
            hasher.finalize()
        })
    }

    /// Serialize checkpoint to CBOR bytes
    pub fn to_cbor(&self) -> Result<Vec<u8>, ciborium::ser::Error<std::io::Error>> {
        let mut bytes = Vec::new();
        ciborium::into_writer(self, &mut bytes)?;
        Ok(bytes)
    }

    /// Deserialize checkpoint from CBOR bytes
    pub fn from_cbor(bytes: &[u8]) -> Result<Self, ciborium::de::Error<std::io::Error>> {
        ciborium::from_reader(bytes)
    }

    pub fn kv_handoff(&self) -> Result<Option<KVCacheHandoff>, String> {
        if let Some(kv_handoff) = &self.kv_handoff {
            kv_handoff.validate()?;
            return Ok(Some(kv_handoff.clone()));
        }

        match (&self.kv_cache_state, self.sequence_position) {
            (Some(bytes), Some(sequence_position)) => {
                let blob = KVCacheBlob {
                    encoding: KVCacheEncoding::FullSnapshotCbor,
                    bytes: bytes.clone(),
                };
                let sequence = KVSequenceState::new(sequence_position, sequence_position)
                    .map_err(|err| err.to_string())?;
                let handoff = KVCacheHandoff {
                    residency: KVCacheResidency::CheckpointStore,
                    owner_worker_id: self.metadata.worker_id.clone(),
                    source_worker_id: self.metadata.worker_id.clone(),
                    sequence,
                    payload_ref: KVPayloadRef::Inline {
                        encoding: KVCacheEncoding::FullSnapshotCbor,
                        size_bytes: bytes.len() as u64,
                    },
                    payload: Some(blob),
                };
                handoff.validate()?;
                Ok(Some(handoff))
            }
            (None, None) => Ok(None),
            (Some(_), None) => Err(
                "Checkpoint contains legacy KV payload bytes without a sequence position"
                    .to_string(),
            ),
            (None, Some(_)) => Err(
                "Checkpoint contains legacy sequence position without KV payload bytes".to_string(),
            ),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if let Some(kv_handoff) = self.kv_handoff()? {
            kv_handoff.validate()?;
        }
        Ok(())
    }

    /// Get file path for this checkpoint
    pub fn file_path(&self, base_dir: &std::path::Path) -> std::path::PathBuf {
        base_dir
            .join(self.metadata.job_id.to_string())
            .join(format!("{}.ckpt", self.metadata.checkpoint_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::kv_cache::{KVCache, KVCacheSnapshot};
    use crate::inference::tensor_ops::Tensor2D;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.max_checkpoints_per_job, 5);
        assert!(!config.replicate_to_peers);
    }

    #[test]
    fn test_checkpoint_metadata_new() {
        let job_id = Uuid::new_v4();
        let metadata = CheckpointMetadata::new(
            job_id,
            50,
            35,
            "llama-70b".to_string(),
            "worker-1".to_string(),
            1024,
            "abc123".to_string(),
        );

        assert_eq!(metadata.job_id, job_id);
        assert_eq!(metadata.token_index, 50);
        assert_eq!(metadata.current_layer, 35);
    }

    #[test]
    fn test_checkpoint_metadata_age() {
        let job_id = Uuid::new_v4();
        let metadata = CheckpointMetadata::new(
            job_id,
            50,
            35,
            "model".to_string(),
            "worker".to_string(),
            0,
            "hash".to_string(),
        );

        // Age should be very small (just created)
        assert!(metadata.age().as_secs() < 1);
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = Checkpoint {
            metadata: CheckpointMetadata::new(
                Uuid::new_v4(),
                10,
                5,
                "model".to_string(),
                "worker".to_string(),
                0,
                "hash".to_string(),
            ),
            request: CheckpointedRequest {
                job_id: Uuid::new_v4(),
                session_id: Uuid::new_v4(),
                network_id: "net-1".to_string(),
                model_id: "llama-70b".to_string(),
                prompt_tokens: vec![1, 2, 3, 4, 5],
                executor_id: "exec-1".to_string(),
                phase: ExecutionPhase::Decode,
            },
            generated_tokens: vec![100, 101, 102],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                stream: false,
                checkpoint_interval: 50,
                total_layers: 70,
            },
            kv_handoff: None,
            kv_cache_state: None,
            sequence_position: None,
            rng_state: None,
        };

        // Serialize and deserialize
        let bytes = checkpoint.to_cbor().unwrap();
        let restored: Checkpoint = Checkpoint::from_cbor(&bytes).unwrap();

        assert_eq!(restored.generated_tokens, checkpoint.generated_tokens);
        assert_eq!(
            restored.request.prompt_tokens,
            checkpoint.request.prompt_tokens
        );
    }

    #[test]
    fn test_checkpoint_hash() {
        let checkpoint1 = Checkpoint {
            metadata: CheckpointMetadata::new(
                Uuid::new_v4(),
                10,
                5,
                "model".to_string(),
                "worker".to_string(),
                0,
                "".to_string(),
            ),
            request: CheckpointedRequest {
                job_id: Uuid::new_v4(),
                session_id: Uuid::new_v4(),
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt_tokens: vec![1, 2, 3],
                executor_id: "exec".to_string(),
                phase: ExecutionPhase::Decode,
            },
            generated_tokens: vec![100, 101],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                stream: false,
                checkpoint_interval: 50,
                total_layers: 70,
            },
            kv_handoff: None,
            kv_cache_state: None,
            sequence_position: None,
            rng_state: None,
        };

        let hash1 = checkpoint1.compute_hash();
        assert!(!hash1.is_empty());

        // Same checkpoint should have same hash
        let hash2 = checkpoint1.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_checkpoint_file_path() {
        let job_id = Uuid::new_v4();
        let checkpoint_id = Uuid::new_v4();

        let checkpoint = Checkpoint {
            metadata: CheckpointMetadata {
                checkpoint_id,
                job_id,
                token_index: 10,
                created_at: 0,
                size_bytes: 0,
                current_layer: 0,
                model_id: "model".to_string(),
                worker_id: "worker".to_string(),
                data_hash: "hash".to_string(),
            },
            request: CheckpointedRequest {
                job_id,
                session_id: Uuid::new_v4(),
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt_tokens: vec![],
                executor_id: "exec".to_string(),
                phase: ExecutionPhase::Decode,
            },
            generated_tokens: vec![],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                stream: false,
                checkpoint_interval: 50,
                total_layers: 70,
            },
            kv_handoff: None,
            kv_cache_state: None,
            sequence_position: None,
            rng_state: None,
        };

        let path = checkpoint.file_path(std::path::Path::new("/tmp/checkpoints"));
        assert!(path.to_string_lossy().contains(&job_id.to_string()));
        assert!(path.to_string_lossy().ends_with(".ckpt"));
    }

    #[test]
    fn test_checkpoint_exposes_handoff_sequence_state() {
        let mut cache = KVCache::new(crate::inference::kv_cache::KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 8,
        });
        cache
            .update_layer(
                0,
                Tensor2D::new(vec![1.0; 8], 1, 8).unwrap(),
                Tensor2D::new(vec![2.0; 8], 1, 8).unwrap(),
            )
            .unwrap();
        let snapshot = KVCacheSnapshot::from_cache(&cache, 1).unwrap();

        let checkpoint = Checkpoint {
            metadata: CheckpointMetadata::new(
                Uuid::new_v4(),
                10,
                5,
                "model".to_string(),
                "worker".to_string(),
                0,
                "".to_string(),
            ),
            request: CheckpointedRequest {
                job_id: Uuid::new_v4(),
                session_id: Uuid::new_v4(),
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt_tokens: vec![1, 2, 3],
                executor_id: "exec".to_string(),
                phase: ExecutionPhase::Decode,
            },
            generated_tokens: vec![100],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                stream: false,
                checkpoint_interval: 50,
                total_layers: 70,
            },
            kv_handoff: Some(KVCacheHandoff::checkpoint_resident(
                snapshot,
                "worker".to_string(),
            )),
            kv_cache_state: None,
            sequence_position: None,
            rng_state: None,
        };

        let handoff = checkpoint.kv_handoff().unwrap().unwrap();
        assert_eq!(handoff.sequence.next_position, 1);
        assert_eq!(handoff.sequence.cached_tokens, 1);
    }

    #[test]
    fn test_checkpoint_upgrades_legacy_kv_fields() {
        let cache = KVCache::new(crate::inference::kv_cache::KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 8,
        });
        let bytes = cache.to_bytes().unwrap();
        let checkpoint = Checkpoint {
            metadata: CheckpointMetadata::new(
                Uuid::new_v4(),
                10,
                5,
                "model".to_string(),
                "worker".to_string(),
                0,
                "".to_string(),
            ),
            request: CheckpointedRequest {
                job_id: Uuid::new_v4(),
                session_id: Uuid::new_v4(),
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt_tokens: vec![1, 2, 3],
                executor_id: "exec".to_string(),
                phase: ExecutionPhase::Decode,
            },
            generated_tokens: vec![],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                stream: false,
                checkpoint_interval: 50,
                total_layers: 70,
            },
            kv_handoff: None,
            kv_cache_state: Some(bytes),
            sequence_position: Some(0),
            rng_state: None,
        };

        let handoff = checkpoint.kv_handoff().unwrap().unwrap();
        assert_eq!(handoff.owner_worker_id, "worker");
        assert_eq!(handoff.sequence.next_position, 0);
        assert_eq!(handoff.sequence.cached_tokens, 0);
    }
}
