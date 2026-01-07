//! Checkpoint types and structures
//!
//! This module defines the types used for checkpointing inference state.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::Duration;

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

    /// KV cache state (placeholder - will be tensor data)
    /// In production, this would be serialized tensor data
    pub kv_cache_state: Option<Vec<u8>>,

    /// RNG state for reproducibility
    pub rng_state: Option<Vec<u8>>,
}

/// Minimal request info needed for checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointedRequest {
    /// Job ID
    pub job_id: Uuid,

    /// Network ID
    pub network_id: String,

    /// Model ID
    pub model_id: String,

    /// Original prompt tokens
    pub prompt_tokens: Vec<u32>,

    /// Executor ID
    pub executor_id: String,
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

    /// Checkpoint interval
    pub checkpoint_interval: u32,
}

impl Checkpoint {
    /// Calculate checkpoint data hash using BLAKE3
    pub fn compute_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple hash for now - production would use BLAKE3
        let mut hasher = DefaultHasher::new();
        self.metadata.job_id.hash(&mut hasher);
        self.metadata.token_index.hash(&mut hasher);
        self.generated_tokens.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
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
                network_id: "net-1".to_string(),
                model_id: "llama-70b".to_string(),
                prompt_tokens: vec![1, 2, 3, 4, 5],
                executor_id: "exec-1".to_string(),
            },
            generated_tokens: vec![100, 101, 102],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                checkpoint_interval: 50,
            },
            kv_cache_state: None,
            rng_state: None,
        };

        // Serialize and deserialize
        let bytes = checkpoint.to_cbor().unwrap();
        let restored: Checkpoint = Checkpoint::from_cbor(&bytes).unwrap();

        assert_eq!(restored.generated_tokens, checkpoint.generated_tokens);
        assert_eq!(restored.request.prompt_tokens, checkpoint.request.prompt_tokens);
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
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt_tokens: vec![1, 2, 3],
                executor_id: "exec".to_string(),
            },
            generated_tokens: vec![100, 101],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                checkpoint_interval: 50,
            },
            kv_cache_state: None,
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
                network_id: "net".to_string(),
                model_id: "model".to_string(),
                prompt_tokens: vec![],
                executor_id: "exec".to_string(),
            },
            generated_tokens: vec![],
            config: CheckpointedConfig {
                max_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                stop_sequences: vec![],
                checkpoint_interval: 50,
            },
            kv_cache_state: None,
            rng_state: None,
        };

        let path = checkpoint.file_path(std::path::Path::new("/tmp/checkpoints"));
        assert!(path.to_string_lossy().contains(&job_id.to_string()));
        assert!(path.to_string_lossy().ends_with(".ckpt"));
    }
}
