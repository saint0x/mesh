//! Checkpoint manager for saving and loading inference state
//!
//! This module provides the CheckpointManager which handles:
//! - Saving checkpoints to disk
//! - Loading checkpoints for recovery
//! - Managing checkpoint storage (cleanup, retention)
//! - Optional replication to peer workers

use crate::errors::{AgentError, Result};
use crate::inference::job::InferenceJob;
use crate::inference::kv_cache::{KVCache, KVCacheSnapshot};
use std::path::Path;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::types::{
    Checkpoint, CheckpointConfig, CheckpointMetadata, CheckpointedConfig, CheckpointedRequest,
    KVCacheHandoff,
};

/// Manages checkpoint creation, storage, and retrieval
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,

    /// Worker ID for this instance
    worker_id: String,

    /// Cache of recent checkpoint metadata
    metadata_cache: RwLock<Vec<CheckpointMetadata>>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig, worker_id: String) -> Result<Self> {
        // Ensure checkpoint directory exists
        std::fs::create_dir_all(&config.checkpoint_dir)?;

        Ok(Self {
            config,
            worker_id,
            metadata_cache: RwLock::new(Vec::new()),
        })
    }

    /// Create with default configuration
    pub fn with_defaults(worker_id: String) -> Result<Self> {
        Self::new(CheckpointConfig::default(), worker_id)
    }

    /// Get checkpoint directory
    pub fn checkpoint_dir(&self) -> &Path {
        &self.config.checkpoint_dir
    }

    /// Save a checkpoint for an inference job
    pub async fn save_checkpoint(
        &self,
        job: &InferenceJob,
        kv_snapshot: Option<&KVCacheSnapshot>,
    ) -> Result<CheckpointMetadata> {
        debug!(
            job_id = %job.request.job_id,
            token_index = job.current_token_idx,
            "Saving checkpoint"
        );

        // Create checkpoint from job state
        let checkpoint = self.create_checkpoint(job, kv_snapshot)?;

        // Serialize to CBOR
        let data = checkpoint
            .to_cbor()
            .map_err(|e| AgentError::Config(format!("Failed to serialize checkpoint: {}", e)))?;

        // Create job directory if needed
        let job_dir = self
            .config
            .checkpoint_dir
            .join(job.request.job_id.to_string());
        fs::create_dir_all(&job_dir).await?;

        // Write checkpoint file
        let file_path = checkpoint.file_path(&self.config.checkpoint_dir);
        fs::write(&file_path, &data).await?;

        // Update metadata with actual size
        let mut metadata = checkpoint.metadata.clone();
        metadata.size_bytes = data.len() as u64;
        metadata.data_hash = checkpoint.compute_hash();

        // Update cache
        {
            let mut cache = self.metadata_cache.write().await;
            cache.push(metadata.clone());
        }

        // Cleanup old checkpoints for this job
        self.cleanup_job_checkpoints(job.request.job_id).await?;

        info!(
            job_id = %job.request.job_id,
            checkpoint_id = %metadata.checkpoint_id,
            token_index = metadata.token_index,
            size_bytes = metadata.size_bytes,
            "Checkpoint saved"
        );

        Ok(metadata)
    }

    /// Load the latest checkpoint for a job
    pub async fn load_checkpoint(&self, job_id: Uuid) -> Result<Option<InferenceJob>> {
        if let Some(checkpoint) = self.load_latest_checkpoint(job_id).await? {
            let job = self.restore_job(checkpoint)?;
            info!(
                job_id = %job_id,
                token_index = job.current_token_idx,
                "Checkpoint loaded"
            );
            return Ok(Some(job));
        }

        Ok(None)
    }

    pub async fn load_checkpoint_kv_cache(&self, job_id: Uuid) -> Result<Option<KVCache>> {
        Ok(self
            .load_checkpoint_kv_handoff(job_id)
            .await?
            .map(|handoff| handoff.materialize_snapshot())
            .transpose()
            .map_err(AgentError::Execution)?
            .flatten()
            .map(|snapshot| snapshot.decode_cache())
            .transpose()?)
    }

    pub async fn load_checkpoint_sequence_position(&self, job_id: Uuid) -> Result<Option<u32>> {
        Ok(self
            .load_checkpoint_kv_handoff(job_id)
            .await?
            .map(|handoff| handoff.sequence.next_position))
    }

    pub async fn load_checkpoint_kv_handoff(&self, job_id: Uuid) -> Result<Option<KVCacheHandoff>> {
        Ok(self
            .load_latest_checkpoint(job_id)
            .await?
            .map(|checkpoint| {
                checkpoint.kv_handoff().map_err(|e| {
                    AgentError::Execution(format!(
                        "Invalid KV handoff in checkpoint {}: {}",
                        checkpoint.metadata.checkpoint_id, e
                    ))
                })
            })
            .transpose()?
            .flatten())
    }

    pub async fn export_latest_checkpoint_bytes(&self, job_id: Uuid) -> Result<Option<Vec<u8>>> {
        let Some(mut checkpoint) = self.load_latest_checkpoint(job_id).await? else {
            return Ok(None);
        };
        checkpoint.kv_handoff = checkpoint
            .kv_handoff()
            .map_err(AgentError::Execution)?
            .map(|handoff| handoff.transfer_bundle());
        checkpoint.kv_cache_state = None;
        checkpoint.sequence_position = None;
        let bytes = checkpoint
            .to_cbor()
            .map_err(|e| AgentError::Config(format!("Failed to serialize checkpoint: {}", e)))?;
        Ok(Some(bytes))
    }

    pub async fn import_checkpoint_bytes(&self, bytes: &[u8]) -> Result<CheckpointMetadata> {
        let mut checkpoint = Checkpoint::from_cbor(bytes)
            .map_err(|e| AgentError::Config(format!("Failed to deserialize checkpoint: {}", e)))?;
        checkpoint
            .validate()
            .map_err(|e| AgentError::Execution(format!("Invalid imported checkpoint: {}", e)))?;
        checkpoint.kv_handoff = checkpoint
            .kv_handoff()
            .map_err(AgentError::Execution)?
            .map(|handoff| handoff.imported_for(self.worker_id.clone()));
        checkpoint.kv_cache_state = None;
        checkpoint.sequence_position = None;
        let bytes = checkpoint
            .to_cbor()
            .map_err(|e| AgentError::Config(format!("Failed to serialize checkpoint: {}", e)))?;
        let job_dir = self
            .config
            .checkpoint_dir
            .join(checkpoint.metadata.job_id.to_string());
        fs::create_dir_all(&job_dir).await?;
        let file_path = checkpoint.file_path(&self.config.checkpoint_dir);
        fs::write(&file_path, &bytes).await?;

        let mut metadata = checkpoint.metadata.clone();
        metadata.size_bytes = bytes.len() as u64;
        metadata.data_hash = checkpoint.compute_hash();

        {
            let mut cache = self.metadata_cache.write().await;
            cache.push(metadata.clone());
        }

        Ok(metadata)
    }

    /// List all checkpoints for a job
    pub async fn list_checkpoints(&self, job_id: Uuid) -> Result<Vec<CheckpointMetadata>> {
        let job_dir = self.config.checkpoint_dir.join(job_id.to_string());

        if !job_dir.exists() {
            return Ok(Vec::new());
        }

        let mut metadata_list = Vec::new();

        for entry in std::fs::read_dir(&job_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "ckpt").unwrap_or(false) {
                match self.load_checkpoint_file(&path).await {
                    Ok(checkpoint) => {
                        metadata_list.push(checkpoint.metadata);
                    }
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "Failed to read checkpoint");
                    }
                }
            }
        }

        // Sort by token index (descending)
        metadata_list.sort_by(|a, b| b.token_index.cmp(&a.token_index));

        Ok(metadata_list)
    }

    /// Delete a specific checkpoint
    pub async fn delete_checkpoint(&self, job_id: Uuid, checkpoint_id: Uuid) -> Result<()> {
        let file_path = self
            .config
            .checkpoint_dir
            .join(job_id.to_string())
            .join(format!("{}.ckpt", checkpoint_id));

        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
            info!(job_id = %job_id, checkpoint_id = %checkpoint_id, "Checkpoint deleted");
        }

        Ok(())
    }

    /// Delete all checkpoints for a job
    pub async fn delete_job_checkpoints(&self, job_id: Uuid) -> Result<()> {
        let job_dir = self.config.checkpoint_dir.join(job_id.to_string());

        if job_dir.exists() {
            std::fs::remove_dir_all(&job_dir)?;
            info!(job_id = %job_id, "All job checkpoints deleted");
        }

        Ok(())
    }

    /// Cleanup old checkpoints across all jobs
    pub async fn cleanup_old_checkpoints(&self) -> Result<u32> {
        let mut deleted = 0;

        // Iterate job directories
        if let Ok(entries) = std::fs::read_dir(&self.config.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Check checkpoint age
                    if let Ok(job_entries) = std::fs::read_dir(&path) {
                        for job_entry in job_entries.flatten() {
                            let ckpt_path = job_entry.path();
                            if let Ok(metadata) = std::fs::metadata(&ckpt_path) {
                                if let Ok(modified) = metadata.modified() {
                                    let age = std::time::SystemTime::now()
                                        .duration_since(modified)
                                        .unwrap_or_default();

                                    if age > self.config.retention_period {
                                        if std::fs::remove_file(&ckpt_path).is_ok() {
                                            deleted += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Remove empty job directories
                    if std::fs::read_dir(&path)
                        .map(|d| d.count() == 0)
                        .unwrap_or(false)
                    {
                        let _ = std::fs::remove_dir(&path);
                    }
                }
            }
        }

        if deleted > 0 {
            info!(deleted = deleted, "Cleaned up old checkpoints");
        }

        Ok(deleted)
    }

    /// Get total checkpoint storage usage
    pub async fn storage_usage(&self) -> Result<u64> {
        fn dir_size(path: &Path) -> std::io::Result<u64> {
            let mut size = 0;
            if path.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    let entry = entry?;
                    let path = entry.path();
                    if path.is_dir() {
                        size += dir_size(&path)?;
                    } else {
                        size += std::fs::metadata(&path)?.len();
                    }
                }
            }
            Ok(size)
        }

        Ok(dir_size(&self.config.checkpoint_dir)?)
    }

    // === Private Methods ===

    /// Create a checkpoint from job state
    fn create_checkpoint(
        &self,
        job: &InferenceJob,
        kv_snapshot: Option<&KVCacheSnapshot>,
    ) -> Result<Checkpoint> {
        let metadata = CheckpointMetadata::new(
            job.request.job_id,
            job.current_token_idx,
            job.current_layer,
            job.request.model_id.clone(),
            self.worker_id.clone(),
            0,             // Will be updated after serialization
            String::new(), // Will be computed after creation
        );

        let request = CheckpointedRequest {
            job_id: job.request.job_id,
            session_id: job.request.session_id,
            network_id: job.request.network_id.clone(),
            model_id: job.request.model_id.clone(),
            prompt_tokens: job.request.prompt_tokens.clone(),
            executor_id: job.request.executor_id.clone(),
            phase: job.request.phase,
        };

        let config = CheckpointedConfig {
            max_tokens: job.request.config.max_tokens,
            temperature: job.request.config.temperature,
            top_p: job.request.config.top_p,
            stop_sequences: job.request.config.stop_sequences.clone(),
            stream: job.request.config.stream,
            checkpoint_interval: job.request.config.checkpoint_interval,
            total_layers: job.total_layers,
        };

        let rng_state = Some(job.request.job_id.as_u128().to_le_bytes().to_vec());
        let kv_handoff = kv_snapshot
            .cloned()
            .map(|snapshot| KVCacheHandoff::checkpoint_resident(snapshot, self.worker_id.clone()));

        let checkpoint = Checkpoint {
            metadata,
            request,
            generated_tokens: job.generated_tokens.clone(),
            config,
            kv_handoff,
            kv_cache_state: None,
            sequence_position: None,
            rng_state,
        };

        Ok(checkpoint)
    }

    /// Load a checkpoint from file
    async fn load_checkpoint_file(&self, path: &Path) -> Result<Checkpoint> {
        let data = fs::read(path).await?;
        let checkpoint = Checkpoint::from_cbor(&data)
            .map_err(|e| AgentError::Config(format!("Failed to deserialize checkpoint: {}", e)))?;
        checkpoint.validate().map_err(|e| {
            AgentError::Execution(format!("Invalid checkpoint {}: {}", path.display(), e))
        })?;
        Ok(checkpoint)
    }

    async fn load_latest_checkpoint(&self, job_id: Uuid) -> Result<Option<Checkpoint>> {
        let job_dir = self.config.checkpoint_dir.join(job_id.to_string());
        if !job_dir.exists() {
            debug!(job_id = %job_id, "No checkpoint directory found");
            return Ok(None);
        }

        let mut checkpoints = Vec::new();
        for entry in std::fs::read_dir(&job_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "ckpt").unwrap_or(false) {
                checkpoints.push(path);
            }
        }

        if checkpoints.is_empty() {
            debug!(job_id = %job_id, "No checkpoints found");
            return Ok(None);
        }

        let mut latest: Option<Checkpoint> = None;
        let mut latest_token_idx = 0;
        for path in checkpoints {
            match self.load_checkpoint_file(&path).await {
                Ok(checkpoint) => {
                    if checkpoint.metadata.token_index >= latest_token_idx {
                        latest_token_idx = checkpoint.metadata.token_index;
                        latest = Some(checkpoint);
                    }
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to load checkpoint");
                }
            }
        }

        Ok(latest)
    }

    /// Restore an InferenceJob from checkpoint
    fn restore_job(&self, checkpoint: Checkpoint) -> Result<InferenceJob> {
        use crate::inference::job::{GenerationConfig, InferenceRequest};

        // Reconstruct the request
        let config = GenerationConfig {
            max_tokens: checkpoint.config.max_tokens,
            temperature: checkpoint.config.temperature,
            top_p: checkpoint.config.top_p,
            stop_sequences: checkpoint.config.stop_sequences,
            stream: checkpoint.config.stream,
            checkpoint_interval: checkpoint.config.checkpoint_interval,
            progress_report_interval: GenerationConfig::default().progress_report_interval,
        };

        let request = InferenceRequest {
            job_id: checkpoint.request.job_id,
            network_id: checkpoint.request.network_id,
            model_id: checkpoint.request.model_id,
            prompt_tokens: checkpoint.request.prompt_tokens,
            config,
            session_id: checkpoint.request.session_id,
            phase: checkpoint.request.phase,
            decode_batch_targets: crate::inference::DecodeBatchTargets::default(),
            runtime_mode: crate::inference::InferenceRuntimeMode::default(),
            executor_id: checkpoint.request.executor_id,
            created_at: checkpoint.metadata.created_at,
        };

        // Create job and restore state
        let mut job = InferenceJob::new(request, checkpoint.config.total_layers);
        job.generated_tokens = checkpoint.generated_tokens;
        job.current_token_idx = checkpoint.metadata.token_index;
        job.current_layer = checkpoint.metadata.current_layer;
        job.last_checkpoint_idx = checkpoint.metadata.token_index;

        Ok(job)
    }

    /// Cleanup old checkpoints for a specific job
    async fn cleanup_job_checkpoints(&self, job_id: Uuid) -> Result<()> {
        let checkpoints = self.list_checkpoints(job_id).await?;

        // Keep only the most recent N checkpoints
        if checkpoints.len() > self.config.max_checkpoints_per_job as usize {
            let to_delete = &checkpoints[self.config.max_checkpoints_per_job as usize..];
            for metadata in to_delete {
                self.delete_checkpoint(job_id, metadata.checkpoint_id)
                    .await?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::types::{KVCacheResidency, KVPayloadRef};
    use crate::inference::job::{GenerationConfig, InferenceRequest};
    use crate::inference::kv_cache::{KVCache, KVCacheConfig, KVCacheEncoding, KVCacheSnapshot};
    use crate::inference::tensor_ops::Tensor2D;
    use tempfile::TempDir;

    fn create_test_job() -> InferenceJob {
        let request = InferenceRequest::new(
            "test-network".to_string(),
            "llama-70b".to_string(),
            vec![1, 2, 3, 4, 5],
            "executor-1".to_string(),
        )
        .with_config(GenerationConfig {
            max_tokens: 100,
            checkpoint_interval: 10,
            ..Default::default()
        });

        let mut job = InferenceJob::new(request, 70);
        // Simulate some tokens generated
        job.add_token(100);
        job.add_token(101);
        job.add_token(102);
        job
    }

    fn create_test_snapshot() -> KVCacheSnapshot {
        let mut kv_cache = KVCache::new(KVCacheConfig {
            num_layers: 2,
            num_heads: 2,
            head_dim: 2,
            max_seq_len: 16,
        });
        for layer_idx in 0..2 {
            kv_cache
                .update_layer(
                    layer_idx,
                    Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 1, 4).unwrap(),
                    Tensor2D::new(vec![5.0, 6.0, 7.0, 8.0], 1, 4).unwrap(),
                )
                .unwrap();
        }
        KVCacheSnapshot::from_cache(&kv_cache, 1).unwrap()
    }

    #[tokio::test]
    async fn test_checkpoint_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        assert!(manager.checkpoint_dir().exists());
    }

    #[tokio::test]
    async fn test_save_and_load_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        let job = create_test_job();
        let job_id = job.request.job_id;

        // Save checkpoint
        let metadata = manager.save_checkpoint(&job, None).await.unwrap();
        assert_eq!(metadata.job_id, job_id);
        assert_eq!(metadata.token_index, 3);

        // Load checkpoint
        let restored = manager.load_checkpoint(job_id).await.unwrap();
        assert!(restored.is_some());

        let restored_job = restored.unwrap();
        assert_eq!(restored_job.request.job_id, job_id);
        assert_eq!(restored_job.generated_tokens, vec![100, 101, 102]);
        assert_eq!(restored_job.current_token_idx, 3);
    }

    #[tokio::test]
    async fn test_export_and_import_checkpoint_bytes() {
        let source_dir = TempDir::new().unwrap();
        let source_manager = CheckpointManager::new(
            CheckpointConfig {
                checkpoint_dir: source_dir.path().to_path_buf(),
                ..Default::default()
            },
            "worker-source".to_string(),
        )
        .unwrap();
        let job = create_test_job();
        let job_id = job.request.job_id;

        source_manager.save_checkpoint(&job, None).await.unwrap();
        let bytes = source_manager
            .export_latest_checkpoint_bytes(job_id)
            .await
            .unwrap()
            .expect("expected exported checkpoint bytes");

        let target_dir = TempDir::new().unwrap();
        let target_manager = CheckpointManager::new(
            CheckpointConfig {
                checkpoint_dir: target_dir.path().to_path_buf(),
                ..Default::default()
            },
            "worker-target".to_string(),
        )
        .unwrap();

        let imported = target_manager
            .import_checkpoint_bytes(&bytes)
            .await
            .unwrap();
        assert_eq!(imported.job_id, job_id);

        let restored = target_manager
            .load_checkpoint(job_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(restored.request.session_id, job.request.session_id);
        assert_eq!(restored.generated_tokens, vec![100, 101, 102]);
        assert_eq!(
            target_manager
                .load_checkpoint_sequence_position(job_id)
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn test_save_and_load_checkpoint_kv_cache() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        let job = create_test_job();

        let kv_snapshot = create_test_snapshot();

        manager
            .save_checkpoint(&job, Some(&kv_snapshot))
            .await
            .unwrap();
        let restored = manager
            .load_checkpoint_kv_cache(job.request.job_id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(restored.seq_len(), 1);
        assert_eq!(
            restored.layers[0].keys.as_ref().unwrap().data,
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            manager
                .load_checkpoint_sequence_position(job.request.job_id)
                .await
                .unwrap(),
            Some(1)
        );
    }

    #[tokio::test]
    async fn test_save_and_load_segmented_checkpoint_kv_cache() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        let job = create_test_job();

        let mut kv_cache = KVCache::new(KVCacheConfig {
            num_layers: 2,
            num_heads: 2,
            head_dim: 2,
            max_seq_len: 16,
        });
        for layer_idx in 0..2 {
            kv_cache
                .update_layer(
                    layer_idx,
                    Tensor2D::new(
                        vec![
                            1.0 + layer_idx as f32,
                            2.0 + layer_idx as f32,
                            3.0 + layer_idx as f32,
                            4.0 + layer_idx as f32,
                            5.0 + layer_idx as f32,
                            6.0 + layer_idx as f32,
                            7.0 + layer_idx as f32,
                            8.0 + layer_idx as f32,
                        ],
                        2,
                        4,
                    )
                    .unwrap(),
                    Tensor2D::new(
                        vec![
                            9.0 + layer_idx as f32,
                            10.0 + layer_idx as f32,
                            11.0 + layer_idx as f32,
                            12.0 + layer_idx as f32,
                            13.0 + layer_idx as f32,
                            14.0 + layer_idx as f32,
                            15.0 + layer_idx as f32,
                            16.0 + layer_idx as f32,
                        ],
                        2,
                        4,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        kv_cache.retain_suffix(1);
        let kv_snapshot =
            KVCacheSnapshot::from_cache(&kv_cache, kv_cache.next_position() as u32).unwrap();

        manager
            .save_checkpoint(&job, Some(&kv_snapshot))
            .await
            .unwrap();
        let restored = manager
            .load_checkpoint_kv_cache(job.request.job_id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(restored.base_position(), 1);
        assert_eq!(restored.seq_len(), 1);
        assert_eq!(
            manager
                .load_checkpoint_sequence_position(job.request.job_id)
                .await
                .unwrap(),
            Some(2)
        );
    }

    #[tokio::test]
    async fn test_export_and_import_checkpoint_transitions_kv_ownership() {
        let source_dir = TempDir::new().unwrap();
        let source_manager = CheckpointManager::new(
            CheckpointConfig {
                checkpoint_dir: source_dir.path().to_path_buf(),
                ..Default::default()
            },
            "worker-source".to_string(),
        )
        .unwrap();
        let job = create_test_job();
        let snapshot = create_test_snapshot();

        source_manager
            .save_checkpoint(&job, Some(&snapshot))
            .await
            .unwrap();
        let bytes = source_manager
            .export_latest_checkpoint_bytes(job.request.job_id)
            .await
            .unwrap()
            .unwrap();

        let exported = Checkpoint::from_cbor(&bytes).unwrap();
        let exported_handoff = exported.kv_handoff().unwrap().unwrap();
        assert_eq!(exported_handoff.residency, KVCacheResidency::TransferBundle);
        assert_eq!(exported_handoff.owner_worker_id, "worker-source");
        assert_eq!(exported_handoff.source_worker_id, "worker-source");

        let target_dir = TempDir::new().unwrap();
        let target_manager = CheckpointManager::new(
            CheckpointConfig {
                checkpoint_dir: target_dir.path().to_path_buf(),
                ..Default::default()
            },
            "worker-target".to_string(),
        )
        .unwrap();
        target_manager
            .import_checkpoint_bytes(&bytes)
            .await
            .unwrap();

        let imported_handoff = target_manager
            .load_checkpoint_kv_handoff(job.request.job_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            imported_handoff.residency,
            KVCacheResidency::CheckpointStore
        );
        assert_eq!(imported_handoff.owner_worker_id, "worker-target");
        assert_eq!(imported_handoff.source_worker_id, "worker-source");
        assert_eq!(imported_handoff.sequence.next_position, 1);
    }

    #[tokio::test]
    async fn test_rejects_inline_handoff_without_materialized_payload() {
        let source_dir = TempDir::new().unwrap();
        let source_manager = CheckpointManager::new(
            CheckpointConfig {
                checkpoint_dir: source_dir.path().to_path_buf(),
                ..Default::default()
            },
            "worker-source".to_string(),
        )
        .unwrap();
        let job = create_test_job();
        let snapshot = create_test_snapshot();

        source_manager
            .save_checkpoint(&job, Some(&snapshot))
            .await
            .unwrap();
        let bytes = source_manager
            .export_latest_checkpoint_bytes(job.request.job_id)
            .await
            .unwrap()
            .unwrap();

        let mut checkpoint = Checkpoint::from_cbor(&bytes).unwrap();
        if let Some(handoff) = checkpoint.kv_handoff.as_mut() {
            handoff.payload = None;
            handoff.payload_ref = KVPayloadRef::Inline {
                encoding: KVCacheEncoding::FullSnapshotCbor,
                size_bytes: 4,
            };
        }
        let invalid_bytes = checkpoint.to_cbor().unwrap();

        let target_dir = TempDir::new().unwrap();
        let target_manager = CheckpointManager::new(
            CheckpointConfig {
                checkpoint_dir: target_dir.path().to_path_buf(),
                ..Default::default()
            },
            "worker-target".to_string(),
        )
        .unwrap();

        let error = target_manager
            .import_checkpoint_bytes(&invalid_bytes)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("missing inline payload bytes"));
    }

    #[tokio::test]
    async fn test_list_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            max_checkpoints_per_job: 10,
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        let mut job = create_test_job();
        let job_id = job.request.job_id;

        // Save multiple checkpoints
        manager.save_checkpoint(&job, None).await.unwrap();
        job.add_token(103);
        manager.save_checkpoint(&job, None).await.unwrap();
        job.add_token(104);
        manager.save_checkpoint(&job, None).await.unwrap();

        // List checkpoints
        let checkpoints = manager.list_checkpoints(job_id).await.unwrap();
        assert_eq!(checkpoints.len(), 3);

        // Should be sorted by token index (descending)
        assert_eq!(checkpoints[0].token_index, 5);
        assert_eq!(checkpoints[1].token_index, 4);
        assert_eq!(checkpoints[2].token_index, 3);
    }

    #[tokio::test]
    async fn test_delete_checkpoint() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        let job = create_test_job();
        let job_id = job.request.job_id;

        // Save and then delete
        let metadata = manager.save_checkpoint(&job, None).await.unwrap();
        manager
            .delete_checkpoint(job_id, metadata.checkpoint_id)
            .await
            .unwrap();

        // Should be empty now
        let checkpoints = manager.list_checkpoints(job_id).await.unwrap();
        assert!(checkpoints.is_empty());
    }

    #[tokio::test]
    async fn test_cleanup_old_checkpoints() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            max_checkpoints_per_job: 2,
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();
        let mut job = create_test_job();
        let job_id = job.request.job_id;

        // Save 4 checkpoints (should keep only 2)
        for i in 0..4 {
            job.add_token(200 + i);
            manager.save_checkpoint(&job, None).await.unwrap();
        }

        // Should only have 2 checkpoints
        let checkpoints = manager.list_checkpoints(job_id).await.unwrap();
        assert_eq!(checkpoints.len(), 2);

        // Should be the most recent ones
        assert_eq!(checkpoints[0].token_index, 7);
        assert_eq!(checkpoints[1].token_index, 6);
    }

    #[tokio::test]
    async fn test_storage_usage() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();

        // Initially empty
        let usage = manager.storage_usage().await.unwrap();
        assert_eq!(usage, 0);

        // Save a checkpoint
        let job = create_test_job();
        manager.save_checkpoint(&job, None).await.unwrap();

        // Should have some usage now
        let usage = manager.storage_usage().await.unwrap();
        assert!(usage > 0);
    }

    #[tokio::test]
    async fn test_no_checkpoint_found() {
        let temp_dir = TempDir::new().unwrap();
        let config = CheckpointConfig {
            checkpoint_dir: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CheckpointManager::new(config, "worker-1".to_string()).unwrap();

        // Try to load non-existent checkpoint
        let result = manager.load_checkpoint(Uuid::new_v4()).await.unwrap();
        assert!(result.is_none());
    }
}
