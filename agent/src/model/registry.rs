//! Shard registry for tracking model shard state
//!
//! The ShardRegistry manages the state of all shards assigned to this worker
//! and provides methods for querying and updating shard status.

use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::shard::{ModelInfo, ShardAssignment, ShardInfo};

/// Status of a shard in the registry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStatus {
    /// Shard is assigned but not downloaded
    Pending,
    /// Shard is currently downloading
    Downloading,
    /// Shard is downloaded but not loaded
    Downloaded,
    /// Shard is loaded in memory and ready
    Ready,
    /// Shard failed to load or download
    Error,
}

/// Registry entry for a shard
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryEntry {
    /// Shard information
    info: ShardInfo,
    /// Current status
    status: ShardStatus,
    /// Error message if status is Error
    error: Option<String>,
}

/// Registry for tracking all shards on this worker
pub struct ShardRegistry {
    /// Path to registry storage
    storage_path: PathBuf,

    /// Registered shards by model ID
    shards: RwLock<HashMap<String, RegistryEntry>>,

    /// Available model information
    models: RwLock<HashMap<String, ModelInfo>>,
}

impl ShardRegistry {
    /// Create a new shard registry
    pub fn new(storage_path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&storage_path)?;

        let registry = Self {
            storage_path,
            shards: RwLock::new(HashMap::new()),
            models: RwLock::new(HashMap::new()),
        };

        Ok(registry)
    }

    /// Create with default storage path
    pub fn with_defaults() -> Result<Self> {
        let path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".meshnet")
            .join("shards");
        Self::new(path)
    }

    /// Load registry state from disk
    pub async fn load(&self) -> Result<()> {
        let registry_file = self.storage_path.join("registry.json");

        if !registry_file.exists() {
            debug!("No existing registry file found");
            return Ok(());
        }

        let data = std::fs::read_to_string(&registry_file)?;
        let entries: HashMap<String, RegistryEntry> =
            serde_json::from_str(&data).map_err(|e| AgentError::Config(e.to_string()))?;

        let mut shards = self.shards.write().await;
        *shards = entries;

        info!(count = shards.len(), "Loaded shard registry");
        Ok(())
    }

    /// Save registry state to disk
    pub async fn save(&self) -> Result<()> {
        let registry_file = self.storage_path.join("registry.json");
        let shards = self.shards.read().await;

        let data = serde_json::to_string_pretty(&*shards)
            .map_err(|e| AgentError::Config(e.to_string()))?;

        std::fs::write(&registry_file, data)?;
        debug!("Saved shard registry");
        Ok(())
    }

    /// Register model information
    pub async fn register_model(&self, model: ModelInfo) {
        let model_id = model.model_id.clone();
        let mut models = self.models.write().await;
        models.insert(model_id.clone(), model);
        info!(model_id = %model_id, "Registered model");
    }

    /// Get model information
    pub async fn get_model(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// List all registered models
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }

    /// Assign a shard to this worker
    pub async fn assign_shard(&self, assignment: ShardAssignment) -> Result<()> {
        let model_id = assignment.model_id.clone();
        let info = ShardInfo::new(assignment);

        let entry = RegistryEntry {
            info,
            status: ShardStatus::Pending,
            error: None,
        };

        let mut shards = self.shards.write().await;
        shards.insert(model_id.clone(), entry);

        info!(
            model_id = %model_id,
            "Assigned shard"
        );

        // Persist
        drop(shards);
        self.save().await?;

        Ok(())
    }

    /// Get shard info for a model
    pub async fn get_shard(&self, model_id: &str) -> Option<ShardInfo> {
        let shards = self.shards.read().await;
        shards.get(model_id).map(|e| e.info.clone())
    }

    /// Get shard status
    pub async fn get_shard_status(&self, model_id: &str) -> Option<ShardStatus> {
        let shards = self.shards.read().await;
        shards.get(model_id).map(|e| e.status)
    }

    /// Update shard status
    pub async fn update_status(
        &self,
        model_id: &str,
        status: ShardStatus,
        error: Option<String>,
    ) -> Result<()> {
        let mut shards = self.shards.write().await;

        if let Some(entry) = shards.get_mut(model_id) {
            entry.status = status;
            entry.error = error;
            debug!(model_id = %model_id, status = ?status, "Updated shard status");
        } else {
            return Err(AgentError::Config(format!(
                "Shard not found for model: {}",
                model_id
            )));
        }

        drop(shards);
        self.save().await?;

        Ok(())
    }

    /// Update shard download progress
    pub async fn update_download_progress(
        &self,
        model_id: &str,
        progress: f32,
    ) -> Result<()> {
        let mut shards = self.shards.write().await;

        if let Some(entry) = shards.get_mut(model_id) {
            entry.info.download_progress = progress;
            if progress >= 1.0 && entry.status == ShardStatus::Downloading {
                entry.status = ShardStatus::Downloaded;
            }
        }

        Ok(())
    }

    /// Mark shard as downloaded
    pub async fn mark_downloaded(
        &self,
        model_id: &str,
        path: PathBuf,
        hash: String,
    ) -> Result<()> {
        let mut shards = self.shards.write().await;

        if let Some(entry) = shards.get_mut(model_id) {
            entry.info.set_downloaded(path, hash);
            entry.status = ShardStatus::Downloaded;
            info!(model_id = %model_id, "Shard downloaded");
        }

        drop(shards);
        self.save().await?;

        Ok(())
    }

    /// Mark shard as loaded
    pub async fn mark_loaded(&self, model_id: &str, memory_bytes: u64) -> Result<()> {
        let mut shards = self.shards.write().await;

        if let Some(entry) = shards.get_mut(model_id) {
            entry.info.set_loaded(memory_bytes);
            entry.status = ShardStatus::Ready;
            info!(
                model_id = %model_id,
                memory_gb = memory_bytes / 1_000_000_000,
                "Shard loaded"
            );
        }

        drop(shards);
        self.save().await?;

        Ok(())
    }

    /// Mark shard as unloaded
    pub async fn mark_unloaded(&self, model_id: &str) -> Result<()> {
        let mut shards = self.shards.write().await;

        if let Some(entry) = shards.get_mut(model_id) {
            entry.info.set_unloaded();
            entry.status = ShardStatus::Downloaded;
            info!(model_id = %model_id, "Shard unloaded");
        }

        drop(shards);
        self.save().await?;

        Ok(())
    }

    /// Remove a shard assignment
    pub async fn remove_shard(&self, model_id: &str) -> Result<()> {
        let mut shards = self.shards.write().await;
        shards.remove(model_id);
        info!(model_id = %model_id, "Removed shard");

        drop(shards);
        self.save().await?;

        Ok(())
    }

    /// List all shards
    pub async fn list_shards(&self) -> Vec<(String, ShardInfo, ShardStatus)> {
        let shards = self.shards.read().await;
        shards
            .iter()
            .map(|(id, entry)| (id.clone(), entry.info.clone(), entry.status))
            .collect()
    }

    /// Get all ready shards
    pub async fn ready_shards(&self) -> Vec<ShardInfo> {
        let shards = self.shards.read().await;
        shards
            .values()
            .filter(|e| e.status == ShardStatus::Ready)
            .map(|e| e.info.clone())
            .collect()
    }

    /// Get total memory used by loaded shards
    pub async fn total_memory_usage(&self) -> u64 {
        let shards = self.shards.read().await;
        shards
            .values()
            .filter(|e| e.status == ShardStatus::Ready)
            .map(|e| e.info.memory_bytes)
            .sum()
    }

    /// Check if a shard is ready for inference
    pub async fn is_ready(&self, model_id: &str) -> bool {
        let shards = self.shards.read().await;
        shards
            .get(model_id)
            .map(|e| e.status == ShardStatus::Ready)
            .unwrap_or(false)
    }

    /// Get shard file path for a model
    pub fn shard_path(&self, model_id: &str) -> PathBuf {
        self.storage_path.join(format!("{}.shard", model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_registry() -> (ShardRegistry, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let registry = ShardRegistry::new(temp_dir.path().to_path_buf()).unwrap();
        (registry, temp_dir)
    }

    #[tokio::test]
    async fn test_registry_creation() {
        let (registry, _temp) = create_test_registry().await;
        let shards = registry.list_shards().await;
        assert!(shards.is_empty());
    }

    #[tokio::test]
    async fn test_assign_shard() {
        let (registry, _temp) = create_test_registry().await;

        let assignment = ShardAssignment::new("llama-70b".to_string(), 0, 10);
        registry.assign_shard(assignment).await.unwrap();

        let shard = registry.get_shard("llama-70b").await;
        assert!(shard.is_some());

        let status = registry.get_shard_status("llama-70b").await;
        assert_eq!(status, Some(ShardStatus::Pending));
    }

    #[tokio::test]
    async fn test_shard_lifecycle() {
        let (registry, _temp) = create_test_registry().await;

        // Assign
        let assignment = ShardAssignment::new("model".to_string(), 0, 10);
        registry.assign_shard(assignment).await.unwrap();
        assert_eq!(registry.get_shard_status("model").await, Some(ShardStatus::Pending));

        // Download
        registry
            .update_status("model", ShardStatus::Downloading, None)
            .await
            .unwrap();
        assert_eq!(registry.get_shard_status("model").await, Some(ShardStatus::Downloading));

        // Mark downloaded
        registry
            .mark_downloaded("model", PathBuf::from("/tmp/shard"), "hash123".to_string())
            .await
            .unwrap();
        assert_eq!(registry.get_shard_status("model").await, Some(ShardStatus::Downloaded));

        // Load
        registry.mark_loaded("model", 7_000_000_000).await.unwrap();
        assert_eq!(registry.get_shard_status("model").await, Some(ShardStatus::Ready));
        assert!(registry.is_ready("model").await);

        // Unload
        registry.mark_unloaded("model").await.unwrap();
        assert_eq!(registry.get_shard_status("model").await, Some(ShardStatus::Downloaded));
        assert!(!registry.is_ready("model").await);
    }

    #[tokio::test]
    async fn test_registry_persistence() {
        let temp_dir = TempDir::new().unwrap();

        // Create registry and add shard
        {
            let registry = ShardRegistry::new(temp_dir.path().to_path_buf()).unwrap();
            let assignment = ShardAssignment::new("model".to_string(), 5, 10);
            registry.assign_shard(assignment).await.unwrap();
        }

        // Create new registry and load
        {
            let registry = ShardRegistry::new(temp_dir.path().to_path_buf()).unwrap();
            registry.load().await.unwrap();

            let shard = registry.get_shard("model").await;
            assert!(shard.is_some());
            assert_eq!(shard.unwrap().assignment.worker_position, 5);
        }
    }

    #[tokio::test]
    async fn test_memory_usage() {
        let (registry, _temp) = create_test_registry().await;

        // Add two shards
        let a1 = ShardAssignment::new("model1".to_string(), 0, 10);
        let a2 = ShardAssignment::new("model2".to_string(), 0, 10);

        registry.assign_shard(a1).await.unwrap();
        registry.assign_shard(a2).await.unwrap();

        // Load one
        registry.mark_loaded("model1", 7_000_000_000).await.unwrap();

        assert_eq!(registry.total_memory_usage().await, 7_000_000_000);

        // Load second
        registry.mark_loaded("model2", 3_000_000_000).await.unwrap();

        assert_eq!(registry.total_memory_usage().await, 10_000_000_000);
    }

    #[tokio::test]
    async fn test_ready_shards() {
        let (registry, _temp) = create_test_registry().await;

        let a1 = ShardAssignment::new("model1".to_string(), 0, 10);
        let a2 = ShardAssignment::new("model2".to_string(), 0, 10);

        registry.assign_shard(a1).await.unwrap();
        registry.assign_shard(a2).await.unwrap();

        // Only load one
        registry.mark_loaded("model1", 5_000_000_000).await.unwrap();

        let ready = registry.ready_shards().await;
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].assignment.model_id, "model1");
    }

    #[tokio::test]
    async fn test_model_registration() {
        let (registry, _temp) = create_test_registry().await;

        let model = ModelInfo {
            model_id: "llama-70b".to_string(),
            name: "LLaMA 70B".to_string(),
            num_layers: 80,
            hidden_dim: 8192,
            num_heads: 64,
            vocab_size: 32000,
            total_size_bytes: 140_000_000_000,
            min_workers: 10,
            recommended_workers: 20,
            memory_per_worker_bytes: 7_000_000_000,
            context_length: 4096,
            format: "safetensors".to_string(),
            shard_urls: vec![],
        };

        registry.register_model(model.clone()).await;

        let fetched = registry.get_model("llama-70b").await;
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().num_layers, 80);
    }
}
