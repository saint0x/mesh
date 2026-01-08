//! # Mock Shard Loader (Development Only)
//!
//! This module provides a realistic simulation of model shard loading for testing
//! the complete inference pipeline without requiring actual model files.
//!
//! ## ⚠️ TODO: Replace with Real Loader
//!
//! For production inference:
//! 1. Implement SafetensorsShardLoader with actual file I/O
//! 2. Swap loader in InferenceCoordinator initialization
//! 3. No changes needed to downstream code (ForwardPass, etc.)
//!
//! ## Purpose
//!
//! Simulates the complete shard lifecycle:
//! - Download phase with progress tracking
//! - Disk storage and hash verification
//! - Loading into memory
//! - Registry state management
//!
//! Uses the same deterministic weight generation as mock_validation.rs
//! but wraps it in realistic shard loader behavior.
//!
//! ## Mock vs Real Behavior
//!
//! | Aspect | Mock (this module) | Real (SafetensorsShardLoader) |
//! |--------|-------------------|------------------------------|
//! | Download | Simulated with progress | Real HTTP download |
//! | Storage | Simulated path | Real file I/O |
//! | Weights | Xavier-initialized | Actual trained weights |
//! | Hash | Deterministic from seed | Real SHA256 verification |
//! | Speed | Configurable simulation | Real network/disk speed |
//!
//! ## Usage
//!
//! ```rust,no_run
//! # use agent::inference::mock_loader::{MockShardLoader, ShardLoader};
//! # use agent::inference::forward_pass::ModelConfig;
//! # use agent::model::shard::ShardAssignment;
//! # use agent::model::registry::ShardRegistry;
//! # use std::path::PathBuf;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ModelConfig::default();
//! let loader = MockShardLoader::new(12345, config, true);
//! let registry = ShardRegistry::new(PathBuf::from("/tmp/shards"))?;
//! let assignment = ShardAssignment::new("llama-70b".into(), 0, 10);
//! let weights = loader.load_shard(
//!     "llama-70b",
//!     &assignment,
//!     &registry,
//! ).await?;
//! # Ok(())
//! # }
//! ```

use crate::errors::{AgentError, Result};
use crate::model::registry::ShardRegistry;
use crate::model::shard::ShardAssignment;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::forward_pass::{ModelConfig, ModelWeights};
use super::mock_validation;

/// Trait for loading model shards (mock or real)
///
/// This trait defines the interface that both mock and production
/// safetensors loaders must implement, enabling seamless migration.
#[async_trait]
pub trait ShardLoader: Send + Sync {
    /// Load weights for a specific shard
    ///
    /// # Arguments
    /// - model_id: Model identifier (e.g., "llama-70b")
    /// - assignment: Column range and position info
    /// - registry: Shard registry for lifecycle tracking
    ///
    /// # Returns
    /// ModelWeights for this worker's shard
    async fn load_shard(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        registry: &ShardRegistry,
    ) -> Result<ModelWeights>;

    /// Check if shard is already cached/downloaded
    async fn is_cached(&self, model_id: &str) -> bool;

    /// Get estimated memory usage for a shard
    fn estimate_memory(&self, assignment: &ShardAssignment) -> u64;
}

/// Cached shard data
#[derive(Debug, Clone)]
struct CachedShard {
    /// When this was "downloaded"
    downloaded_at: Instant,

    /// File path (simulated)
    path: PathBuf,

    /// Hash (deterministic from seed + model_id)
    hash: String,

    /// Shard assignment info
    assignment: ShardAssignment,

    /// Whether loaded in memory
    is_loaded: bool,
}

/// Mock shard loader that simulates realistic download/loading behavior
///
/// Uses the same deterministic weight generation as mock_validation.rs
/// but wraps it in realistic shard lifecycle simulation.
pub struct MockShardLoader {
    /// Seed for deterministic weight generation
    seed: u64,

    /// Model configuration
    config: ModelConfig,

    /// Simulated download speed (bytes/sec)
    download_speed_bytes_per_sec: u64,

    /// Cache of already "downloaded" shards (model_id -> shard data)
    cache: Arc<RwLock<HashMap<String, CachedShard>>>,

    /// Whether to simulate download delays
    simulate_download: bool,
}

impl MockShardLoader {
    /// Create with custom configuration
    pub fn new(seed: u64, config: ModelConfig, simulate_download: bool) -> Self {
        Self {
            seed,
            config,
            download_speed_bytes_per_sec: 100_000_000, // 100 MB/s default
            cache: Arc::new(RwLock::new(HashMap::new())),
            simulate_download,
        }
    }

    /// Create with sensible defaults (no download simulation)
    pub fn with_defaults() -> Self {
        Self::new(12345, ModelConfig::default(), false)
    }

    /// Create with download simulation for testing (uses small config for speed)
    pub fn with_simulation(seed: u64) -> Self {
        let small_config = ModelConfig {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        Self::new(seed, small_config, true)
    }

    /// Set simulated download speed
    pub fn with_download_speed(mut self, bytes_per_sec: u64) -> Self {
        self.download_speed_bytes_per_sec = bytes_per_sec;
        self
    }

    /// Generate deterministic hash from seed and model_id
    fn generate_shard_hash(&self, model_id: &str) -> String {
        // Use simple deterministic hash (not cryptographic)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        model_id.hash(&mut hasher);

        format!("mock-sha256-{:016x}", hasher.finish())
    }

    /// Simulate downloading a shard with progress tracking
    ///
    /// Note: Assumes shard is already assigned in registry (called from load_shard)
    async fn simulate_download_phase(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        registry: &ShardRegistry,
    ) -> Result<()> {
        // Mark as Downloading
        registry
            .update_status(model_id, crate::model::registry::ShardStatus::Downloading, None)
            .await?;

        // Calculate shard size based on column range
        let shard_bytes = self.estimate_memory(assignment);

        if self.simulate_download {
            // Simulate download with progress updates
            let total_steps = 20;
            let bytes_per_step = shard_bytes / total_steps;

            for step in 0..total_steps {
                // Simulate network delay
                let step_duration = std::time::Duration::from_millis(
                    (bytes_per_step * 1000 / self.download_speed_bytes_per_sec) as u64,
                );
                tokio::time::sleep(step_duration).await;

                // Update progress
                let progress = (step + 1) as f32 / total_steps as f32;
                registry
                    .update_download_progress(model_id, progress)
                    .await?;

                debug!(
                    model_id = %model_id,
                    progress = format!("{:.1}%", progress * 100.0),
                    "Simulated download progress"
                );
            }
        }

        // Generate deterministic hash from seed + model_id
        let hash = self.generate_shard_hash(model_id);

        // "Write" to cache directory (simulated path)
        let path = registry.shard_path(model_id);

        // Mark as Downloaded
        registry
            .mark_downloaded(model_id, path.clone(), hash.clone())
            .await?;

        // Add to cache
        let mut cache = self.cache.write().await;
        cache.insert(
            model_id.to_string(),
            CachedShard {
                downloaded_at: Instant::now(),
                path,
                hash,
                assignment: assignment.clone(),
                is_loaded: false,
            },
        );

        Ok(())
    }

    /// Simulate loading from "disk" to memory
    async fn simulate_load_phase(
        &self,
        model_id: &str,
        _assignment: &ShardAssignment,
        _registry: &ShardRegistry,
    ) -> Result<()> {
        // Check cache
        let mut cache = self.cache.write().await;
        let cached = cache.get_mut(model_id).ok_or_else(|| {
            AgentError::Config(format!("Shard not downloaded: {}", model_id))
        })?;

        if cached.is_loaded {
            // Already loaded, return early
            return Ok(());
        }

        // Simulate "loading" delay (reading from disk, decompression, etc.)
        if self.simulate_download {
            let load_delay = std::time::Duration::from_millis(100); // Simulated disk I/O
            tokio::time::sleep(load_delay).await;
        }

        cached.is_loaded = true;

        Ok(())
    }

    /// Generate deterministic weights (delegates to mock_validation)
    fn generate_weights(&self, assignment: &ShardAssignment) -> ModelWeights {
        let shard_cols = assignment.num_columns() as usize;

        // Use same deterministic generation as mock_validation
        mock_validation::generate_mock_weights(&self.config, shard_cols, self.seed)
    }
}

#[async_trait]
impl ShardLoader for MockShardLoader {
    /// Load weights for a specific shard
    ///
    /// This simulates the complete shard lifecycle:
    /// 1. Check cache (avoid redundant downloads)
    /// 2. Download phase (with progress tracking)
    /// 3. Load phase (simulate disk I/O)
    /// 4. Generate weights (deterministic from seed)
    /// 5. Mark as Ready in registry
    async fn load_shard(
        &self,
        model_id: &str,
        assignment: &ShardAssignment,
        registry: &ShardRegistry,
    ) -> Result<ModelWeights> {
        // Validate model_id matches assignment
        if model_id != assignment.model_id {
            return Err(AgentError::Config(format!(
                "model_id mismatch: parameter='{}' vs assignment.model_id='{}'",
                model_id, assignment.model_id
            )));
        }

        info!(
            model_id = %model_id,
            position = assignment.worker_position,
            column_range = ?assignment.column_range(),
            "Loading shard"
        );

        // Ensure shard is assigned in registry
        if registry.get_shard(model_id).await.is_none() {
            debug!(model_id = %model_id, "Assigning shard to registry");
            registry.assign_shard(assignment.clone()).await?;
        }

        // 1. Check cache first
        if !self.is_cached(model_id).await {
            debug!(model_id = %model_id, "Cache miss, simulating download");
            // 2. Simulate download phase
            self.simulate_download_phase(model_id, assignment, registry)
                .await?;
        } else {
            debug!(model_id = %model_id, "Cache hit, skipping download");
        }

        // 3. Simulate loading from "disk" to memory
        self.simulate_load_phase(model_id, assignment, registry)
            .await?;

        // 4. Generate deterministic weights (same as mock_validation)
        let weights = self.generate_weights(assignment);

        // 5. Mark as Ready in registry
        registry
            .mark_loaded(model_id, weights.memory_usage() as u64)
            .await?;

        info!(
            model_id = %model_id,
            memory_mb = weights.memory_usage() / 1_000_000,
            "Shard loaded successfully"
        );

        Ok(weights)
    }

    /// Check if shard is already cached/downloaded
    async fn is_cached(&self, model_id: &str) -> bool {
        let cache = self.cache.read().await;
        cache.contains_key(model_id)
    }

    /// Get estimated memory usage for a shard
    ///
    /// This estimates based on:
    /// - Number of columns in shard
    /// - Hidden dimension
    /// - Number of layers
    /// - Vocabulary size (for embedding/lm_head)
    ///
    /// Formula: Roughly 4 bytes per f32 weight
    fn estimate_memory(&self, assignment: &ShardAssignment) -> u64 {
        let shard_cols = assignment.num_columns() as usize;
        let hidden_dim = self.config.hidden_dim;
        let num_layers = self.config.num_layers;
        let vocab_size = self.config.vocab_size;

        // Per-layer weights:
        // - w_q, w_k, w_v: hidden_dim × shard_cols each (3 matrices)
        // - w_o: shard_cols × hidden_dim (1 matrix)
        // - w_up, w_gate: hidden_dim × shard_cols each (2 matrices)
        // - w_down: shard_cols × hidden_dim (1 matrix)
        // - norms: hidden_dim each (2 vectors)
        let per_layer_weights = (hidden_dim * shard_cols * 3) // Q, K, V
            + (shard_cols * hidden_dim) // O
            + (hidden_dim * shard_cols * 2) // Up, Gate
            + (shard_cols * hidden_dim) // Down
            + (hidden_dim * 2); // Norms

        let total_layer_weights = per_layer_weights * num_layers;

        // Embedding and LM head
        let embedding_weights = vocab_size * hidden_dim;
        let lm_head_weights = hidden_dim * vocab_size;

        let total_weights = total_layer_weights + embedding_weights + lm_head_weights;

        // 4 bytes per f32
        (total_weights * 4) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a small model config for fast testing
    ///
    /// Production config (80 layers, 8192 hidden, 32k vocab) generates gigabytes of data.
    /// This test config (2 layers, 128 hidden, 1k vocab) is 1000x smaller and runs instantly.
    fn test_model_config() -> ModelConfig {
        ModelConfig {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    async fn create_test_registry() -> (ShardRegistry, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let registry = ShardRegistry::new(temp_dir.path().to_path_buf()).unwrap();
        (registry, temp_dir)
    }

    #[tokio::test]
    async fn test_mock_loader_determinism() {
        let config = test_model_config();
        let loader1 = MockShardLoader::new(12345, config.clone(), false);
        let loader2 = MockShardLoader::new(12345, config, false);

        let (registry, _temp) = create_test_registry().await;
        let assignment1 = ShardAssignment::new("model1".into(), 0, 10);
        let assignment2 = ShardAssignment::new("model2".into(), 0, 10);

        let w1 = loader1
            .load_shard("model1", &assignment1, &registry)
            .await
            .unwrap();
        let w2 = loader2
            .load_shard("model2", &assignment2, &registry)
            .await
            .unwrap();

        // Same seed must produce identical weights
        assert_eq!(
            w1.embedding.data, w2.embedding.data,
            "Same seed must produce identical embedding weights"
        );
        assert_eq!(
            w1.layers[0].w_q.data, w2.layers[0].w_q.data,
            "Same seed must produce identical layer weights"
        );
    }

    #[tokio::test]
    async fn test_mock_loader_different_seeds() {
        let config = test_model_config();
        let loader1 = MockShardLoader::new(12345, config.clone(), false);
        let loader2 = MockShardLoader::new(54321, config, false);

        let (registry, _temp) = create_test_registry().await;
        let assignment1 = ShardAssignment::new("model1".into(), 0, 10);
        let assignment2 = ShardAssignment::new("model2".into(), 0, 10);

        let w1 = loader1
            .load_shard("model1", &assignment1, &registry)
            .await
            .unwrap();
        let w2 = loader2
            .load_shard("model2", &assignment2, &registry)
            .await
            .unwrap();

        // Different seeds must produce different weights
        assert_ne!(
            w1.embedding.data, w2.embedding.data,
            "Different seeds must produce different weights"
        );
    }

    #[tokio::test]
    async fn test_mock_loader_lifecycle_tracking() {
        let config = test_model_config();
        let loader = MockShardLoader::new(12345, config, true);
        let (registry, _temp) = create_test_registry().await;
        let assignment = ShardAssignment::new("model".into(), 0, 10);

        // Assign shard
        registry.assign_shard(assignment.clone()).await.unwrap();

        // Initially Pending
        assert_eq!(
            registry.get_shard_status("model").await,
            Some(crate::model::registry::ShardStatus::Pending)
        );

        // Load shard (will go through full lifecycle)
        let _weights = loader
            .load_shard("model", &assignment, &registry)
            .await
            .unwrap();

        // Should now be Ready
        assert_eq!(
            registry.get_shard_status("model").await,
            Some(crate::model::registry::ShardStatus::Ready)
        );
    }

    #[tokio::test]
    async fn test_cache_hit_behavior() {
        let config = test_model_config();
        let loader = MockShardLoader::new(12345, config, false);
        let (registry, _temp) = create_test_registry().await;
        let assignment = ShardAssignment::new("model".into(), 0, 10);

        // First load (cache miss)
        assert!(!loader.is_cached("model").await);
        let _w1 = loader
            .load_shard("model", &assignment, &registry)
            .await
            .unwrap();

        // Should now be cached
        assert!(loader.is_cached("model").await);

        // Second load (cache hit)
        let _w2 = loader
            .load_shard("model", &assignment, &registry)
            .await
            .unwrap();

        // Cache should still be valid
        assert!(loader.is_cached("model").await);
    }

    #[tokio::test]
    async fn test_shard_hash_generation() {
        let loader = MockShardLoader::new(12345, ModelConfig::default(), false);

        let hash1 = loader.generate_shard_hash("llama-70b");
        let hash2 = loader.generate_shard_hash("llama-70b");

        // Same model_id should produce same hash
        assert_eq!(hash1, hash2);

        let hash3 = loader.generate_shard_hash("different-model");
        // Different model_id should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[tokio::test]
    async fn test_memory_estimation() {
        let loader = MockShardLoader::new(12345, ModelConfig::default(), false);

        let assignment1 = ShardAssignment::new("model".into(), 0, 10);
        let assignment2 = ShardAssignment::new("model".into(), 0, 5);

        let mem1 = loader.estimate_memory(&assignment1);
        let mem2 = loader.estimate_memory(&assignment2);

        // More columns should use more memory
        assert!(mem2 > mem1, "Fewer workers (more columns) should use more memory");

        // Memory should be reasonable (not 0, not absurdly large)
        assert!(mem1 > 1_000_000, "Memory should be > 1MB");
        assert!(mem1 < 100_000_000_000, "Memory should be < 100GB");
    }

    #[tokio::test]
    async fn test_weights_match_mock_validation() {
        // Weights should match direct mock_validation call
        let config = test_model_config();
        let seed = 12345_u64;

        let loader = MockShardLoader::new(seed, config.clone(), false);
        let (registry, _temp) = create_test_registry().await;
        let assignment = ShardAssignment::new("model".into(), 0, 10);

        // Get actual shard_cols from assignment (8192 / 10 = 819 columns)
        let shard_cols = assignment.num_columns() as usize;

        let loader_weights = loader
            .load_shard("model", &assignment, &registry)
            .await
            .unwrap();
        let direct_weights = mock_validation::generate_mock_weights(&config, shard_cols, seed);

        // Should produce identical weights
        assert_eq!(
            loader_weights.embedding.data, direct_weights.embedding.data,
            "Loader weights should match direct mock_validation call"
        );
        assert_eq!(
            loader_weights.layers[0].w_q.data, direct_weights.layers[0].w_q.data,
            "Layer weights should match"
        );
    }

    #[tokio::test]
    async fn test_download_simulation_progress() {
        let config = test_model_config();
        let loader = MockShardLoader::new(12345, config, true);
        let (registry, _temp) = create_test_registry().await;
        let assignment = ShardAssignment::new("model".into(), 0, 10);

        registry.assign_shard(assignment.clone()).await.unwrap();

        // Load with simulation
        let _weights = loader
            .load_shard("model", &assignment, &registry)
            .await
            .unwrap();

        // Should have progress tracked (check shard info)
        let shard = registry.get_shard("model").await.unwrap();
        // Download progress should be 1.0 (100%)
        assert!(
            (shard.download_progress - 1.0).abs() < 0.01,
            "Download progress should be 100%"
        );
    }

    #[tokio::test]
    async fn test_concurrent_loads() {
        let config = test_model_config();
        let loader = Arc::new(MockShardLoader::new(12345, config, false));
        let (registry, _temp) = create_test_registry().await;
        let registry = Arc::new(registry);

        let assignment = ShardAssignment::new("model".into(), 0, 10);

        // Spawn multiple concurrent loads
        let mut handles = vec![];
        for _ in 0..5 {
            let loader = Arc::clone(&loader);
            let registry = Arc::clone(&registry);
            let assignment = assignment.clone();

            let handle = tokio::spawn(async move {
                loader
                    .load_shard("model", &assignment, &registry)
                    .await
                    .unwrap()
            });
            handles.push(handle);
        }

        // All should complete successfully
        let results: Vec<_> = futures::future::join_all(handles).await;
        assert_eq!(results.len(), 5);

        // All should produce identical weights (same seed)
        let first = results[0].as_ref().unwrap();
        for result in &results[1..] {
            let weights = result.as_ref().unwrap();
            assert_eq!(
                first.embedding.data, weights.embedding.data,
                "Concurrent loads should produce identical weights"
            );
        }
    }

    #[tokio::test]
    async fn test_error_handling_missing_assignment() {
        let config = test_model_config();
        let loader = MockShardLoader::new(12345, config, false);
        let (registry, _temp) = create_test_registry().await;
        let assignment = ShardAssignment::new("model".into(), 0, 10);

        // Loader automatically assigns if missing - should succeed
        let result = loader.load_shard("model", &assignment, &registry).await;
        assert!(result.is_ok(), "Loader should auto-assign shard");
    }

    #[tokio::test]
    async fn test_with_defaults() {
        let loader = MockShardLoader::with_defaults();
        assert_eq!(loader.seed, 12345);
        assert!(!loader.simulate_download);
    }

    #[tokio::test]
    async fn test_with_simulation() {
        let loader = MockShardLoader::with_simulation(99999);
        assert_eq!(loader.seed, 99999);
        assert!(loader.simulate_download);
    }

    #[tokio::test]
    async fn test_with_download_speed() {
        let loader = MockShardLoader::with_defaults().with_download_speed(50_000_000);
        assert_eq!(loader.download_speed_bytes_per_sec, 50_000_000);
    }

    /// Production-scale test with full ModelConfig::default()
    ///
    /// This test validates that the loader works with production-scale models:
    /// - 80 layers
    /// - 8192 hidden dimension
    /// - 32,000 vocabulary size
    /// - Generates multiple gigabytes of weights
    ///
    /// Run with: `cargo test test_production_scale_model -- --ignored --nocapture`
    ///
    /// Expected: Test passes but may take 30-60 seconds due to large weight generation
    #[tokio::test]
    #[ignore] // Run explicitly with --ignored flag
    async fn test_production_scale_model() {
        let config = ModelConfig::default(); // Full production config
        let loader = MockShardLoader::new(12345, config, false);
        let (registry, _temp) = create_test_registry().await;
        let assignment = ShardAssignment::new("llama-70b".into(), 0, 10);

        println!("Loading production-scale model (80 layers, 8192 hidden, 32k vocab)...");
        println!("This will generate ~1GB of weights per worker...");

        let start = Instant::now();
        let weights = loader
            .load_shard("llama-70b", &assignment, &registry)
            .await
            .unwrap();
        let duration = start.elapsed();

        println!("Production-scale load completed in {:?}", duration);
        println!("Weights memory usage: {} MB", weights.memory_usage() / 1_000_000);

        // Verify structure is correct
        assert_eq!(weights.layers.len(), 80, "Should have 80 layers");
        assert_eq!(weights.config.hidden_dim, 8192, "Should have 8192 hidden dim");
        assert_eq!(weights.config.vocab_size, 32000, "Should have 32k vocab");

        // Verify cache works
        assert!(loader.is_cached("llama-70b").await);
    }
}
