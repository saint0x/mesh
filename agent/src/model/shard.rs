//! Model shard information and assignment types
//!
//! This module defines the types for tracking model shard assignments
//! and metadata across workers in a tensor-parallel configuration.

use serde::{Deserialize, Serialize};

/// Total number of columns in the shard space (fixed)
pub const TOTAL_COLUMNS: u32 = 8192;

/// Information about a model available for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g., "llama-70b", "mixtral-8x7b")
    pub model_id: String,

    /// Human-readable model name
    pub name: String,

    /// Number of transformer layers
    pub num_layers: u32,

    /// Hidden dimension size
    pub hidden_dim: u32,

    /// Number of attention heads
    pub num_heads: u32,

    /// Vocabulary size
    pub vocab_size: u32,

    /// Total model size in bytes
    pub total_size_bytes: u64,

    /// Minimum workers required for inference
    pub min_workers: u32,

    /// Recommended workers for optimal performance
    pub recommended_workers: u32,

    /// Memory required per worker (estimated)
    pub memory_per_worker_bytes: u64,

    /// Context length supported
    pub context_length: u32,

    /// Model format (e.g., "safetensors", "ggml")
    pub format: String,

    /// Download URLs for shards
    pub shard_urls: Vec<String>,
}

impl ModelInfo {
    /// Calculate memory required per worker for a given number of workers
    pub fn memory_for_workers(&self, num_workers: u32) -> u64 {
        if num_workers == 0 {
            return self.total_size_bytes;
        }
        // Each worker gets a proportional share of model weights
        // Plus some overhead for activations and KV cache
        let weight_share = self.total_size_bytes / num_workers as u64;
        let overhead = weight_share / 10; // ~10% overhead
        weight_share + overhead
    }

    /// Calculate columns per worker for a given number of workers
    pub fn columns_per_worker(&self, num_workers: u32) -> u32 {
        if num_workers == 0 {
            return TOTAL_COLUMNS;
        }
        TOTAL_COLUMNS / num_workers
    }

    /// Check if the given number of workers can run this model
    pub fn can_run_with_workers(&self, num_workers: u32) -> bool {
        num_workers >= self.min_workers
    }
}

/// Assignment of shard columns to this worker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShardAssignment {
    /// Model this shard belongs to
    pub model_id: String,

    /// Worker position in the ring (0-indexed)
    pub worker_position: u32,

    /// Total workers in the ring
    pub total_workers: u32,

    /// Start column (inclusive)
    pub column_start: u32,

    /// End column (exclusive)
    pub column_end: u32,
}

impl ShardAssignment {
    /// Create a new shard assignment
    pub fn new(
        model_id: String,
        worker_position: u32,
        total_workers: u32,
    ) -> Self {
        // Calculate column range for this worker
        let cols_per_worker = TOTAL_COLUMNS / total_workers;
        let column_start = worker_position * cols_per_worker;
        let column_end = if worker_position == total_workers - 1 {
            TOTAL_COLUMNS // Last worker gets any remainder
        } else {
            (worker_position + 1) * cols_per_worker
        };

        Self {
            model_id,
            worker_position,
            total_workers,
            column_start,
            column_end,
        }
    }

    /// Get the number of columns in this shard
    pub fn num_columns(&self) -> u32 {
        self.column_end - self.column_start
    }

    /// Check if a column is in this shard
    pub fn contains_column(&self, column: u32) -> bool {
        column >= self.column_start && column < self.column_end
    }

    /// Get the column range as a tuple
    pub fn column_range(&self) -> (u32, u32) {
        (self.column_start, self.column_end)
    }
}

/// Detailed information about a shard on this worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Shard assignment
    pub assignment: ShardAssignment,

    /// Whether shard weights are loaded in memory
    pub is_loaded: bool,

    /// Memory used by this shard in bytes
    pub memory_bytes: u64,

    /// Path to shard file (if downloaded)
    pub file_path: Option<std::path::PathBuf>,

    /// SHA256 hash of shard file for verification
    pub file_hash: Option<String>,

    /// Download progress (0.0 - 1.0)
    pub download_progress: f32,

    /// Last verification timestamp
    pub last_verified: Option<u64>,
}

impl ShardInfo {
    /// Create new shard info from assignment
    pub fn new(assignment: ShardAssignment) -> Self {
        Self {
            assignment,
            is_loaded: false,
            memory_bytes: 0,
            file_path: None,
            file_hash: None,
            download_progress: 0.0,
            last_verified: None,
        }
    }

    /// Check if shard is ready for inference
    pub fn is_ready(&self) -> bool {
        self.is_loaded && self.memory_bytes > 0
    }

    /// Mark as downloaded with path
    pub fn set_downloaded(&mut self, path: std::path::PathBuf, hash: String) {
        self.file_path = Some(path);
        self.file_hash = Some(hash);
        self.download_progress = 1.0;
    }

    /// Mark as loaded in memory
    pub fn set_loaded(&mut self, memory_bytes: u64) {
        self.is_loaded = true;
        self.memory_bytes = memory_bytes;
        self.last_verified = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );
    }

    /// Mark as unloaded from memory
    pub fn set_unloaded(&mut self) {
        self.is_loaded = false;
        self.memory_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_memory() {
        let model = ModelInfo {
            model_id: "llama-70b".to_string(),
            name: "LLaMA 70B".to_string(),
            num_layers: 80,
            hidden_dim: 8192,
            num_heads: 64,
            vocab_size: 32000,
            total_size_bytes: 140_000_000_000, // ~140GB
            min_workers: 10,
            recommended_workers: 20,
            memory_per_worker_bytes: 7_000_000_000,
            context_length: 4096,
            format: "safetensors".to_string(),
            shard_urls: vec![],
        };

        // 10 workers = ~15.4GB each (14GB + 10% overhead)
        let mem_10 = model.memory_for_workers(10);
        assert!(mem_10 > 14_000_000_000);
        assert!(mem_10 < 16_000_000_000);

        // 20 workers = ~7.7GB each
        let mem_20 = model.memory_for_workers(20);
        assert!(mem_20 > 7_000_000_000);
        assert!(mem_20 < 8_000_000_000);
    }

    #[test]
    fn test_model_info_columns() {
        let model = ModelInfo {
            model_id: "test".to_string(),
            name: "Test".to_string(),
            num_layers: 10,
            hidden_dim: 8192,
            num_heads: 32,
            vocab_size: 32000,
            total_size_bytes: 10_000_000_000,
            min_workers: 2,
            recommended_workers: 4,
            memory_per_worker_bytes: 5_000_000_000,
            context_length: 4096,
            format: "safetensors".to_string(),
            shard_urls: vec![],
        };

        assert_eq!(model.columns_per_worker(8), 1024); // 8192 / 8
        assert_eq!(model.columns_per_worker(10), 819); // 8192 / 10
    }

    #[test]
    fn test_shard_assignment_new() {
        // 10 workers, position 0
        let shard = ShardAssignment::new("llama-70b".to_string(), 0, 10);
        assert_eq!(shard.column_start, 0);
        assert_eq!(shard.column_end, 819);
        assert_eq!(shard.num_columns(), 819);

        // 10 workers, position 5 (middle)
        let shard = ShardAssignment::new("llama-70b".to_string(), 5, 10);
        assert_eq!(shard.column_start, 4095);
        assert_eq!(shard.column_end, 4914);

        // 10 workers, position 9 (last)
        let shard = ShardAssignment::new("llama-70b".to_string(), 9, 10);
        assert_eq!(shard.column_start, 7371);
        assert_eq!(shard.column_end, 8192); // Gets remainder
    }

    #[test]
    fn test_shard_assignment_contains_column() {
        let shard = ShardAssignment::new("model".to_string(), 1, 4);
        // Worker 1 of 4 = columns 2048-4095

        assert!(!shard.contains_column(0));
        assert!(!shard.contains_column(2047));
        assert!(shard.contains_column(2048));
        assert!(shard.contains_column(3000));
        assert!(shard.contains_column(4095));
        assert!(!shard.contains_column(4096));
    }

    #[test]
    fn test_shard_info_lifecycle() {
        let assignment = ShardAssignment::new("model".to_string(), 0, 10);
        let mut shard = ShardInfo::new(assignment);

        // Initially not ready
        assert!(!shard.is_ready());
        assert!(!shard.is_loaded);
        assert_eq!(shard.download_progress, 0.0);

        // Download complete
        shard.set_downloaded(
            std::path::PathBuf::from("/tmp/shard.bin"),
            "abc123".to_string(),
        );
        assert_eq!(shard.download_progress, 1.0);
        assert!(!shard.is_ready()); // Still not loaded

        // Load into memory
        shard.set_loaded(7_000_000_000);
        assert!(shard.is_ready());
        assert!(shard.is_loaded);
        assert_eq!(shard.memory_bytes, 7_000_000_000);

        // Unload
        shard.set_unloaded();
        assert!(!shard.is_ready());
        assert!(!shard.is_loaded);
        assert_eq!(shard.memory_bytes, 0);
    }

    #[test]
    fn test_full_column_coverage() {
        // Verify that 10 workers cover all 8192 columns with no gaps
        let mut all_columns = vec![false; TOTAL_COLUMNS as usize];

        for position in 0..10 {
            let shard = ShardAssignment::new("model".to_string(), position, 10);
            for col in shard.column_start..shard.column_end {
                all_columns[col as usize] = true;
            }
        }

        // All columns should be covered
        for (i, covered) in all_columns.iter().enumerate() {
            assert!(*covered, "Column {} not covered", i);
        }
    }
}
