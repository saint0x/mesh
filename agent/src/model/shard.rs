//! Model shard assignment for pipeline-parallel distributed inference
//!
//! In pipeline parallelism, each worker is responsible for a contiguous range
//! of transformer layers rather than a column range of weight matrices.
//!
//! ## Architecture
//!
//! ```text
//! Full Model (80 layers)
//! ┌─────────┬─────────┬─────────┬─────────┐
//! │ Worker 0 │ Worker 1 │ Worker 2 │ Worker 3│
//! │ L0-L19  │ L20-L39 │ L40-L59 │ L60-L79 │
//! └─────────┴─────────┴─────────┴─────────┘
//! ```
//!
//! Activations flow sequentially through the pipeline:
//! Worker 0 → Worker 1 → ... → Worker N-1 (forward pass)
//! Worker N-1 produces logits and samples the next token.
//! The sampled token is broadcast back to Worker 0 for the next step.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Total number of layers used as the default shard namespace (LLaMA 70B).
pub const DEFAULT_TOTAL_LAYERS: u32 = 80;

/// Shard assignment for pipeline parallelism (layer-based)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardAssignment {
    /// Model this shard belongs to
    pub model_id: String,

    /// This worker's position in the pipeline (0-indexed)
    pub worker_position: u32,

    /// Total workers in the pipeline
    pub total_workers: u32,

    /// First layer this worker is responsible for (inclusive)
    pub layer_start: u32,

    /// Last layer this worker is responsible for (exclusive)
    pub layer_end: u32,
}

impl ShardAssignment {
    /// Create a new shard assignment using layer-based pipeline partitioning
    ///
    /// Layers are distributed as evenly as possible. If the number of layers
    /// doesn't divide evenly, earlier workers get one extra layer each.
    pub fn new(model_id: String, worker_position: u32, total_workers: u32) -> Self {
        Self::with_layers(model_id, worker_position, total_workers, DEFAULT_TOTAL_LAYERS)
    }

    /// Create a shard assignment with a specific total layer count
    pub fn with_layers(
        model_id: String,
        worker_position: u32,
        total_workers: u32,
        total_layers: u32,
    ) -> Self {
        let (start, end) = Self::compute_layer_range(worker_position, total_workers, total_layers);
        Self {
            model_id,
            worker_position,
            total_workers,
            layer_start: start,
            layer_end: end,
        }
    }

    /// Compute the layer range for a given position
    fn compute_layer_range(position: u32, total_workers: u32, total_layers: u32) -> (u32, u32) {
        if total_workers == 0 {
            return (0, 0);
        }
        let layers_per_worker = total_layers / total_workers;
        let remainder = total_layers % total_workers;

        let start = if position < remainder {
            position * (layers_per_worker + 1)
        } else {
            remainder * (layers_per_worker + 1) + (position - remainder) * layers_per_worker
        };

        let end = if position < remainder {
            start + layers_per_worker + 1
        } else {
            start + layers_per_worker
        };

        (start, end)
    }

    /// Number of layers this worker is responsible for
    pub fn num_layers(&self) -> u32 {
        self.layer_end - self.layer_start
    }

    /// Layer range as (start_inclusive, end_exclusive)
    pub fn layer_range(&self) -> (u32, u32) {
        (self.layer_start, self.layer_end)
    }

    /// Whether this worker owns the given layer index
    pub fn contains_layer(&self, layer_idx: u32) -> bool {
        layer_idx >= self.layer_start && layer_idx < self.layer_end
    }

    /// Whether this is the first stage in the pipeline (runs embedding)
    pub fn is_first_stage(&self) -> bool {
        self.worker_position == 0
    }

    /// Whether this is the last stage in the pipeline (runs lm_head + sampling)
    pub fn is_last_stage(&self) -> bool {
        self.worker_position == self.total_workers - 1
    }

    // === Legacy compatibility shims for registry/loader code ===

    /// Number of columns (legacy compat — returns num_layers)
    pub fn num_columns(&self) -> u32 {
        self.num_layers()
    }

    /// Column range (legacy compat — returns layer_range)
    pub fn column_range(&self) -> (u32, u32) {
        self.layer_range()
    }
}

impl fmt::Display for ShardAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ShardAssignment(model={}, worker={}/{}, layers={}-{})",
            self.model_id,
            self.worker_position,
            self.total_workers,
            self.layer_start,
            self.layer_end
        )
    }
}

/// Static model information for the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub name: String,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub total_size_bytes: u64,
    pub min_workers: u32,
    pub recommended_workers: u32,
    pub memory_per_worker_bytes: u64,
    pub context_length: usize,
    pub format: String,
    pub shard_urls: Vec<String>,
}

/// Shard info stored in the registry (enriched assignment + runtime state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    pub assignment: ShardAssignment,
    pub download_progress: f32,
    pub path: Option<std::path::PathBuf>,
    pub hash: Option<String>,
    pub memory_bytes: u64,
    pub is_loaded: bool,
    pub last_updated: u64,
}

impl ShardInfo {
    pub fn new(assignment: ShardAssignment) -> Self {
        Self {
            assignment,
            download_progress: 0.0,
            path: None,
            hash: None,
            memory_bytes: 0,
            is_loaded: false,
            last_updated: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    pub fn set_downloaded(&mut self, path: std::path::PathBuf, hash: String) {
        self.path = Some(path);
        self.hash = Some(hash);
        self.download_progress = 1.0;
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    pub fn set_loaded(&mut self, memory_bytes: u64) {
        self.memory_bytes = memory_bytes;
        self.is_loaded = true;
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    pub fn set_unloaded(&mut self) {
        self.is_loaded = false;
        self.memory_bytes = 0;
        self.last_updated = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_worker_gets_all_layers() {
        let shard = ShardAssignment::with_layers("model".into(), 0, 1, 80);
        assert_eq!(shard.layer_start, 0);
        assert_eq!(shard.layer_end, 80);
        assert_eq!(shard.num_layers(), 80);
        assert!(shard.is_first_stage());
        assert!(shard.is_last_stage());
    }

    #[test]
    fn test_two_workers_even_split() {
        let s0 = ShardAssignment::with_layers("m".into(), 0, 2, 80);
        let s1 = ShardAssignment::with_layers("m".into(), 1, 2, 80);

        assert_eq!(s0.layer_range(), (0, 40));
        assert_eq!(s1.layer_range(), (40, 80));
        assert!(s0.is_first_stage());
        assert!(!s0.is_last_stage());
        assert!(!s1.is_first_stage());
        assert!(s1.is_last_stage());
    }

    #[test]
    fn test_three_workers_uneven_split() {
        let s0 = ShardAssignment::with_layers("m".into(), 0, 3, 80);
        let s1 = ShardAssignment::with_layers("m".into(), 1, 3, 80);
        let s2 = ShardAssignment::with_layers("m".into(), 2, 3, 80);

        // 80/3 = 26 remainder 2, so workers 0 and 1 get 27 layers each
        assert_eq!(s0.layer_range(), (0, 27));
        assert_eq!(s1.layer_range(), (27, 54));
        assert_eq!(s2.layer_range(), (54, 80));
        assert_eq!(s0.num_layers() + s1.num_layers() + s2.num_layers(), 80);
    }

    #[test]
    fn test_no_gaps_no_overlaps() {
        for total_workers in 1..=20 {
            let mut ranges: Vec<(u32, u32)> = Vec::new();
            for pos in 0..total_workers {
                let s = ShardAssignment::with_layers("m".into(), pos, total_workers, 80);
                ranges.push(s.layer_range());
            }

            assert_eq!(ranges[0].0, 0);
            assert_eq!(ranges.last().unwrap().1, 80);
            for i in 0..ranges.len() - 1 {
                assert_eq!(
                    ranges[i].1,
                    ranges[i + 1].0,
                    "Gap between workers {} and {}",
                    i,
                    i + 1
                );
            }
        }
    }

    #[test]
    fn test_contains_layer() {
        let s = ShardAssignment::with_layers("m".into(), 1, 4, 80);
        assert!(!s.contains_layer(0));
        assert!(s.contains_layer(20));
        assert!(s.contains_layer(39));
        assert!(!s.contains_layer(40));
    }

    #[test]
    fn test_more_workers_than_layers() {
        // Edge case: 100 workers for 80 layers means 20 workers get 0 layers
        let total_layers = 80u32;
        let total_workers = 100u32;
        let mut total_assigned = 0u32;

        for pos in 0..total_workers {
            let s = ShardAssignment::with_layers("m".into(), pos, total_workers, total_layers);
            total_assigned += s.num_layers();
        }
        assert_eq!(total_assigned, total_layers, "Total assigned layers must equal total layers");

        // First 80 workers should each get exactly 1 layer
        for pos in 0..80 {
            let s = ShardAssignment::with_layers("m".into(), pos, total_workers, total_layers);
            assert_eq!(s.num_layers(), 1, "Worker {} should get 1 layer", pos);
        }
        // Workers 80-99 should get 0 layers
        for pos in 80..100 {
            let s = ShardAssignment::with_layers("m".into(), pos, total_workers, total_layers);
            assert_eq!(s.num_layers(), 0, "Worker {} should get 0 layers", pos);
        }
    }

    #[test]
    fn test_single_layer_model() {
        let s = ShardAssignment::with_layers("m".into(), 0, 1, 1);
        assert_eq!(s.layer_range(), (0, 1));
        assert_eq!(s.num_layers(), 1);
        assert!(s.is_first_stage());
        assert!(s.is_last_stage());
    }

    #[test]
    fn test_zero_workers_edge_case() {
        // Should not panic
        let s = ShardAssignment::with_layers("m".into(), 0, 0, 80);
        assert_eq!(s.layer_range(), (0, 0));
        assert_eq!(s.num_layers(), 0);
    }

    #[test]
    fn test_display_format() {
        let s = ShardAssignment::with_layers("llama-70b".into(), 2, 4, 80);
        let display = format!("{}", s);
        assert!(display.contains("llama-70b"), "Display should contain model name");
        assert!(display.contains("2/4"), "Display should contain position");
    }

    #[test]
    fn test_shard_info_lifecycle() {
        let assignment = ShardAssignment::new("model".into(), 0, 10);
        let mut info = ShardInfo::new(assignment);

        assert!(!info.is_loaded);
        assert_eq!(info.download_progress, 0.0);

        info.set_downloaded(std::path::PathBuf::from("/tmp/shard"), "hash123".into());
        assert_eq!(info.download_progress, 1.0);

        info.set_loaded(7_000_000_000);
        assert!(info.is_loaded);
        assert_eq!(info.memory_bytes, 7_000_000_000);

        info.set_unloaded();
        assert!(!info.is_loaded);
    }
}
