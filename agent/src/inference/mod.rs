//! Inference module for tensor-parallel distributed inference
//!
//! This module provides the orchestration layer for running distributed inference
//! across a ring of workers using tensor parallelism. Each worker:
//!
//! 1. Holds a shard (column range) of the model weights
//! 2. Computes partial matrix multiplications for their columns
//! 3. Participates in ring all-reduce to combine results
//! 4. Produces identical full activations across all workers
//!
//! ## Architecture
//!
//! ```text
//! Control Plane
//!       │
//!       ▼ (inference request)
//! ┌─────────────────────────────────────────┐
//! │         InferenceCoordinator            │
//! │  • Receives jobs from control plane     │
//! │  • Manages inference lifecycle          │
//! │  • Coordinates checkpointing            │
//! └─────────────────────────────────────────┘
//!       │
//!       ▼ (per layer)
//! ┌─────────────────────────────────────────┐
//! │         Forward Pass (per layer)        │
//! │  1. Compute partial matmul (my shard)   │
//! │  2. Ring all-reduce with neighbors      │
//! │  3. Apply activation function           │
//! │  4. Repeat for next layer               │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`InferenceCoordinator`]: Main orchestrator for inference jobs
//! - [`InferenceJob`]: Represents a single inference request
//! - [`InferenceStats`]: Tracks inference performance metrics
//! - [`tensor_ops`]: Tensor operations (matmul, activations, etc.)
//! - [`kv_cache`]: KV cache management for transformer attention
//! - [`forward_pass`]: Tensor-parallel forward pass implementation

pub mod coordinator;
pub mod forward_pass;
pub mod job;
pub mod kv_cache;
pub mod mock_loader; // MOCK: Mock shard loader for validation
pub mod mock_validation; // MOCK: For validation only - TODO: Remove when using real weights
pub mod stats;
pub mod tensor_ops;

pub use coordinator::{InferenceCoordinator, InferenceConfig};
pub use forward_pass::{ForwardPass, LayerWeights, ModelWeights};
pub use job::{GenerationConfig, InferenceJob, InferenceRequest, InferenceResult};
pub use kv_cache::{KVCache, KVCacheConfig, LayerKVCache};
pub use mock_loader::{MockShardLoader, ShardLoader}; // Export loader trait and mock impl
pub use stats::InferenceStats;
pub use tensor_ops::{Tensor1D, Tensor2D};
