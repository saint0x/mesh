//! Model shard management module
//!
//! This module provides infrastructure for managing model shards in a
//! tensor-parallel distributed inference setup. Each worker is responsible
//! for a specific column range of the model weights.
//!
//! ## Key Concepts
//!
//! - **Shard**: A contiguous column range of model weight matrices
//! - **Total Columns**: 8192 (fixed for the shard space)
//! - **Column Range**: The specific columns this worker owns
//!
//! ## Architecture
//!
//! ```text
//! Full Model Matrix (hidden_dim × 8192)
//! ┌────────────────────────────────────────────────────┐
//! │ Worker 0  │ Worker 1  │ Worker 2  │ ... │ Worker N │
//! │ cols 0-819│ 820-1639  │ 1640-2459 │     │ 7373-8191│
//! └────────────────────────────────────────────────────┘
//! ```
//!
//! This module does NOT handle actual model loading - that will be done
//! when we integrate with safetensors/GGML. This provides the metadata
//! and coordination layer.

pub mod shard;
pub mod registry;

pub use shard::{ShardInfo, ShardAssignment, ModelInfo};
pub use registry::{ShardRegistry, ShardStatus};
