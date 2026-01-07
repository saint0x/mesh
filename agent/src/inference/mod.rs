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

pub mod coordinator;
pub mod job;
pub mod stats;

pub use coordinator::{InferenceCoordinator, InferenceConfig};
pub use job::{InferenceJob, InferenceRequest, InferenceResult, GenerationConfig};
pub use stats::InferenceStats;
