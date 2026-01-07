//! Checkpoint module for fault tolerance in distributed inference
//!
//! This module provides checkpointing functionality that allows inference jobs
//! to be recovered after worker failures. Checkpoints are saved periodically
//! during token generation and can be used to resume inference from the last
//! saved state.
//!
//! ## Features
//!
//! - Periodic checkpoint creation (every N tokens)
//! - Checkpoint storage and retrieval
//! - Cross-worker checkpoint synchronization
//! - Automatic cleanup of old checkpoints
//!
//! ## Architecture
//!
//! ```text
//! InferenceJob State
//!         │
//!         ▼
//! ┌───────────────────┐
//! │ CheckpointManager │
//! │  • save_checkpoint│
//! │  • load_checkpoint│
//! │  • list_checkpoints│
//! │  • cleanup_old    │
//! └───────────────────┘
//!         │
//!         ▼
//! ┌───────────────────┐
//! │   CheckpointStore │
//! │  (Local filesystem│
//! │   or distributed) │
//! └───────────────────┘
//! ```

pub mod manager;
pub mod types;

pub use manager::CheckpointManager;
pub use types::{Checkpoint, CheckpointMetadata, CheckpointConfig};
