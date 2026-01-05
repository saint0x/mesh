//! Job execution module for Mesh AI compute network
//!
//! This module provides workload executors for different AI tasks:
//! - Embeddings: Text embedding generation using ONNX models
//! - Ring All-Reduce: Distributed gradient aggregation for training
//! - OCR: Optical character recognition (future)
//! - Chat: Language model inference (future)
//!
//! The job_runner module integrates executors with the network layer.

pub mod embeddings;
pub mod job_runner;
pub mod ring_allreduce;
pub mod types;

pub use embeddings::EmbeddingsExecutor;
pub use job_runner::{JobRunner, JobStats};
pub use ring_allreduce::{AllReducePhase, Tensor, TensorMessage, WorkerRing};
pub use types::{EmbeddingsInput, EmbeddingsOutput, ExecutorError, ExecutorResult};
