//! Job execution module for Mesh AI compute network
//!
//! This module provides the tensor communication primitives used by the
//! production distributed inference runtime.

pub mod ring_allreduce;

pub use ring_allreduce::{Tensor, WorkerRing};
