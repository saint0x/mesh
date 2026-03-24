use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Phase of the ring all-reduce algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllReducePhase {
    ReduceScatter,
    AllGather,
    Barrier,
}

/// Tensor payload exchanged on the dedicated tensor data plane.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorMessage {
    /// Job ID this tensor belongs to.
    pub job_id: Uuid,
    /// Layer index in the model.
    pub layer_idx: u32,
    /// Current phase of the all-reduce algorithm.
    pub phase: AllReducePhase,
    /// Step number within the current phase.
    pub step: u32,
    /// Tensor chunk data.
    pub chunk_data: Vec<f32>,
    /// Shape of the chunk.
    pub chunk_shape: Vec<usize>,
    /// Unix timestamp when the message was created.
    pub timestamp: u64,
}

impl TensorMessage {
    /// Special step number indicating a barrier synchronization message.
    pub const BARRIER_STEP: u32 = 0xFFFF_FFFF;

    pub fn new(
        job_id: Uuid,
        layer_idx: u32,
        phase: AllReducePhase,
        step: u32,
        chunk_data: Vec<f32>,
        chunk_shape: Vec<usize>,
    ) -> Self {
        Self {
            job_id,
            layer_idx,
            phase,
            step,
            chunk_data,
            chunk_shape,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn is_barrier(&self) -> bool {
        self.step == Self::BARRIER_STEP
    }

    pub fn size_bytes(&self) -> usize {
        let fixed_size = 16 + 4 + 1 + 4 + 8;
        let chunk_data_size = self.chunk_data.len() * std::mem::size_of::<f32>();
        let chunk_shape_size = self.chunk_shape.len() * std::mem::size_of::<usize>();
        fixed_size + chunk_data_size + chunk_shape_size
    }
}
