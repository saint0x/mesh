use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Phase of the ring all-reduce algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllReducePhase {
    ReduceScatter,
    AllGather,
    Barrier,
}

/// Transport class used by the tensor plane to prioritize latency-sensitive
/// traffic ahead of bulk transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorTrafficClass {
    LatencyCritical,
    Interactive,
    Bulk,
}

impl TensorTrafficClass {
    pub fn is_bulk(self) -> bool {
        matches!(self, Self::Bulk)
    }
}

/// Tensor payload exchanged on the dedicated tensor data plane.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorMessage {
    /// Sender ring position for protocol-level validation.
    pub sender_position: u32,
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
        sender_position: u32,
        job_id: Uuid,
        layer_idx: u32,
        phase: AllReducePhase,
        step: u32,
        chunk_data: Vec<f32>,
        chunk_shape: Vec<usize>,
    ) -> Self {
        Self {
            sender_position,
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

    pub fn traffic_class(&self) -> TensorTrafficClass {
        if self.is_barrier() {
            return TensorTrafficClass::LatencyCritical;
        }

        match self.phase {
            AllReducePhase::Barrier => TensorTrafficClass::LatencyCritical,
            AllReducePhase::AllGather => TensorTrafficClass::Interactive,
            AllReducePhase::ReduceScatter => TensorTrafficClass::Bulk,
        }
    }

    pub fn size_bytes(&self) -> usize {
        let fixed_size = 4 + 16 + 4 + 1 + 4 + 8;
        let chunk_data_size = self.chunk_data.len() * std::mem::size_of::<f32>();
        let chunk_shape_size = self.chunk_shape.len() * std::mem::size_of::<usize>();
        fixed_size + chunk_data_size + chunk_shape_size
    }
}
