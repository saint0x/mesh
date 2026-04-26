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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorChunkHeader {
    pub sender_position: u32,
    pub job_id: Uuid,
    pub layer_idx: u32,
    pub phase: AllReducePhase,
    pub step: u32,
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
        TensorChunkHeader::fixed_size()
            + 4
            + self.chunk_data.len() * std::mem::size_of::<f32>()
            + 4
            + self.chunk_shape.len() * std::mem::size_of::<u64>()
    }

    pub fn encode_binary(&self) -> Vec<u8> {
        self.header()
            .encode_binary(&self.chunk_data, &self.chunk_shape)
    }

    pub fn decode_binary(bytes: &[u8]) -> std::io::Result<Self> {
        fn take<const N: usize>(bytes: &[u8], offset: &mut usize) -> std::io::Result<[u8; N]> {
            if *offset + N > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "tensor message ended early",
                ));
            }
            let mut out = [0u8; N];
            out.copy_from_slice(&bytes[*offset..*offset + N]);
            *offset += N;
            Ok(out)
        }

        let mut offset = 0usize;
        let sender_position = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let job_id = Uuid::from_bytes(take::<16>(bytes, &mut offset)?);
        let layer_idx = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let phase = match take::<1>(bytes, &mut offset)?[0] {
            0 => AllReducePhase::ReduceScatter,
            1 => AllReducePhase::AllGather,
            2 => AllReducePhase::Barrier,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid all-reduce phase tag {}", other),
                ));
            }
        };
        let step = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let timestamp = u64::from_be_bytes(take::<8>(bytes, &mut offset)?);
        let chunk_len = u32::from_be_bytes(take::<4>(bytes, &mut offset)?) as usize;
        let mut chunk_data = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            chunk_data.push(f32::from_bits(u32::from_be_bytes(take::<4>(
                bytes,
                &mut offset,
            )?)));
        }
        let shape_len = u32::from_be_bytes(take::<4>(bytes, &mut offset)?) as usize;
        let mut chunk_shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let dim = u64::from_be_bytes(take::<8>(bytes, &mut offset)?) as usize;
            chunk_shape.push(dim);
        }
        if offset != bytes.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tensor message contained trailing bytes",
            ));
        }

        Ok(Self {
            sender_position,
            job_id,
            layer_idx,
            phase,
            step,
            chunk_data,
            chunk_shape,
            timestamp,
        })
    }

    pub fn header(&self) -> TensorChunkHeader {
        TensorChunkHeader {
            sender_position: self.sender_position,
            job_id: self.job_id,
            layer_idx: self.layer_idx,
            phase: self.phase,
            step: self.step,
            timestamp: self.timestamp,
        }
    }
}

impl TensorChunkHeader {
    pub fn new(
        sender_position: u32,
        job_id: Uuid,
        layer_idx: u32,
        phase: AllReducePhase,
        step: u32,
    ) -> Self {
        Self {
            sender_position,
            job_id,
            layer_idx,
            phase,
            step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub const fn fixed_size() -> usize {
        4 + 16 + 4 + 1 + 4 + 8
    }

    pub fn size_bytes(&self, chunk_data_len: usize, chunk_shape_len: usize) -> usize {
        Self::fixed_size()
            + 4
            + chunk_data_len * std::mem::size_of::<f32>()
            + 4
            + chunk_shape_len * std::mem::size_of::<u64>()
    }

    pub fn encode_binary(&self, chunk_data: &[f32], chunk_shape: &[usize]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.size_bytes(chunk_data.len(), chunk_shape.len()));
        bytes.extend_from_slice(&self.sender_position.to_be_bytes());
        bytes.extend_from_slice(self.job_id.as_bytes());
        bytes.extend_from_slice(&self.layer_idx.to_be_bytes());
        bytes.push(match self.phase {
            AllReducePhase::ReduceScatter => 0,
            AllReducePhase::AllGather => 1,
            AllReducePhase::Barrier => 2,
        });
        bytes.extend_from_slice(&self.step.to_be_bytes());
        bytes.extend_from_slice(&self.timestamp.to_be_bytes());
        bytes.extend_from_slice(&(chunk_data.len() as u32).to_be_bytes());
        for value in chunk_data {
            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        bytes.extend_from_slice(&(chunk_shape.len() as u32).to_be_bytes());
        for dim in chunk_shape {
            bytes.extend_from_slice(&(*dim as u64).to_be_bytes());
        }
        bytes
    }
}
