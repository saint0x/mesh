use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Phase of the ring all-reduce algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
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

/// Collective-native lane identifier used by serving sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveLane {
    ReduceScatter,
    AllGather,
    Control,
    BulkTransfer,
    Checkpoint,
}

impl CollectiveLane {
    pub fn traffic_class(self) -> TensorTrafficClass {
        match self {
            Self::Control => TensorTrafficClass::LatencyCritical,
            Self::AllGather => TensorTrafficClass::Interactive,
            Self::ReduceScatter | Self::BulkTransfer | Self::Checkpoint => TensorTrafficClass::Bulk,
        }
    }

    pub fn all_reduce_phase(self) -> Option<AllReducePhase> {
        match self {
            Self::ReduceScatter => Some(AllReducePhase::ReduceScatter),
            Self::AllGather => Some(AllReducePhase::AllGather),
            Self::Control => Some(AllReducePhase::Barrier),
            Self::BulkTransfer | Self::Checkpoint => None,
        }
    }
}

/// Slot address used by the serving dataplane receive side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServingSlotKey {
    pub session_id: Uuid,
    pub collective_id: Uuid,
    pub lane: CollectiveLane,
    pub layer_idx: u32,
    pub step: u32,
    pub slot: u32,
    pub stream_id: u32,
}

/// Fixed-size collective header for the serving-critical dataplane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingFrameHeader {
    pub session_id: Uuid,
    pub collective_id: Uuid,
    pub sender_position: u32,
    pub layer_idx: u32,
    pub step: u32,
    pub slot: u32,
    pub stream_id: u32,
    pub element_count: u32,
    pub shape_len: u32,
    pub lane: CollectiveLane,
    pub timestamp: u64,
}

impl ServingFrameHeader {
    pub fn new(
        session_id: Uuid,
        collective_id: Uuid,
        sender_position: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        lane: CollectiveLane,
        element_count: u32,
        shape_len: u32,
    ) -> Self {
        Self {
            session_id,
            collective_id,
            sender_position,
            layer_idx,
            step,
            slot,
            stream_id,
            element_count,
            shape_len,
            lane,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub const fn fixed_size() -> usize {
        72
    }

    pub fn slot_key(&self) -> ServingSlotKey {
        ServingSlotKey {
            session_id: self.session_id,
            collective_id: self.collective_id,
            lane: self.lane,
            layer_idx: self.layer_idx,
            step: self.step,
            slot: self.slot,
            stream_id: self.stream_id,
        }
    }

    pub fn size_bytes(&self) -> usize {
        Self::fixed_size()
            + self.element_count as usize * std::mem::size_of::<f32>()
            + self.shape_len as usize * std::mem::size_of::<u64>()
    }

    pub fn encode_binary(&self) -> [u8; Self::fixed_size()] {
        let mut bytes = [0u8; Self::fixed_size()];
        let mut offset = 0usize;

        fn put(bytes: &mut [u8], offset: &mut usize, value: &[u8]) {
            bytes[*offset..*offset + value.len()].copy_from_slice(value);
            *offset += value.len();
        }

        put(&mut bytes, &mut offset, self.session_id.as_bytes());
        put(&mut bytes, &mut offset, self.collective_id.as_bytes());
        put(&mut bytes, &mut offset, &self.sender_position.to_be_bytes());
        put(&mut bytes, &mut offset, &self.layer_idx.to_be_bytes());
        put(&mut bytes, &mut offset, &self.step.to_be_bytes());
        put(&mut bytes, &mut offset, &self.slot.to_be_bytes());
        put(&mut bytes, &mut offset, &self.stream_id.to_be_bytes());
        put(&mut bytes, &mut offset, &self.element_count.to_be_bytes());
        put(&mut bytes, &mut offset, &self.shape_len.to_be_bytes());
        bytes[offset] = match self.lane {
            CollectiveLane::ReduceScatter => 0,
            CollectiveLane::AllGather => 1,
            CollectiveLane::Control => 2,
            CollectiveLane::BulkTransfer => 3,
            CollectiveLane::Checkpoint => 4,
        };
        offset += 4;
        put(&mut bytes, &mut offset, &self.timestamp.to_be_bytes());
        bytes
    }

    pub fn decode_binary(bytes: &[u8]) -> std::io::Result<Self> {
        fn take<const N: usize>(bytes: &[u8], offset: &mut usize) -> std::io::Result<[u8; N]> {
            if *offset + N > bytes.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "serving frame header ended early",
                ));
            }
            let mut out = [0u8; N];
            out.copy_from_slice(&bytes[*offset..*offset + N]);
            *offset += N;
            Ok(out)
        }

        if bytes.len() != Self::fixed_size() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "serving frame header size {} did not match {}",
                    bytes.len(),
                    Self::fixed_size()
                ),
            ));
        }

        let mut offset = 0usize;
        let session_id = Uuid::from_bytes(take::<16>(bytes, &mut offset)?);
        let collective_id = Uuid::from_bytes(take::<16>(bytes, &mut offset)?);
        let sender_position = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let layer_idx = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let step = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let slot = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let stream_id = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let element_count = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let shape_len = u32::from_be_bytes(take::<4>(bytes, &mut offset)?);
        let lane = match take::<1>(bytes, &mut offset)?[0] {
            0 => CollectiveLane::ReduceScatter,
            1 => CollectiveLane::AllGather,
            2 => CollectiveLane::Control,
            3 => CollectiveLane::BulkTransfer,
            4 => CollectiveLane::Checkpoint,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("invalid collective lane tag {}", other),
                ));
            }
        };
        offset += 3;
        let timestamp = u64::from_be_bytes(take::<8>(bytes, &mut offset)?);

        Ok(Self {
            session_id,
            collective_id,
            sender_position,
            layer_idx,
            step,
            slot,
            stream_id,
            element_count,
            shape_len,
            lane,
            timestamp,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServingFrame {
    pub header: ServingFrameHeader,
    pub chunk_data: Vec<f32>,
    pub chunk_shape: Vec<usize>,
}

impl ServingFrame {
    pub fn phase(&self) -> Option<AllReducePhase> {
        self.header.lane.all_reduce_phase()
    }
}
