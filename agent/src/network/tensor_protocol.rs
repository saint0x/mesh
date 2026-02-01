//! Pipeline Activation Protocol for Distributed Inference
//!
//! This protocol carries hidden-state activations between pipeline stages.
//! In pipeline parallelism, each worker runs a contiguous range of layers
//! and sends the resulting activations to the next stage.
//!
//! Every message carries full distributed tracing context so that the
//! end-to-end path of a request can be reconstructed across all nodes.

use futures::prelude::*;
use libp2p::request_response::{self, Behaviour, Config, ProtocolSupport};
use libp2p::StreamProtocol;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Protocol identifier
pub const TENSOR_PROTOCOL_ID: StreamProtocol = StreamProtocol::new("/mesh/pipeline/1.0.0");

/// Phase of the pipeline communication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelinePhase {
    /// Forward pass: activations flowing from stage N to stage N+1
    ForwardActivation,
    /// Token broadcast: sampled token flowing from last stage back to all stages
    TokenBroadcast,
    /// Barrier synchronization between stages
    Barrier,
}

// Keep the old name as an alias so existing code outside this crate compiles
pub type AllReducePhase = PipelinePhase;

/// Message sent between pipeline stages
///
/// Every message carries a full trace context so operations can be correlated
/// across all devices in the pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMessage {
    // === Distributed Tracing Context ===

    /// Unique job identifier (trace root) — assigned when the request enters the system
    pub job_id: Uuid,

    /// Span ID for this specific operation (unique per send)
    pub span_id: u64,

    /// Span ID of the parent operation (the sender's current span)
    pub parent_span_id: u64,

    /// Monotonic sequence number within a job (0, 1, 2, ...) for ordering verification
    pub sequence_num: u64,

    // === Pipeline Metadata ===

    /// Which layer's output this activation represents (the last layer the sender processed)
    pub layer_idx: u32,

    /// Pipeline phase
    pub phase: PipelinePhase,

    /// Token index in the generation sequence (which token step we're on)
    pub token_idx: u32,

    /// Position of the sender in the pipeline
    pub sender_stage: u32,

    // === Payload ===

    /// Activation data (flattened f32 tensor)
    pub activation_data: Vec<f32>,

    /// Shape of the activation tensor (e.g., [seq_len, hidden_dim])
    pub activation_shape: Vec<usize>,

    /// SHA-256 checksum of activation_data for divergence detection (first 8 bytes)
    pub checksum: [u8; 8],

    // === Legacy/compat fields ===

    /// Timestamp when this message was created (millis since epoch)
    pub timestamp: u64,

    /// Step within phase (legacy compat for ring all-reduce callers)
    pub step: u32,
}

impl TensorMessage {
    /// Sentinel step value used for barrier messages
    pub const BARRIER_STEP: u32 = 0xFFFFFFFF;

    /// Create a new pipeline activation message with tracing context
    pub fn new_activation(
        job_id: Uuid,
        layer_idx: u32,
        token_idx: u32,
        sender_stage: u32,
        sequence_num: u64,
        parent_span_id: u64,
        data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Self {
        let checksum = Self::compute_checksum(&data);
        let span_id = Self::generate_span_id();

        Self {
            job_id,
            span_id,
            parent_span_id,
            sequence_num,
            layer_idx,
            phase: PipelinePhase::ForwardActivation,
            token_idx,
            sender_stage,
            activation_data: data,
            activation_shape: shape,
            checksum,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            step: 0,
        }
    }

    /// Create a token broadcast message (last stage → all stages)
    pub fn new_token_broadcast(
        job_id: Uuid,
        token_idx: u32,
        sender_stage: u32,
        sequence_num: u64,
        parent_span_id: u64,
        token_id: u32,
    ) -> Self {
        let data = vec![token_id as f32];
        let shape = vec![1];
        let checksum = Self::compute_checksum(&data);

        Self {
            job_id,
            span_id: Self::generate_span_id(),
            parent_span_id,
            sequence_num,
            layer_idx: u32::MAX,
            phase: PipelinePhase::TokenBroadcast,
            token_idx,
            sender_stage,
            activation_data: data,
            activation_shape: shape,
            checksum,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            step: 0,
        }
    }

    /// Create a legacy-format message (backward compat with ring all-reduce callers)
    pub fn new(
        job_id: Uuid,
        layer_idx: u32,
        phase: PipelinePhase,
        step: u32,
        chunk_data: Vec<f32>,
        chunk_shape: Vec<usize>,
    ) -> Self {
        let checksum = Self::compute_checksum(&chunk_data);

        Self {
            job_id,
            span_id: Self::generate_span_id(),
            parent_span_id: 0,
            sequence_num: 0,
            layer_idx,
            phase,
            token_idx: 0,
            sender_stage: 0,
            activation_data: chunk_data,
            activation_shape: chunk_shape,
            checksum,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            step,
        }
    }

    /// Check if this is a barrier message
    pub fn is_barrier(&self) -> bool {
        self.phase == PipelinePhase::Barrier && self.step == Self::BARRIER_STEP
    }

    /// Verify the checksum of the activation data
    pub fn verify_checksum(&self) -> bool {
        let expected = Self::compute_checksum(&self.activation_data);
        self.checksum == expected
    }

    /// Compute SHA-256 checksum (first 8 bytes) of f32 data
    pub fn compute_checksum(data: &[f32]) -> [u8; 8] {
        let mut hasher = Sha256::new();
        for val in data {
            hasher.update(val.to_le_bytes());
        }
        let hash = hasher.finalize();
        let mut result = [0u8; 8];
        result.copy_from_slice(&hash[..8]);
        result
    }

    /// Generate a unique span ID
    fn generate_span_id() -> u64 {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        // XOR with random bits for uniqueness even at nanosecond resolution
        ts ^ (rand::random::<u64>() >> 16)
    }

    // === Legacy accessor shims ===

    /// Legacy: access activation_data as chunk_data
    pub fn chunk_data(&self) -> &[f32] {
        &self.activation_data
    }

    /// Legacy: access activation_shape as chunk_shape
    pub fn chunk_shape(&self) -> &[usize] {
        &self.activation_shape
    }
}

/// CBOR codec for TensorMessage (length-prefixed)
#[derive(Debug, Clone, Default)]
pub struct TensorCodec;

/// Maximum message size: 100 MB (activations can be large for long sequences)
const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024;

#[async_trait::async_trait]
impl request_response::Codec for TensorCodec {
    type Protocol = StreamProtocol;
    type Request = TensorMessage;
    type Response = TensorMessage;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_message(io).await
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_message(io).await
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_message(io, &req).await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        write_message(io, &res).await
    }
}

async fn read_message<T: AsyncRead + Unpin>(
    io: &mut T,
) -> io::Result<TensorMessage> {
    use tokio::io::AsyncReadExt;

    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Message too large: {} bytes (max {})", len, MAX_MESSAGE_SIZE),
        ));
    }

    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;

    ciborium::de::from_reader(&buf[..]).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("CBOR decode: {}", e))
    })
}

async fn write_message<T: AsyncWrite + Unpin>(
    io: &mut T,
    msg: &TensorMessage,
) -> io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let mut buf = Vec::new();
    ciborium::ser::into_writer(msg, &mut buf).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("CBOR encode: {}", e))
    })?;

    let len = buf.len() as u32;
    io.write_all(&len.to_be_bytes()).await?;
    io.write_all(&buf).await?;

    Ok(())
}

/// TensorProtocol type alias
pub type TensorProtocol = request_response::Behaviour<TensorCodec>;

/// Configuration for tensor protocol
#[derive(Debug, Clone)]
pub struct TensorProtocolConfig {
    pub max_message_size: usize,
}

impl Default for TensorProtocolConfig {
    fn default() -> Self {
        Self {
            max_message_size: MAX_MESSAGE_SIZE,
        }
    }
}

/// Create a new tensor protocol behaviour
pub fn new_tensor_protocol(config: TensorProtocolConfig) -> TensorProtocol {
    let protocols = std::iter::once((TENSOR_PROTOCOL_ID, ProtocolSupport::Full));
    let mut req_resp_config = Config::default();
    req_resp_config.set_request_timeout(Duration::from_secs(30));

    Behaviour::with_codec(TensorCodec, protocols, req_resp_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_activation_message() {
        let job_id = Uuid::new_v4();
        let msg = TensorMessage::new_activation(
            job_id,
            5,
            0,
            1,
            42,
            0,
            vec![1.0, 2.0, 3.0],
            vec![1, 3],
        );

        assert_eq!(msg.job_id, job_id);
        assert_eq!(msg.layer_idx, 5);
        assert_eq!(msg.phase, PipelinePhase::ForwardActivation);
        assert_eq!(msg.token_idx, 0);
        assert_eq!(msg.sender_stage, 1);
        assert_eq!(msg.sequence_num, 42);
        assert!(msg.span_id != 0);
        assert!(msg.verify_checksum());
    }

    #[test]
    fn test_token_broadcast_message() {
        let job_id = Uuid::new_v4();
        let msg = TensorMessage::new_token_broadcast(job_id, 5, 3, 100, 0, 42);

        assert_eq!(msg.phase, PipelinePhase::TokenBroadcast);
        assert_eq!(msg.activation_data, vec![42.0]);
        assert!(msg.verify_checksum());
    }

    #[test]
    fn test_checksum_verification() {
        let msg = TensorMessage::new_activation(
            Uuid::new_v4(),
            0,
            0,
            0,
            0,
            0,
            vec![1.0, 2.0, 3.0],
            vec![3],
        );

        assert!(msg.verify_checksum());

        let mut corrupted = msg.clone();
        corrupted.activation_data[0] = 999.0;
        assert!(!corrupted.verify_checksum());
    }

    #[test]
    fn test_legacy_new_compat() {
        let msg = TensorMessage::new(
            Uuid::new_v4(),
            5,
            PipelinePhase::Barrier,
            TensorMessage::BARRIER_STEP,
            vec![0.0],
            vec![1],
        );
        assert!(msg.is_barrier());
    }

    #[test]
    fn test_cbor_roundtrip() {
        let msg = TensorMessage::new_activation(
            Uuid::new_v4(),
            10,
            3,
            1,
            99,
            55,
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        );

        let mut buf = Vec::new();
        ciborium::ser::into_writer(&msg, &mut buf).unwrap();
        let decoded: TensorMessage = ciborium::de::from_reader(&buf[..]).unwrap();

        assert_eq!(decoded.job_id, msg.job_id);
        assert_eq!(decoded.layer_idx, msg.layer_idx);
        assert_eq!(decoded.token_idx, msg.token_idx);
        assert_eq!(decoded.sender_stage, msg.sender_stage);
        assert_eq!(decoded.sequence_num, msg.sequence_num);
        assert_eq!(decoded.activation_data, msg.activation_data);
        assert_eq!(decoded.checksum, msg.checksum);
    }
}
