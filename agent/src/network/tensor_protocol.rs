// Tensor protocol implementation for ring all-reduce communication over libp2p

use futures::prelude::*;
use libp2p::request_response::{self, Behaviour, Config, ProtocolSupport};
use libp2p::StreamProtocol;
use serde::{Deserialize, Serialize};
use std::io;
use std::time::Duration;
use uuid::Uuid;

/// Protocol ID for mesh tensor communication
pub const TENSOR_PROTOCOL_ID: StreamProtocol = StreamProtocol::new("/mesh/tensor/1.0.0");

/// Maximum message size for tensor messages (10MB)
pub const MESSAGE_SIZE_LIMIT: usize = 10 * 1024 * 1024;

/// Phase of the ring all-reduce algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllReducePhase {
    /// Reduce-scatter phase: each node sends a chunk to the next node
    /// and accumulates chunks received from the previous node
    ReduceScatter,
    /// All-gather phase: each node broadcasts its fully reduced chunk
    /// around the ring so all nodes end up with the complete result
    AllGather,
    /// Barrier synchronization
    Barrier,
}

/// Tensor message for ring all-reduce communication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorMessage {
    /// Job ID this tensor belongs to
    pub job_id: Uuid,
    /// Layer index in the model
    pub layer_idx: u32,
    /// Current phase of the all-reduce algorithm
    pub phase: AllReducePhase,
    /// Step number within the current phase
    pub step: u32,
    /// Tensor chunk data (f32 values)
    pub chunk_data: Vec<f32>,
    /// Shape of the chunk
    pub chunk_shape: Vec<usize>,
    /// Unix timestamp when the message was created
    pub timestamp: u64,
}

impl TensorMessage {
    /// Special step number indicating a barrier synchronization message
    pub const BARRIER_STEP: u32 = 0xFFFFFFFF;

    /// Create a new tensor message with current timestamp
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

    /// Check if this is a barrier message
    pub fn is_barrier(&self) -> bool {
        self.step == Self::BARRIER_STEP
    }

    /// Calculate the approximate size of this message in bytes.
    ///
    /// Note: This is an estimate based on raw field sizes, not the actual
    /// CBOR-encoded size which may differ due to encoding overhead.
    pub fn size_bytes(&self) -> usize {
        // UUID (16) + layer_idx (4) + phase (1) + step (4) + timestamp (8)
        // + chunk_data (len * 4) + chunk_shape (len * 8) + vec overhead
        let fixed_size = 16 + 4 + 1 + 4 + 8;
        let chunk_data_size = self.chunk_data.len() * std::mem::size_of::<f32>();
        let chunk_shape_size = self.chunk_shape.len() * std::mem::size_of::<usize>();
        fixed_size + chunk_data_size + chunk_shape_size
    }
}

/// CBOR codec for tensor protocol messages
#[derive(Debug, Clone, Default)]
pub struct TensorCodec;

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
        read_cbor_message(io).await
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_cbor_message(io).await
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
        write_cbor_message(io, &req).await
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
        write_cbor_message(io, &res).await
    }
}

/// Read a length-prefixed CBOR message from an async stream
async fn read_cbor_message<T, M>(io: &mut T) -> io::Result<M>
where
    T: AsyncRead + Unpin + Send,
    M: for<'de> Deserialize<'de>,
{
    // Read u32 length prefix (big-endian)
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    // Enforce size limit
    if len > MESSAGE_SIZE_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Message size {} exceeds limit {}", len, MESSAGE_SIZE_LIMIT),
        ));
    }

    // Read CBOR payload
    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;

    // Deserialize CBOR
    ciborium::from_reader(&buf[..]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Write a length-prefixed CBOR message to an async stream
async fn write_cbor_message<T, M>(io: &mut T, message: &M) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
    M: Serialize,
{
    // Serialize to CBOR
    let mut buf = Vec::new();
    ciborium::into_writer(message, &mut buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Check size limit
    if buf.len() > MESSAGE_SIZE_LIMIT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Message size {} exceeds limit {}",
                buf.len(),
                MESSAGE_SIZE_LIMIT
            ),
        ));
    }

    // Write u32 length prefix (big-endian)
    let len = buf.len() as u32;
    io.write_all(&len.to_be_bytes()).await?;

    // Write CBOR payload
    io.write_all(&buf).await?;
    io.flush().await?;

    Ok(())
}

/// Configuration for the tensor protocol behavior
#[derive(Debug, Clone)]
pub struct TensorProtocolConfig {
    /// Request timeout (short for tensor communication)
    pub request_timeout: Duration,
}

impl Default for TensorProtocolConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(2), // Short timeout for tensor exchange
        }
    }
}

/// Type alias for tensor protocol behavior
pub type TensorProtocol = Behaviour<TensorCodec>;

/// Create a new tensor protocol behavior
pub fn new_tensor_protocol(config: TensorProtocolConfig) -> TensorProtocol {
    let protocols = std::iter::once((TENSOR_PROTOCOL_ID, ProtocolSupport::Full));

    let mut cfg = Config::default();
    cfg = cfg.with_request_timeout(config.request_timeout);

    Behaviour::new(protocols, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::io::Cursor;

    #[test]
    fn test_tensor_message_serialization() {
        let msg = TensorMessage {
            job_id: Uuid::new_v4(),
            layer_idx: 5,
            phase: AllReducePhase::ReduceScatter,
            step: 3,
            chunk_data: vec![1.0, 2.0, 3.0, 4.0],
            chunk_shape: vec![2, 2],
            timestamp: 1234567890,
        };

        // Serialize to CBOR
        let mut buf = Vec::new();
        ciborium::into_writer(&msg, &mut buf).unwrap();

        // Deserialize from CBOR
        let decoded: TensorMessage = ciborium::from_reader(&buf[..]).unwrap();

        assert_eq!(msg, decoded);
    }

    #[test]
    fn test_all_reduce_phase_serialization() {
        // Test ReduceScatter
        let phase = AllReducePhase::ReduceScatter;
        let mut buf = Vec::new();
        ciborium::into_writer(&phase, &mut buf).unwrap();
        let decoded: AllReducePhase = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(phase, decoded);

        // Test AllGather
        let phase = AllReducePhase::AllGather;
        let mut buf = Vec::new();
        ciborium::into_writer(&phase, &mut buf).unwrap();
        let decoded: AllReducePhase = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(phase, decoded);
    }

    #[test]
    fn test_tensor_message_new() {
        let job_id = Uuid::new_v4();
        let msg = TensorMessage::new(
            job_id,
            1,
            AllReducePhase::AllGather,
            0,
            vec![1.0, 2.0, 3.0],
            vec![3],
        );

        assert_eq!(msg.job_id, job_id);
        assert_eq!(msg.layer_idx, 1);
        assert_eq!(msg.phase, AllReducePhase::AllGather);
        assert_eq!(msg.step, 0);
        assert_eq!(msg.chunk_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(msg.chunk_shape, vec![3]);
        assert!(msg.timestamp > 0);
    }

    #[test]
    fn test_tensor_message_size_bytes() {
        let msg = TensorMessage {
            job_id: Uuid::new_v4(),
            layer_idx: 0,
            phase: AllReducePhase::ReduceScatter,
            step: 0,
            chunk_data: vec![1.0, 2.0, 3.0, 4.0], // 4 * 4 = 16 bytes
            chunk_shape: vec![2, 2],              // 2 * 8 = 16 bytes
            timestamp: 0,
        };

        let size = msg.size_bytes();
        // fixed (33) + chunk_data (16) + chunk_shape (16) = 65
        assert!(size > 0);
        assert!(size >= 33 + 16); // At minimum
    }

    #[test]
    fn test_tensor_message_different_shapes() {
        // Test with 1D tensor
        let msg_1d = TensorMessage::new(
            Uuid::new_v4(),
            0,
            AllReducePhase::ReduceScatter,
            0,
            vec![1.0; 100],
            vec![100],
        );

        let mut buf = Vec::new();
        ciborium::into_writer(&msg_1d, &mut buf).unwrap();
        let decoded: TensorMessage = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(msg_1d, decoded);

        // Test with 3D tensor shape
        let msg_3d = TensorMessage::new(
            Uuid::new_v4(),
            1,
            AllReducePhase::AllGather,
            5,
            vec![2.0; 8],
            vec![2, 2, 2],
        );

        let mut buf = Vec::new();
        ciborium::into_writer(&msg_3d, &mut buf).unwrap();
        let decoded: TensorMessage = ciborium::from_reader(&buf[..]).unwrap();
        assert_eq!(msg_3d, decoded);
    }

    #[test]
    fn test_protocol_id() {
        assert_eq!(TENSOR_PROTOCOL_ID.as_ref(), "/mesh/tensor/1.0.0");
    }

    #[test]
    fn test_default_config() {
        let config = TensorProtocolConfig::default();
        assert_eq!(config.request_timeout, Duration::from_secs(2));
    }

    #[tokio::test]
    async fn test_cbor_roundtrip() {
        let msg = TensorMessage::new(
            Uuid::new_v4(),
            3,
            AllReducePhase::ReduceScatter,
            7,
            vec![1.5, 2.5, 3.5],
            vec![3],
        );

        // Write to buffer
        let mut write_buf = Vec::new();
        write_cbor_message(&mut write_buf, &msg).await.unwrap();

        // Read from buffer
        let mut read_buf = Cursor::new(write_buf);
        let decoded: TensorMessage = read_cbor_message(&mut read_buf).await.unwrap();

        assert_eq!(msg, decoded);
    }

    #[tokio::test]
    async fn test_message_size_limit() {
        // Create a message that exceeds the size limit (>10MB)
        // CBOR uses half-precision floats for 0.0, so we need enough data
        // 10MB / 4 bytes per f32 = ~2.5M floats, but CBOR is more compact for zeros
        // Use non-zero values and more data to ensure we exceed 10MB
        let large_msg = TensorMessage {
            job_id: Uuid::new_v4(),
            layer_idx: 0,
            phase: AllReducePhase::ReduceScatter,
            step: 0,
            chunk_data: vec![1.234_567_9; 4_000_000], // Should exceed 10MB with non-zero values
            chunk_shape: vec![4_000_000],
            timestamp: 0,
        };

        let mut buf = Vec::new();
        let result = write_cbor_message(&mut buf, &large_msg).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn test_read_size_limit_enforcement() {
        // Create a valid message
        let msg = TensorMessage::new(
            Uuid::new_v4(),
            0,
            AllReducePhase::AllGather,
            0,
            vec![1.0, 2.0],
            vec![2],
        );

        // Write it normally
        let mut write_buf = Vec::new();
        write_cbor_message(&mut write_buf, &msg).await.unwrap();

        // Tamper with the length prefix to be larger than limit
        let large_len: u32 = (MESSAGE_SIZE_LIMIT + 1) as u32;
        write_buf[0..4].copy_from_slice(&large_len.to_be_bytes());

        // Try to read it (should fail)
        let mut read_buf = Cursor::new(write_buf);
        let result: io::Result<TensorMessage> = read_cbor_message(&mut read_buf).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn test_malformed_cbor() {
        // Create a buffer with valid length prefix but invalid CBOR data
        let mut buf = Vec::new();
        let len: u32 = 10;
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

        let mut read_buf = Cursor::new(buf);
        let result: io::Result<TensorMessage> = read_cbor_message(&mut read_buf).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn test_incomplete_read() {
        // Create a buffer with length prefix indicating more data than available
        let mut buf = Vec::new();
        let len: u32 = 100;
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&[0x01, 0x02, 0x03]); // Only 3 bytes, not 100

        let mut read_buf = Cursor::new(buf);
        let result: io::Result<TensorMessage> = read_cbor_message(&mut read_buf).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_max_valid_message() {
        // Create a message close to but under the size limit
        // ~9MB of data (under 10MB limit)
        let msg = TensorMessage {
            job_id: Uuid::new_v4(),
            layer_idx: 0,
            phase: AllReducePhase::ReduceScatter,
            step: 0,
            chunk_data: vec![1.0; 2_000_000], // ~8MB of f32s
            chunk_shape: vec![2_000_000],
            timestamp: 0,
        };

        // Should write successfully
        let mut write_buf = Vec::new();
        write_cbor_message(&mut write_buf, &msg).await.unwrap();

        // Should read successfully
        let mut read_buf = Cursor::new(write_buf);
        let decoded: TensorMessage = read_cbor_message(&mut read_buf).await.unwrap();
        assert_eq!(msg.chunk_data.len(), decoded.chunk_data.len());
    }
}
