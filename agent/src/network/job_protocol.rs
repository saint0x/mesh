// Job protocol implementation for distributed job execution over libp2p

use futures::prelude::*;
use libp2p::request_response::{self, Behaviour, Config, ProtocolSupport};
use libp2p::StreamProtocol;
use serde::{Deserialize, Serialize};
use std::io;
use std::time::Duration;
use uuid::Uuid;

/// Protocol ID for mesh job distribution
pub const JOB_PROTOCOL_ID: StreamProtocol = StreamProtocol::new("/mesh/job/1.0.0");

/// Job envelope containing all information needed to execute a job
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobEnvelope {
    /// Unique identifier for this job
    pub job_id: Uuid,
    /// Network this job belongs to
    pub network_id: String,
    /// Type of workload (e.g., "embeddings-v1", "ocr-v1")
    pub workload_id: String,
    /// CBOR-encoded workload-specific data
    pub payload: Vec<u8>,
    /// Maximum execution time in milliseconds
    pub timeout_ms: u64,
    /// Ed25519 signature of the job (for authentication)
    pub auth_signature: Vec<u8>,
    /// Unix timestamp when the job was created
    pub created_at: u64,
}

/// Result of job execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobResult {
    /// Job ID this result corresponds to
    pub job_id: Uuid,
    /// Whether the job succeeded
    pub success: bool,
    /// CBOR-encoded result data (if successful)
    pub result: Option<Vec<u8>>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Actual execution time in milliseconds
    pub execution_time_ms: u64,
}

/// CBOR codec for job protocol messages
#[derive(Debug, Clone, Default)]
pub struct JobCodec;

#[async_trait::async_trait]
impl request_response::Codec for JobCodec {
    type Protocol = StreamProtocol;
    type Request = JobEnvelope;
    type Response = JobResult;

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

    // Enforce maximum message size (10MB)
    const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;
    if len > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Message too large: {} bytes (max {})",
                len, MAX_MESSAGE_SIZE
            ),
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

    // Write u32 length prefix (big-endian)
    let len = buf.len() as u32;
    io.write_all(&len.to_be_bytes()).await?;

    // Write CBOR payload
    io.write_all(&buf).await?;
    io.flush().await?;

    Ok(())
}

/// Configuration for the job protocol behavior
#[derive(Debug, Clone)]
pub struct JobProtocolConfig {
    /// Request timeout
    pub request_timeout: Duration,
    /// Connection keep-alive duration
    pub keep_alive: Duration,
}

impl Default for JobProtocolConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(300), // 5 minutes for job execution
            keep_alive: Duration::from_secs(60),
        }
    }
}

/// Type alias for job protocol behavior
pub type JobProtocol = Behaviour<JobCodec>;

/// Create a new job protocol behavior
pub fn new_job_protocol(config: JobProtocolConfig) -> JobProtocol {
    let protocols = std::iter::once((JOB_PROTOCOL_ID, ProtocolSupport::Full));

    let mut cfg = Config::default();
    cfg = cfg.with_request_timeout(config.request_timeout);

    Behaviour::new(protocols, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_job_envelope_serialization() {
        let job = JobEnvelope {
            job_id: Uuid::new_v4(),
            network_id: "test-network".to_string(),
            workload_id: "embeddings-v1".to_string(),
            payload: vec![1, 2, 3, 4],
            timeout_ms: 5000,
            auth_signature: vec![5, 6, 7, 8],
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Serialize to CBOR
        let mut buf = Vec::new();
        ciborium::into_writer(&job, &mut buf).unwrap();

        // Deserialize from CBOR
        let decoded: JobEnvelope = ciborium::from_reader(&buf[..]).unwrap();

        assert_eq!(job, decoded);
    }

    #[test]
    fn test_job_result_serialization() {
        let result = JobResult {
            job_id: Uuid::new_v4(),
            success: true,
            result: Some(vec![9, 10, 11]),
            error: None,
            execution_time_ms: 1234,
        };

        // Serialize to CBOR
        let mut buf = Vec::new();
        ciborium::into_writer(&result, &mut buf).unwrap();

        // Deserialize from CBOR
        let decoded: JobResult = ciborium::from_reader(&buf[..]).unwrap();

        assert_eq!(result, decoded);
    }

    #[test]
    fn test_protocol_id() {
        assert_eq!(JOB_PROTOCOL_ID.as_ref(), "/mesh/job/1.0.0");
    }

    #[test]
    fn test_default_config() {
        let config = JobProtocolConfig::default();
        assert_eq!(config.request_timeout, Duration::from_secs(300));
        assert_eq!(config.keep_alive, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_cbor_roundtrip() {
        use futures::io::Cursor;

        let job = JobEnvelope {
            job_id: Uuid::new_v4(),
            network_id: "test".to_string(),
            workload_id: "test-workload".to_string(),
            payload: vec![1, 2, 3],
            timeout_ms: 1000,
            auth_signature: vec![4, 5, 6],
            created_at: 1234567890,
        };

        // Write to buffer
        let mut write_buf = Vec::new();
        write_cbor_message(&mut write_buf, &job).await.unwrap();

        // Read from buffer
        let mut read_buf = Cursor::new(write_buf);
        let decoded: JobEnvelope = read_cbor_message(&mut read_buf).await.unwrap();

        assert_eq!(job, decoded);
    }

    #[tokio::test]
    async fn test_message_size_limit() {
        use futures::io::Cursor;

        // Create a message that exceeds the size limit
        let large_job = JobEnvelope {
            job_id: Uuid::new_v4(),
            network_id: "test".to_string(),
            workload_id: "test".to_string(),
            payload: vec![0u8; 11 * 1024 * 1024], // 11MB
            timeout_ms: 1000,
            auth_signature: vec![],
            created_at: 0,
        };

        let mut buf = Vec::new();
        write_cbor_message(&mut buf, &large_job).await.unwrap();

        // Try to read it (should fail)
        let mut read_buf = Cursor::new(buf);
        let result: io::Result<JobEnvelope> = read_cbor_message(&mut read_buf).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }
}
