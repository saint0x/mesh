use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tracing::{debug, warn};

use crate::errors::{AgentError, Result};

use super::tensor_message::TensorMessage;

pub const DATA_PLANE_ENDPOINT_PREFIX: &str = "dataplane://";
pub const MESSAGE_SIZE_LIMIT: usize = 10 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct TensorPlaneConfig {
    pub bind_addr: SocketAddr,
    pub connect_timeout: Duration,
    pub io_timeout: Duration,
}

impl Default for TensorPlaneConfig {
    fn default() -> Self {
        Self {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            connect_timeout: Duration::from_secs(2),
            io_timeout: Duration::from_secs(5),
        }
    }
}

#[derive(Debug)]
pub struct InboundTensorMessage {
    pub tensor: TensorMessage,
    pub remote_addr: SocketAddr,
}

#[derive(Debug)]
struct TensorPlaneState {
    local_addr: SocketAddr,
    advertised_addr: SocketAddr,
    connect_timeout: Duration,
    io_timeout: Duration,
}

pub struct TensorPlane {
    state: Arc<TensorPlaneState>,
    inbound_rx: mpsc::UnboundedReceiver<InboundTensorMessage>,
    _accept_task: tokio::task::JoinHandle<()>,
}

impl TensorPlane {
    pub async fn bind(config: TensorPlaneConfig) -> Result<Self> {
        let listener = TcpListener::bind(config.bind_addr)
            .await
            .map_err(|e| AgentError::Network(format!("Failed to bind tensor plane: {}", e)))?;
        let local_addr = listener.local_addr().map_err(|e| {
            AgentError::Network(format!("Failed to inspect tensor plane bind: {}", e))
        })?;
        let advertised_addr = SocketAddr::new(resolve_advertised_ip()?, local_addr.port());

        let (inbound_tx, inbound_rx) = mpsc::unbounded_channel();
        let io_timeout = config.io_timeout;
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((mut stream, remote_addr)) => {
                        let tx = inbound_tx.clone();
                        tokio::spawn(async move {
                            match tokio::time::timeout(io_timeout, read_tensor_message(&mut stream))
                                .await
                            {
                                Ok(Ok(tensor)) => {
                                    if tx
                                        .send(InboundTensorMessage {
                                            tensor,
                                            remote_addr,
                                        })
                                        .is_err()
                                    {
                                        debug!("Tensor plane receiver dropped");
                                    }
                                }
                                Ok(Err(error)) => {
                                    warn!(error = %error, remote_addr = %remote_addr, "Tensor plane receive failed");
                                }
                                Err(_) => {
                                    warn!(remote_addr = %remote_addr, "Tensor plane receive timed out");
                                }
                            }
                        });
                    }
                    Err(error) => {
                        warn!(error = %error, "Tensor plane accept failed");
                    }
                }
            }
        });

        Ok(Self {
            state: Arc::new(TensorPlaneState {
                local_addr,
                advertised_addr,
                connect_timeout: config.connect_timeout,
                io_timeout: config.io_timeout,
            }),
            inbound_rx,
            _accept_task: accept_task,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.state.local_addr
    }

    pub fn advertised_addr(&self) -> SocketAddr {
        self.state.advertised_addr
    }

    pub fn advertised_endpoint(&self) -> String {
        format!(
            "{}{}",
            DATA_PLANE_ENDPOINT_PREFIX, self.state.advertised_addr
        )
    }

    pub async fn send(&self, target: SocketAddr, message: &TensorMessage) -> Result<()> {
        let mut stream =
            tokio::time::timeout(self.state.connect_timeout, TcpStream::connect(target))
                .await
                .map_err(|_| {
                    AgentError::Network(format!("Timed out connecting to tensor peer {}", target))
                })?
                .map_err(|e| {
                    AgentError::Network(format!(
                        "Failed to connect to tensor peer {}: {}",
                        target, e
                    ))
                })?;

        tokio::time::timeout(
            self.state.io_timeout,
            write_tensor_message(&mut stream, message),
        )
        .await
        .map_err(|_| AgentError::Network(format!("Timed out sending tensor to {}", target)))?
        .map_err(|e| AgentError::Network(format!("Failed to send tensor to {}: {}", target, e)))?;

        Ok(())
    }

    pub async fn recv(&mut self) -> Option<InboundTensorMessage> {
        self.inbound_rx.recv().await
    }
}

pub fn parse_data_plane_endpoint(endpoint: &str) -> Option<SocketAddr> {
    endpoint
        .strip_prefix(DATA_PLANE_ENDPOINT_PREFIX)
        .and_then(|addr| addr.parse::<SocketAddr>().ok())
}

fn resolve_advertised_ip() -> Result<IpAddr> {
    let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(|e| {
        AgentError::Network(format!(
            "Failed to prepare tensor plane IP discovery: {}",
            e
        ))
    })?;
    socket
        .connect((Ipv4Addr::new(8, 8, 8, 8), 80))
        .map_err(|e| AgentError::Network(format!("Failed to discover tensor plane IP: {}", e)))?;
    socket
        .local_addr()
        .map(|addr| addr.ip())
        .map_err(|e| AgentError::Network(format!("Failed to read tensor plane local IP: {}", e)))
}

async fn read_tensor_message<T>(io: &mut T) -> std::io::Result<TensorMessage>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MESSAGE_SIZE_LIMIT {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Message size {} exceeds limit {}", len, MESSAGE_SIZE_LIMIT),
        ));
    }

    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    ciborium::from_reader(&buf[..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

async fn write_tensor_message<T>(io: &mut T, message: &TensorMessage) -> std::io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
{
    let mut buf = Vec::new();
    ciborium::into_writer(message, &mut buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if buf.len() > MESSAGE_SIZE_LIMIT {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Message size {} exceeds limit {}",
                buf.len(),
                MESSAGE_SIZE_LIMIT
            ),
        ));
    }

    io.write_all(&(buf.len() as u32).to_be_bytes()).await?;
    io.write_all(&buf).await?;
    io.flush().await?;
    Ok(())
}
