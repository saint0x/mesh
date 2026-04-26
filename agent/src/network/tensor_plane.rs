use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tracing::warn;

use crate::errors::{AgentError, Result};

use super::tensor_message::{AllReducePhase, TensorMessage, TensorTrafficClass};

pub const DATA_PLANE_ENDPOINT_PREFIX: &str = "dataplane://";
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 10 * 1024 * 1024;
pub const DEFAULT_MAX_INBOUND_MESSAGES: usize = 64;
pub const DEFAULT_MAX_INBOUND_QUEUED_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_OUTBOUND_INFLIGHT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_BULK_BANDWIDTH_SHARE_NUMERATOR: u64 = 3;
const DEFAULT_BULK_BANDWIDTH_SHARE_DENOMINATOR: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorPlaneProfile {
    Lan,
    Conservative,
}

impl Default for TensorPlaneProfile {
    fn default() -> Self {
        Self::Conservative
    }
}

impl TensorPlaneProfile {
    fn default_bandwidth_bytes_per_sec(self, max_message_bytes: usize) -> u64 {
        match self {
            Self::Lan => (256 * 1024 * 1024) as u64,
            Self::Conservative => max_message_bytes as u64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorPlaneConfig {
    pub profile: TensorPlaneProfile,
    pub bind_addr: SocketAddr,
    pub advertised_addr: Option<SocketAddr>,
    pub connect_timeout: Duration,
    pub io_timeout: Duration,
    pub max_message_bytes: usize,
    pub max_inbound_messages: usize,
    pub max_inbound_queued_bytes: usize,
    pub max_outbound_inflight_bytes: usize,
    pub max_send_bandwidth_bytes_per_sec: u64,
}

impl Default for TensorPlaneConfig {
    fn default() -> Self {
        Self {
            profile: TensorPlaneProfile::default(),
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
            advertised_addr: None,
            connect_timeout: Duration::from_secs(2),
            io_timeout: Duration::from_secs(5),
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
            max_inbound_messages: DEFAULT_MAX_INBOUND_MESSAGES,
            max_inbound_queued_bytes: DEFAULT_MAX_INBOUND_QUEUED_BYTES,
            max_outbound_inflight_bytes: DEFAULT_MAX_OUTBOUND_INFLIGHT_BYTES,
            max_send_bandwidth_bytes_per_sec: DEFAULT_MAX_MESSAGE_BYTES as u64,
        }
    }
}

#[derive(Debug)]
pub struct InboundTensorMessage {
    pub tensor: TensorMessage,
    pub remote_addr: SocketAddr,
    queued_at: Instant,
    _queued_bytes_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorPlaneMetricsSnapshot {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub reduce_scatter_bytes_sent: u64,
    pub reduce_scatter_bytes_received: u64,
    pub all_gather_bytes_sent: u64,
    pub all_gather_bytes_received: u64,
    pub barrier_bytes_sent: u64,
    pub barrier_bytes_received: u64,
    pub outbound_backpressure_wait_count: u64,
    pub outbound_backpressure_wait_ms: u64,
    pub outbound_bandwidth_wait_count: u64,
    pub outbound_bandwidth_wait_ms: u64,
    pub send_count: u64,
    pub send_latency_ms: u64,
    pub receive_count: u64,
    pub receive_latency_ms: u64,
    pub receive_queue_wait_ms: u64,
    pub send_timeout_count: u64,
    pub receive_timeout_count: u64,
    pub inbound_queue_full_rejections: u64,
    pub inbound_byte_budget_rejections: u64,
    pub oversized_message_rejections: u64,
    pub current_inbound_queued_bytes: u64,
    pub peak_inbound_queued_bytes: u64,
    pub current_outbound_inflight_bytes: u64,
    pub peak_outbound_inflight_bytes: u64,
    pub current_outbound_connections: u64,
    pub latency_critical_send_count: u64,
    pub interactive_send_count: u64,
    pub bulk_send_count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorPlaneCapabilitiesSnapshot {
    pub profile: TensorPlaneProfile,
    pub max_message_bytes: usize,
    pub inbound_queue_byte_capacity: usize,
    pub outbound_inflight_byte_capacity: usize,
    pub reserved_priority_outbound_inflight_bytes: usize,
    pub bulk_outbound_inflight_byte_capacity: usize,
    pub max_send_bandwidth_bytes_per_sec: u64,
    pub bulk_send_bandwidth_bytes_per_sec: u64,
    pub concurrent_receive_waiters: bool,
    pub prioritized_traffic_classes: bool,
}

#[derive(Debug)]
struct TensorPlaneMetrics {
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    reduce_scatter_bytes_sent: AtomicU64,
    reduce_scatter_bytes_received: AtomicU64,
    all_gather_bytes_sent: AtomicU64,
    all_gather_bytes_received: AtomicU64,
    barrier_bytes_sent: AtomicU64,
    barrier_bytes_received: AtomicU64,
    outbound_backpressure_wait_count: AtomicU64,
    outbound_backpressure_wait_ms: AtomicU64,
    outbound_bandwidth_wait_count: AtomicU64,
    outbound_bandwidth_wait_ms: AtomicU64,
    send_count: AtomicU64,
    send_latency_ms: AtomicU64,
    receive_count: AtomicU64,
    receive_latency_ms: AtomicU64,
    receive_queue_wait_ms: AtomicU64,
    send_timeout_count: AtomicU64,
    receive_timeout_count: AtomicU64,
    inbound_queue_full_rejections: AtomicU64,
    inbound_byte_budget_rejections: AtomicU64,
    oversized_message_rejections: AtomicU64,
    peak_inbound_queued_bytes: AtomicU64,
    peak_outbound_inflight_bytes: AtomicU64,
    current_outbound_connections: AtomicU64,
    latency_critical_send_count: AtomicU64,
    interactive_send_count: AtomicU64,
    bulk_send_count: AtomicU64,
}

impl TensorPlaneMetrics {
    fn new() -> Self {
        Self {
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            reduce_scatter_bytes_sent: AtomicU64::new(0),
            reduce_scatter_bytes_received: AtomicU64::new(0),
            all_gather_bytes_sent: AtomicU64::new(0),
            all_gather_bytes_received: AtomicU64::new(0),
            barrier_bytes_sent: AtomicU64::new(0),
            barrier_bytes_received: AtomicU64::new(0),
            outbound_backpressure_wait_count: AtomicU64::new(0),
            outbound_backpressure_wait_ms: AtomicU64::new(0),
            outbound_bandwidth_wait_count: AtomicU64::new(0),
            outbound_bandwidth_wait_ms: AtomicU64::new(0),
            send_count: AtomicU64::new(0),
            send_latency_ms: AtomicU64::new(0),
            receive_count: AtomicU64::new(0),
            receive_latency_ms: AtomicU64::new(0),
            receive_queue_wait_ms: AtomicU64::new(0),
            send_timeout_count: AtomicU64::new(0),
            receive_timeout_count: AtomicU64::new(0),
            inbound_queue_full_rejections: AtomicU64::new(0),
            inbound_byte_budget_rejections: AtomicU64::new(0),
            oversized_message_rejections: AtomicU64::new(0),
            peak_inbound_queued_bytes: AtomicU64::new(0),
            peak_outbound_inflight_bytes: AtomicU64::new(0),
            current_outbound_connections: AtomicU64::new(0),
            latency_critical_send_count: AtomicU64::new(0),
            interactive_send_count: AtomicU64::new(0),
            bulk_send_count: AtomicU64::new(0),
        }
    }
}

#[derive(Debug)]
struct InboundState {
    pending_inbound: VecDeque<InboundTensorMessage>,
    closed: bool,
}

#[derive(Debug)]
struct TensorPlaneState {
    profile: TensorPlaneProfile,
    local_addr: SocketAddr,
    advertised_addr: SocketAddr,
    connect_timeout: Duration,
    io_timeout: Duration,
    max_message_bytes: usize,
    inbound_queue_byte_capacity: usize,
    outbound_inflight_byte_capacity: usize,
    reserved_priority_outbound_inflight_bytes: usize,
    bulk_outbound_inflight_byte_capacity: usize,
    max_send_bandwidth_bytes_per_sec: u64,
    bulk_send_bandwidth_bytes_per_sec: u64,
    inbound_queue_bytes: Arc<Semaphore>,
    outbound_inflight_bytes: Arc<Semaphore>,
    bulk_outbound_inflight_bytes: Arc<Semaphore>,
    bulk_outbound_bandwidth_limiter: Mutex<BandwidthLimiter>,
    outbound_connections: Mutex<HashMap<SocketAddr, Arc<Mutex<TcpStream>>>>,
    metrics: TensorPlaneMetrics,
}

#[derive(Debug)]
struct BandwidthLimiter {
    rate_bytes_per_sec: u64,
    burst_bytes: f64,
    available_bytes: f64,
    last_refill: Instant,
}

impl BandwidthLimiter {
    fn new(rate_bytes_per_sec: u64, burst_bytes: usize) -> Self {
        let rate_bytes_per_sec = rate_bytes_per_sec.max(1);
        let burst_bytes = burst_bytes.max(1) as f64;
        Self {
            rate_bytes_per_sec,
            burst_bytes,
            available_bytes: burst_bytes,
            last_refill: Instant::now(),
        }
    }

    fn reserve_wait(&mut self, bytes: usize) -> Duration {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let replenished = self.available_bytes + elapsed * self.rate_bytes_per_sec as f64;
        self.available_bytes = replenished.min(self.burst_bytes);
        self.last_refill = now;

        let bytes = bytes.max(1) as f64;
        if self.available_bytes >= bytes {
            self.available_bytes -= bytes;
            return Duration::ZERO;
        }

        let deficit = bytes - self.available_bytes;
        let wait = Duration::from_secs_f64(deficit / self.rate_bytes_per_sec as f64);
        self.available_bytes = 0.0;
        self.last_refill = now + wait;
        wait
    }
}

pub struct TensorPlane {
    state: Arc<TensorPlaneState>,
    inbound: Arc<Mutex<InboundState>>,
    inbound_notify: Arc<Notify>,
    _accept_task: tokio::task::JoinHandle<()>,
    _inbound_dispatch_task: tokio::task::JoinHandle<()>,
}

impl TensorPlane {
    pub async fn bind(config: TensorPlaneConfig) -> Result<Self> {
        let config = sanitized_config(config)?;
        let listener = TcpListener::bind(config.bind_addr)
            .await
            .map_err(|e| AgentError::Network(format!("Failed to bind tensor plane: {}", e)))?;
        let local_addr = listener.local_addr().map_err(|e| {
            AgentError::Network(format!("Failed to inspect tensor plane bind: {}", e))
        })?;
        let advertised_addr = resolve_advertised_addr(config.advertised_addr, local_addr.port())?;

        let inbound_queue_bytes = Arc::new(Semaphore::new(
            config.max_inbound_queued_bytes.try_into().map_err(|_| {
                AgentError::Config("tensor inbound byte budget exceeds u32".to_string())
            })?,
        ));
        let outbound_inflight_bytes = Arc::new(Semaphore::new(
            config.max_outbound_inflight_bytes.try_into().map_err(|_| {
                AgentError::Config("tensor outbound byte budget exceeds u32".to_string())
            })?,
        ));
        let reserved_priority_outbound_inflight_bytes =
            compute_reserved_priority_outbound_inflight_bytes(
                config.max_outbound_inflight_bytes,
                config.max_message_bytes,
            );
        let bulk_outbound_inflight_byte_capacity = config
            .max_outbound_inflight_bytes
            .saturating_sub(reserved_priority_outbound_inflight_bytes)
            .max(config.max_message_bytes);
        let bulk_outbound_inflight_bytes = Arc::new(Semaphore::new(
            bulk_outbound_inflight_byte_capacity
                .try_into()
                .map_err(|_| {
                    AgentError::Config("tensor bulk byte budget exceeds u32".to_string())
                })?,
        ));
        let bulk_send_bandwidth_bytes_per_sec = compute_bulk_bandwidth_bytes_per_sec(
            config.max_send_bandwidth_bytes_per_sec,
            config.max_message_bytes,
        );
        let metrics = TensorPlaneMetrics::new();
        let (inbound_tx, inbound_rx) = mpsc::channel(config.max_inbound_messages);
        let io_timeout = config.io_timeout;
        let max_message_bytes = config.max_message_bytes;
        let inbound = Arc::new(Mutex::new(InboundState {
            pending_inbound: VecDeque::new(),
            closed: false,
        }));
        let inbound_notify = Arc::new(Notify::new());
        let metrics_state = Arc::new(TensorPlaneState {
            profile: config.profile,
            local_addr,
            advertised_addr,
            connect_timeout: config.connect_timeout,
            io_timeout: config.io_timeout,
            max_message_bytes,
            inbound_queue_byte_capacity: config.max_inbound_queued_bytes,
            outbound_inflight_byte_capacity: config.max_outbound_inflight_bytes,
            reserved_priority_outbound_inflight_bytes,
            bulk_outbound_inflight_byte_capacity,
            max_send_bandwidth_bytes_per_sec: config.max_send_bandwidth_bytes_per_sec,
            bulk_send_bandwidth_bytes_per_sec,
            inbound_queue_bytes,
            outbound_inflight_bytes,
            bulk_outbound_inflight_bytes,
            bulk_outbound_bandwidth_limiter: Mutex::new(BandwidthLimiter::new(
                bulk_send_bandwidth_bytes_per_sec,
                config.max_message_bytes,
            )),
            outbound_connections: Mutex::new(HashMap::new()),
            metrics,
        });
        let inbound_state = Arc::clone(&inbound);
        let inbound_notify_state = Arc::clone(&inbound_notify);
        let inbound_dispatch_task = tokio::spawn(async move {
            let mut inbound_rx = inbound_rx;
            while let Some(message) = inbound_rx.recv().await {
                let mut state = inbound_state.lock().await;
                state.pending_inbound.push_back(message);
                drop(state);
                inbound_notify_state.notify_waiters();
            }

            let mut state = inbound_state.lock().await;
            state.closed = true;
            drop(state);
            inbound_notify_state.notify_waiters();
        });
        let accept_state = Arc::clone(&metrics_state);
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((mut stream, remote_addr)) => {
                        let tx = inbound_tx.clone();
                        let state = Arc::clone(&accept_state);
                        tokio::spawn(async move {
                            loop {
                                match tokio::time::timeout(
                                    io_timeout,
                                    read_tensor_message(&mut stream, max_message_bytes),
                                )
                                .await
                                {
                                    Ok(Ok(tensor)) => {
                                        let queued_bytes = tensor.size_bytes().max(1);
                                        let queued_bytes_u32: u32 = match queued_bytes.try_into() {
                                            Ok(value) => value,
                                            Err(_) => {
                                                state
                                                    .metrics
                                                    .oversized_message_rejections
                                                    .fetch_add(1, Ordering::Relaxed);
                                                warn!(
                                                    queued_bytes,
                                                    remote_addr = %remote_addr,
                                                    "Tensor message size exceeded supported queue accounting"
                                                );
                                                return;
                                            }
                                        };
                                        state
                                            .metrics
                                            .bytes_received
                                            .fetch_add(queued_bytes as u64, Ordering::Relaxed);
                                        record_phase_bytes(
                                            &state.metrics,
                                            tensor.phase,
                                            queued_bytes as u64,
                                            false,
                                        );

                                        let queued_bytes_permit = match state
                                            .inbound_queue_bytes
                                            .clone()
                                            .try_acquire_many_owned(queued_bytes_u32)
                                        {
                                            Ok(permit) => permit,
                                            Err(_) => {
                                                state
                                                    .metrics
                                                    .inbound_byte_budget_rejections
                                                    .fetch_add(1, Ordering::Relaxed);
                                                warn!(
                                                    queued_bytes,
                                                    remote_addr = %remote_addr,
                                                    "Tensor plane inbound byte budget exhausted"
                                                );
                                                continue;
                                            }
                                        };

                                        if tx
                                            .try_send(InboundTensorMessage {
                                                tensor,
                                                remote_addr,
                                                queued_at: Instant::now(),
                                                _queued_bytes_permit: Some(queued_bytes_permit),
                                            })
                                            .is_err()
                                        {
                                            state
                                                .metrics
                                                .inbound_queue_full_rejections
                                                .fetch_add(1, Ordering::Relaxed);
                                            warn!(
                                                remote_addr = %remote_addr,
                                                "Tensor plane inbound queue full"
                                            );
                                        } else {
                                            update_peak(
                                                &state.metrics.peak_inbound_queued_bytes,
                                                (state.inbound_queue_byte_capacity
                                                    - state.inbound_queue_bytes.available_permits()
                                                        as usize)
                                                    as u64,
                                            );
                                        }
                                    }
                                    Ok(Err(error)) => {
                                        if error.kind() == std::io::ErrorKind::InvalidData {
                                            state
                                                .metrics
                                                .oversized_message_rejections
                                                .fetch_add(1, Ordering::Relaxed);
                                            warn!(error = %error, remote_addr = %remote_addr, "Tensor plane receive failed");
                                        }
                                        return;
                                    }
                                    Err(_) => {
                                        state
                                            .metrics
                                            .receive_timeout_count
                                            .fetch_add(1, Ordering::Relaxed);
                                        warn!(remote_addr = %remote_addr, "Tensor plane receive timed out");
                                        return;
                                    }
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
            state: metrics_state,
            inbound,
            inbound_notify,
            _accept_task: accept_task,
            _inbound_dispatch_task: inbound_dispatch_task,
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

    pub fn capabilities_snapshot(&self) -> TensorPlaneCapabilitiesSnapshot {
        TensorPlaneCapabilitiesSnapshot {
            profile: self.state.profile,
            max_message_bytes: self.state.max_message_bytes,
            inbound_queue_byte_capacity: self.state.inbound_queue_byte_capacity,
            outbound_inflight_byte_capacity: self.state.outbound_inflight_byte_capacity,
            reserved_priority_outbound_inflight_bytes: self
                .state
                .reserved_priority_outbound_inflight_bytes,
            bulk_outbound_inflight_byte_capacity: self.state.bulk_outbound_inflight_byte_capacity,
            max_send_bandwidth_bytes_per_sec: self.state.max_send_bandwidth_bytes_per_sec,
            bulk_send_bandwidth_bytes_per_sec: self.state.bulk_send_bandwidth_bytes_per_sec,
            concurrent_receive_waiters: true,
            prioritized_traffic_classes: true,
        }
    }

    pub fn metrics_snapshot(&self) -> TensorPlaneMetricsSnapshot {
        TensorPlaneMetricsSnapshot {
            bytes_sent: self.state.metrics.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.state.metrics.bytes_received.load(Ordering::Relaxed),
            reduce_scatter_bytes_sent: self
                .state
                .metrics
                .reduce_scatter_bytes_sent
                .load(Ordering::Relaxed),
            reduce_scatter_bytes_received: self
                .state
                .metrics
                .reduce_scatter_bytes_received
                .load(Ordering::Relaxed),
            all_gather_bytes_sent: self
                .state
                .metrics
                .all_gather_bytes_sent
                .load(Ordering::Relaxed),
            all_gather_bytes_received: self
                .state
                .metrics
                .all_gather_bytes_received
                .load(Ordering::Relaxed),
            barrier_bytes_sent: self
                .state
                .metrics
                .barrier_bytes_sent
                .load(Ordering::Relaxed),
            barrier_bytes_received: self
                .state
                .metrics
                .barrier_bytes_received
                .load(Ordering::Relaxed),
            outbound_backpressure_wait_count: self
                .state
                .metrics
                .outbound_backpressure_wait_count
                .load(Ordering::Relaxed),
            outbound_backpressure_wait_ms: self
                .state
                .metrics
                .outbound_backpressure_wait_ms
                .load(Ordering::Relaxed),
            outbound_bandwidth_wait_count: self
                .state
                .metrics
                .outbound_bandwidth_wait_count
                .load(Ordering::Relaxed),
            outbound_bandwidth_wait_ms: self
                .state
                .metrics
                .outbound_bandwidth_wait_ms
                .load(Ordering::Relaxed),
            send_count: self.state.metrics.send_count.load(Ordering::Relaxed),
            send_latency_ms: self.state.metrics.send_latency_ms.load(Ordering::Relaxed),
            receive_count: self.state.metrics.receive_count.load(Ordering::Relaxed),
            receive_latency_ms: self
                .state
                .metrics
                .receive_latency_ms
                .load(Ordering::Relaxed),
            receive_queue_wait_ms: self
                .state
                .metrics
                .receive_queue_wait_ms
                .load(Ordering::Relaxed),
            send_timeout_count: self
                .state
                .metrics
                .send_timeout_count
                .load(Ordering::Relaxed),
            receive_timeout_count: self
                .state
                .metrics
                .receive_timeout_count
                .load(Ordering::Relaxed),
            inbound_queue_full_rejections: self
                .state
                .metrics
                .inbound_queue_full_rejections
                .load(Ordering::Relaxed),
            inbound_byte_budget_rejections: self
                .state
                .metrics
                .inbound_byte_budget_rejections
                .load(Ordering::Relaxed),
            oversized_message_rejections: self
                .state
                .metrics
                .oversized_message_rejections
                .load(Ordering::Relaxed),
            current_inbound_queued_bytes: (self.state.inbound_queue_byte_capacity
                - self.state.inbound_queue_bytes.available_permits() as usize)
                as u64,
            peak_inbound_queued_bytes: self
                .state
                .metrics
                .peak_inbound_queued_bytes
                .load(Ordering::Relaxed),
            current_outbound_inflight_bytes: (self.state.outbound_inflight_byte_capacity
                - self.state.outbound_inflight_bytes.available_permits() as usize)
                as u64,
            peak_outbound_inflight_bytes: self
                .state
                .metrics
                .peak_outbound_inflight_bytes
                .load(Ordering::Relaxed),
            current_outbound_connections: self
                .state
                .metrics
                .current_outbound_connections
                .load(Ordering::Relaxed),
            latency_critical_send_count: self
                .state
                .metrics
                .latency_critical_send_count
                .load(Ordering::Relaxed),
            interactive_send_count: self
                .state
                .metrics
                .interactive_send_count
                .load(Ordering::Relaxed),
            bulk_send_count: self.state.metrics.bulk_send_count.load(Ordering::Relaxed),
        }
    }

    pub async fn send(&self, target: SocketAddr, message: &TensorMessage) -> Result<()> {
        self.send_with_traffic_class(target, message, message.traffic_class())
            .await
    }

    pub async fn send_with_traffic_class(
        &self,
        target: SocketAddr,
        message: &TensorMessage,
        traffic_class: TensorTrafficClass,
    ) -> Result<()> {
        let send_started = Instant::now();
        let message_bytes = message.size_bytes().max(1);
        if message_bytes > self.state.max_message_bytes {
            self.state
                .metrics
                .oversized_message_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(AgentError::Network(format!(
                "Tensor message size {} exceeds limit {}",
                message_bytes, self.state.max_message_bytes
            )));
        }
        if message_bytes > self.state.outbound_inflight_byte_capacity {
            return Err(AgentError::Network(format!(
                "Tensor message size {} exceeds outbound byte budget {}",
                message_bytes, self.state.outbound_inflight_byte_capacity
            )));
        }

        let message_bytes_u32: u32 = message_bytes.try_into().map_err(|_| {
            AgentError::Network("Tensor message too large for byte accounting".to_string())
        })?;
        let wait_started = Instant::now();
        let outbound_permit = match self
            .state
            .outbound_inflight_bytes
            .clone()
            .try_acquire_many_owned(message_bytes_u32)
        {
            Ok(permit) => permit,
            Err(_) => {
                self.state
                    .metrics
                    .outbound_backpressure_wait_count
                    .fetch_add(1, Ordering::Relaxed);
                let permit = self
                    .state
                    .outbound_inflight_bytes
                    .clone()
                    .acquire_many_owned(message_bytes_u32)
                    .await
                    .map_err(|_| {
                        AgentError::Network(
                            "Tensor plane outbound byte budget unexpectedly closed".to_string(),
                        )
                    })?;
                self.state
                    .metrics
                    .outbound_backpressure_wait_ms
                    .fetch_add(wait_started.elapsed().as_millis() as u64, Ordering::Relaxed);
                permit
            }
        };
        let bulk_outbound_permit = if traffic_class.is_bulk() {
            let wait_started = Instant::now();
            Some(
                match self
                    .state
                    .bulk_outbound_inflight_bytes
                    .clone()
                    .try_acquire_many_owned(message_bytes_u32)
                {
                    Ok(permit) => permit,
                    Err(_) => {
                        self.state
                            .metrics
                            .outbound_backpressure_wait_count
                            .fetch_add(1, Ordering::Relaxed);
                        let permit = self
                            .state
                            .bulk_outbound_inflight_bytes
                            .clone()
                            .acquire_many_owned(message_bytes_u32)
                            .await
                            .map_err(|_| {
                                AgentError::Network(
                                    "Tensor plane bulk outbound byte budget unexpectedly closed"
                                        .to_string(),
                                )
                            })?;
                        self.state.metrics.outbound_backpressure_wait_ms.fetch_add(
                            wait_started.elapsed().as_millis() as u64,
                            Ordering::Relaxed,
                        );
                        permit
                    }
                },
            )
        } else {
            None
        };
        update_peak(
            &self.state.metrics.peak_outbound_inflight_bytes,
            (self.state.outbound_inflight_byte_capacity
                - self.state.outbound_inflight_bytes.available_permits() as usize)
                as u64,
        );
        let bandwidth_wait = if traffic_class.is_bulk() {
            let mut limiter = self.state.bulk_outbound_bandwidth_limiter.lock().await;
            limiter.reserve_wait(message_bytes)
        } else {
            Duration::ZERO
        };
        if !bandwidth_wait.is_zero() {
            self.state
                .metrics
                .outbound_bandwidth_wait_count
                .fetch_add(1, Ordering::Relaxed);
            self.state
                .metrics
                .outbound_bandwidth_wait_ms
                .fetch_add(bandwidth_wait.as_millis() as u64, Ordering::Relaxed);
            tokio::time::sleep(bandwidth_wait).await;
        }
        let stream = self.get_or_connect(target).await?;
        let send_result = {
            let mut stream_guard = stream.lock().await;
            tokio::time::timeout(
                self.state.io_timeout,
                write_tensor_message(&mut *stream_guard, message, self.state.max_message_bytes),
            )
            .await
        };
        let send_result = match send_result {
            Ok(result) => result,
            Err(_) => {
                self.state
                    .metrics
                    .send_timeout_count
                    .fetch_add(1, Ordering::Relaxed);
                self.evict_connection(target).await;
                drop(outbound_permit);
                return Err(AgentError::Network(format!(
                    "Timed out sending tensor to {}",
                    target
                )));
            }
        };
        if let Err(error) = send_result {
            self.evict_connection(target).await;
            drop(outbound_permit);
            return Err(AgentError::Network(format!(
                "Failed to send tensor to {}: {}",
                target, error
            )));
        }
        self.state
            .metrics
            .bytes_sent
            .fetch_add(message_bytes as u64, Ordering::Relaxed);
        record_phase_bytes(
            &self.state.metrics,
            message.phase,
            message_bytes as u64,
            true,
        );
        self.state
            .metrics
            .send_count
            .fetch_add(1, Ordering::Relaxed);
        self.state
            .metrics
            .send_latency_ms
            .fetch_add(send_started.elapsed().as_millis() as u64, Ordering::Relaxed);
        record_traffic_class_send(&self.state.metrics, traffic_class);
        drop(outbound_permit);
        drop(bulk_outbound_permit);

        Ok(())
    }

    async fn get_or_connect(&self, target: SocketAddr) -> Result<Arc<Mutex<TcpStream>>> {
        {
            let connections = self.state.outbound_connections.lock().await;
            if let Some(existing) = connections.get(&target) {
                return Ok(Arc::clone(existing));
            }
        }

        let stream = tokio::time::timeout(self.state.connect_timeout, TcpStream::connect(target))
            .await
            .map_err(|_| {
                self.state
                    .metrics
                    .send_timeout_count
                    .fetch_add(1, Ordering::Relaxed);
                AgentError::Network(format!("Timed out connecting to tensor peer {}", target))
            })?
            .map_err(|e| {
                AgentError::Network(format!(
                    "Failed to connect to tensor peer {}: {}",
                    target, e
                ))
            })?;
        stream.set_nodelay(true).map_err(|e| {
            AgentError::Network(format!("Failed to set TCP_NODELAY on {}: {}", target, e))
        })?;
        let stream = Arc::new(Mutex::new(stream));

        let mut connections = self.state.outbound_connections.lock().await;
        let was_present = connections.contains_key(&target);
        let entry = connections
            .entry(target)
            .or_insert_with(|| Arc::clone(&stream));
        if !was_present {
            self.state
                .metrics
                .current_outbound_connections
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(Arc::clone(entry))
    }

    async fn evict_connection(&self, target: SocketAddr) {
        let mut connections = self.state.outbound_connections.lock().await;
        if connections.remove(&target).is_some() {
            self.state
                .metrics
                .current_outbound_connections
                .fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub async fn recv(&self) -> Option<InboundTensorMessage> {
        loop {
            let notified = self.inbound_notify.notified();
            {
                let mut inbound = self.inbound.lock().await;
                if let Some(message) = inbound.pending_inbound.pop_front() {
                    self.record_receive_metrics(&message);
                    return Some(message);
                }
                if inbound.closed {
                    return None;
                }
            }
            notified.await;
        }
    }

    pub async fn recv_matching<F>(&self, mut predicate: F) -> Option<InboundTensorMessage>
    where
        F: FnMut(&InboundTensorMessage) -> bool,
    {
        loop {
            let notified = self.inbound_notify.notified();
            {
                let mut inbound = self.inbound.lock().await;
                if let Some(index) = inbound.pending_inbound.iter().position(&mut predicate) {
                    let message = inbound.pending_inbound.remove(index)?;
                    self.record_receive_metrics(&message);
                    return Some(message);
                }
                if inbound.closed {
                    return None;
                }
            }
            notified.await;
        }
    }

    fn record_receive_metrics(&self, inbound: &InboundTensorMessage) {
        self.state
            .metrics
            .receive_count
            .fetch_add(1, Ordering::Relaxed);
        self.state.metrics.receive_latency_ms.fetch_add(
            inbound.queued_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        self.state.metrics.receive_queue_wait_ms.fetch_add(
            inbound.queued_at.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        update_peak(
            &self.state.metrics.peak_inbound_queued_bytes,
            (self.state.inbound_queue_byte_capacity
                - self.state.inbound_queue_bytes.available_permits() as usize) as u64,
        );
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

fn resolve_advertised_addr(
    configured_addr: Option<SocketAddr>,
    local_port: u16,
) -> Result<SocketAddr> {
    match configured_addr {
        Some(addr) if addr.port() != 0 => Ok(addr),
        Some(addr) => Ok(SocketAddr::new(addr.ip(), local_port)),
        None => Ok(SocketAddr::new(resolve_advertised_ip()?, local_port)),
    }
}

pub fn parse_tensor_plane_bind_addr_env() -> Option<SocketAddr> {
    std::env::var("MESHNET_TENSOR_BIND_ADDR")
        .ok()
        .and_then(|value| value.parse().ok())
}

pub fn parse_tensor_plane_advertised_addr_env() -> Option<SocketAddr> {
    std::env::var("MESHNET_TENSOR_ADVERTISED_ADDR")
        .ok()
        .and_then(|value| value.parse().ok())
}

fn sanitized_config(config: TensorPlaneConfig) -> Result<TensorPlaneConfig> {
    let max_message_bytes = config.max_message_bytes.max(1);
    let max_outbound_inflight_bytes = config.max_outbound_inflight_bytes.max(max_message_bytes);
    let max_send_bandwidth_bytes_per_sec = if config.max_send_bandwidth_bytes_per_sec == 0 {
        config
            .profile
            .default_bandwidth_bytes_per_sec(max_message_bytes)
    } else {
        config.max_send_bandwidth_bytes_per_sec
    };

    Ok(TensorPlaneConfig {
        max_message_bytes,
        max_inbound_messages: config.max_inbound_messages.max(1),
        max_inbound_queued_bytes: config.max_inbound_queued_bytes.max(1),
        max_outbound_inflight_bytes,
        max_send_bandwidth_bytes_per_sec,
        ..config
    })
}

fn update_peak(metric: &AtomicU64, value: u64) {
    let mut current = metric.load(Ordering::Relaxed);
    while value > current {
        match metric.compare_exchange(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

fn compute_reserved_priority_outbound_inflight_bytes(
    outbound_inflight_byte_capacity: usize,
    max_message_bytes: usize,
) -> usize {
    if outbound_inflight_byte_capacity <= max_message_bytes {
        return 0;
    }

    let target_reserve = (outbound_inflight_byte_capacity / 4).max(max_message_bytes);
    let max_reserve = outbound_inflight_byte_capacity.saturating_sub(max_message_bytes);
    target_reserve.min(max_reserve)
}

fn compute_bulk_bandwidth_bytes_per_sec(
    max_send_bandwidth_bytes_per_sec: u64,
    max_message_bytes: usize,
) -> u64 {
    if max_send_bandwidth_bytes_per_sec <= max_message_bytes as u64 {
        return max_send_bandwidth_bytes_per_sec.max(1);
    }

    let scaled = max_send_bandwidth_bytes_per_sec
        .saturating_mul(DEFAULT_BULK_BANDWIDTH_SHARE_NUMERATOR)
        / DEFAULT_BULK_BANDWIDTH_SHARE_DENOMINATOR;
    scaled
        .max(max_message_bytes as u64)
        .min(max_send_bandwidth_bytes_per_sec)
        .max(1)
}

fn record_phase_bytes(
    metrics: &TensorPlaneMetrics,
    phase: AllReducePhase,
    bytes: u64,
    outbound: bool,
) {
    let target = match (phase, outbound) {
        (AllReducePhase::ReduceScatter, true) => &metrics.reduce_scatter_bytes_sent,
        (AllReducePhase::ReduceScatter, false) => &metrics.reduce_scatter_bytes_received,
        (AllReducePhase::AllGather, true) => &metrics.all_gather_bytes_sent,
        (AllReducePhase::AllGather, false) => &metrics.all_gather_bytes_received,
        (AllReducePhase::Barrier, true) => &metrics.barrier_bytes_sent,
        (AllReducePhase::Barrier, false) => &metrics.barrier_bytes_received,
    };
    target.fetch_add(bytes, Ordering::Relaxed);
}

fn record_traffic_class_send(metrics: &TensorPlaneMetrics, traffic_class: TensorTrafficClass) {
    match traffic_class {
        TensorTrafficClass::LatencyCritical => metrics
            .latency_critical_send_count
            .fetch_add(1, Ordering::Relaxed),
        TensorTrafficClass::Interactive => metrics
            .interactive_send_count
            .fetch_add(1, Ordering::Relaxed),
        TensorTrafficClass::Bulk => metrics.bulk_send_count.fetch_add(1, Ordering::Relaxed),
    };
}

async fn read_tensor_message<T>(
    io: &mut T,
    max_message_bytes: usize,
) -> std::io::Result<TensorMessage>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max_message_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Message size {} exceeds limit {}", len, max_message_bytes),
        ));
    }

    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    ciborium::from_reader(&buf[..])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

async fn write_tensor_message<T>(
    io: &mut T,
    message: &TensorMessage,
    max_message_bytes: usize,
) -> std::io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
{
    let mut buf = Vec::new();
    ciborium::into_writer(message, &mut buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if buf.len() > max_message_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Message size {} exceeds limit {}",
                buf.len(),
                max_message_bytes
            ),
        ));
    }

    io.write_all(&(buf.len() as u32).to_be_bytes()).await?;
    io.write_all(&buf).await?;
    io.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokio::time::{sleep, timeout};

    #[test]
    fn test_resolve_advertised_addr_preserves_explicit_port() {
        let addr = resolve_advertised_addr(
            Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4242)),
            9000,
        )
        .unwrap();
        assert_eq!(addr, SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4242));
    }

    #[test]
    fn test_resolve_advertised_addr_fills_missing_port_from_bind() {
        let addr = resolve_advertised_addr(
            Some(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)),
            9000,
        )
        .unwrap();
        assert_eq!(addr, SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9000));
    }

    fn test_message(size: usize) -> TensorMessage {
        TensorMessage {
            sender_position: 0,
            job_id: uuid::Uuid::nil(),
            layer_idx: 0,
            phase: super::super::tensor_message::AllReducePhase::ReduceScatter,
            step: 0,
            chunk_data: vec![1.0; size],
            chunk_shape: vec![size],
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    async fn wait_for_metric<F>(plane: &TensorPlane, predicate: F)
    where
        F: Fn(TensorPlaneMetricsSnapshot) -> bool,
    {
        timeout(Duration::from_secs(1), async {
            loop {
                let snapshot = plane.metrics_snapshot();
                if predicate(snapshot) {
                    return;
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_tensor_plane_clamps_invalid_limits_to_one() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            max_message_bytes: 0,
            max_inbound_messages: 0,
            max_inbound_queued_bytes: 0,
            max_outbound_inflight_bytes: 0,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();

        assert_eq!(plane.state.max_message_bytes, 1);
        assert_eq!(plane.state.inbound_queue_byte_capacity, 1);
        assert_eq!(plane.state.outbound_inflight_byte_capacity, 1);
        assert_eq!(plane.state.bulk_outbound_inflight_byte_capacity, 1);
    }

    #[tokio::test]
    async fn test_tensor_plane_rejects_inbound_when_queue_is_full() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            max_inbound_messages: 1,
            max_inbound_queued_bytes: 1024 * 1024,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();

        let addr = plane.local_addr();
        let first = test_message(16);
        let second = test_message(16);
        plane.send(addr, &first).await.unwrap();
        plane.send(addr, &second).await.unwrap();
        wait_for_metric(&plane, |snapshot| {
            snapshot.inbound_queue_full_rejections >= 1
        })
        .await;

        let received = timeout(Duration::from_secs(1), plane.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(received.tensor.chunk_data.len(), 16);

        let snapshot = plane.metrics_snapshot();
        assert!(snapshot.inbound_queue_full_rejections >= 1);
    }

    #[tokio::test]
    async fn test_tensor_plane_rejects_inbound_when_byte_budget_is_exhausted() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            max_inbound_messages: 4,
            max_inbound_queued_bytes: 100,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();

        let addr = plane.local_addr();
        let message = test_message(32);
        plane.send(addr, &message).await.unwrap();
        wait_for_metric(&plane, |snapshot| {
            snapshot.inbound_byte_budget_rejections >= 1
        })
        .await;
        assert!(timeout(Duration::from_millis(100), plane.recv())
            .await
            .is_err());
        let snapshot = plane.metrics_snapshot();
        assert!(snapshot.inbound_byte_budget_rejections >= 1);
    }

    #[tokio::test]
    async fn test_recv_matching_buffers_nonmatching_messages() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let addr = plane.local_addr();

        let mut first = test_message(4);
        first.step = 1;
        let mut second = test_message(4);
        second.step = 2;

        plane.send(addr, &first).await.unwrap();
        plane.send(addr, &second).await.unwrap();

        let matched = timeout(
            Duration::from_secs(1),
            plane.recv_matching(|inbound| inbound.tensor.step == 2),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(matched.tensor.step, 2);

        let buffered = timeout(Duration::from_secs(1), plane.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(buffered.tensor.step, 1);
    }

    #[tokio::test]
    async fn test_recv_matching_supports_multiple_concurrent_waiters() {
        let plane = Arc::new(
            TensorPlane::bind(TensorPlaneConfig::default())
                .await
                .unwrap(),
        );
        let addr = plane.local_addr();

        let waiter_for_two = {
            let plane = Arc::clone(&plane);
            tokio::spawn(async move {
                timeout(
                    Duration::from_secs(1),
                    plane.recv_matching(|inbound| inbound.tensor.step == 2),
                )
                .await
                .unwrap()
                .unwrap()
                .tensor
                .step
            })
        };
        sleep(Duration::from_millis(25)).await;
        let waiter_for_one = {
            let plane = Arc::clone(&plane);
            tokio::spawn(async move {
                timeout(
                    Duration::from_millis(250),
                    plane.recv_matching(|inbound| inbound.tensor.step == 1),
                )
                .await
                .unwrap()
                .unwrap()
                .tensor
                .step
            })
        };

        let mut first = test_message(4);
        first.step = 1;
        plane.send(addr, &first).await.unwrap();
        assert_eq!(waiter_for_one.await.unwrap(), 1);

        let mut second = test_message(4);
        second.step = 2;
        plane.send(addr, &second).await.unwrap();
        assert_eq!(waiter_for_two.await.unwrap(), 2);
    }

    #[tokio::test]
    async fn test_tensor_plane_reuses_outbound_connection_for_multiple_messages() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let addr = plane.local_addr();

        let mut first = test_message(4);
        first.step = 1;
        let mut second = test_message(4);
        second.step = 2;

        plane.send(addr, &first).await.unwrap();
        plane.send(addr, &second).await.unwrap();

        let first_received = timeout(
            Duration::from_secs(1),
            plane.recv_matching(|inbound| inbound.tensor.step == 1),
        )
        .await
        .unwrap()
        .unwrap();
        let second_received = timeout(
            Duration::from_secs(1),
            plane.recv_matching(|inbound| inbound.tensor.step == 2),
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(first_received.tensor.step, 1);
        assert_eq!(second_received.tensor.step, 2);
        assert_eq!(plane.metrics_snapshot().current_outbound_connections, 1);
    }

    #[tokio::test]
    async fn test_latency_critical_send_skips_bulk_bandwidth_throttle() {
        let plane = Arc::new(
            TensorPlane::bind(TensorPlaneConfig {
                // Keep the hard message limit above the test payload size so this
                // exercises bandwidth throttling rather than size rejection.
                max_message_bytes: 512,
                max_outbound_inflight_bytes: 1024,
                max_send_bandwidth_bytes_per_sec: 256,
                ..TensorPlaneConfig::default()
            })
            .await
            .unwrap(),
        );
        let addr = plane.local_addr();

        let first_bulk = test_message(64);
        plane.send(addr, &first_bulk).await.unwrap();

        let bulk_plane = Arc::clone(&plane);
        let throttled_bulk_send = tokio::spawn(async move {
            let bulk = test_message(48);
            bulk_plane.send(addr, &bulk).await.unwrap();
        });

        sleep(Duration::from_millis(25)).await;

        let priority_plane = Arc::clone(&plane);
        let priority_send = tokio::spawn(async move {
            let mut barrier = test_message(1);
            barrier.phase = AllReducePhase::Barrier;
            barrier.step = TensorMessage::BARRIER_STEP;
            let started = Instant::now();
            priority_plane.send(addr, &barrier).await.unwrap();
            started.elapsed()
        });

        let priority_elapsed = priority_send.await.unwrap();
        assert!(priority_elapsed < Duration::from_millis(200));
        throttled_bulk_send.await.unwrap();

        let snapshot = plane.metrics_snapshot();
        assert_eq!(snapshot.latency_critical_send_count, 1);
        assert_eq!(snapshot.bulk_send_count, 2);
        assert!(snapshot.outbound_bandwidth_wait_count >= 1);
    }

    #[test]
    fn test_tensor_plane_capabilities_report_priority_headroom() {
        let reserved = compute_reserved_priority_outbound_inflight_bytes(4096, 512);
        assert_eq!(reserved, 1024);

        let bulk_rate = compute_bulk_bandwidth_bytes_per_sec(4096, 512);
        assert_eq!(bulk_rate, 3072);
    }

    #[test]
    fn test_bandwidth_limiter_reserves_wait_when_tokens_exhausted() {
        let mut limiter = BandwidthLimiter::new(100, 100);
        assert_eq!(limiter.reserve_wait(100), Duration::ZERO);
        assert!(limiter.reserve_wait(100) >= Duration::from_millis(900));
    }
}
