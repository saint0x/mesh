use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::{BufMut, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tracing::warn;
use uuid::Uuid;

use crate::errors::{AgentError, Result};
use crate::inference::InferenceRuntimeMode;
use crate::provider::ExecutionProviderKind;

use super::tensor_message::{
    CollectiveLane, ServingFrame, ServingFrameHeader, ServingSlotKey, TensorTrafficClass,
};

pub const DATA_PLANE_ENDPOINT_PREFIX: &str = "dataplane://";
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 10 * 1024 * 1024;
pub const DEFAULT_MAX_INBOUND_MESSAGES: usize = 64;
pub const DEFAULT_MAX_INBOUND_QUEUED_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_OUTBOUND_INFLIGHT_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_CONCURRENT_OUTBOUND_STREAMS_PER_PEER: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportFallbackPolicy {
    PreserveFit,
    PreserveThroughput,
    PreserveLatency,
    PreserveResiliency,
}

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
    pub max_concurrent_outbound_streams_per_peer: usize,
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
            max_concurrent_outbound_streams_per_peer:
                DEFAULT_MAX_CONCURRENT_OUTBOUND_STREAMS_PER_PEER,
        }
    }
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
    pub bulk_transfer_bytes_sent: u64,
    pub bulk_transfer_bytes_received: u64,
    pub checkpoint_bytes_sent: u64,
    pub checkpoint_bytes_received: u64,
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
    pub max_concurrent_outbound_streams_per_peer: usize,
    pub peer_bulk_outbound_byte_capacity: usize,
    pub concurrent_receive_waiters: bool,
    pub prioritized_traffic_classes: bool,
    pub persistent_serving_peer_channels: bool,
    pub per_peer_bulk_fairness: bool,
    pub runtime_mode_aware_fallbacks: bool,
    pub provider_specialized_collectives: bool,
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
    bulk_transfer_bytes_sent: AtomicU64,
    bulk_transfer_bytes_received: AtomicU64,
    checkpoint_bytes_sent: AtomicU64,
    checkpoint_bytes_received: AtomicU64,
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
            bulk_transfer_bytes_sent: AtomicU64::new(0),
            bulk_transfer_bytes_received: AtomicU64::new(0),
            checkpoint_bytes_sent: AtomicU64::new(0),
            checkpoint_bytes_received: AtomicU64::new(0),
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
struct InboundServingFrame {
    frame: ServingFrame,
    remote_addr: SocketAddr,
    queued_at: Instant,
    _queued_bytes_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug, Default)]
struct ServingInboundState {
    pending_slots: HashMap<ServingSlotKey, VecDeque<InboundServingFrame>>,
    queued_messages: usize,
}

#[derive(Debug)]
struct TensorPlaneState {
    profile: TensorPlaneProfile,
    local_addr: SocketAddr,
    advertised_addr: SocketAddr,
    connect_timeout: Duration,
    io_timeout: Duration,
    max_message_bytes: usize,
    inbound_queue_message_capacity: usize,
    inbound_queue_byte_capacity: usize,
    outbound_inflight_byte_capacity: usize,
    reserved_priority_outbound_inflight_bytes: usize,
    bulk_outbound_inflight_byte_capacity: usize,
    max_send_bandwidth_bytes_per_sec: u64,
    bulk_send_bandwidth_bytes_per_sec: u64,
    peer_bulk_outbound_byte_capacity: usize,
    inbound_queue_bytes: Arc<Semaphore>,
    outbound_inflight_bytes: Arc<Semaphore>,
    priority_outbound_inflight_bytes: Arc<Semaphore>,
    bulk_outbound_inflight_bytes: Arc<Semaphore>,
    max_concurrent_outbound_streams_per_peer: usize,
    metrics: TensorPlaneMetrics,
    outbound_connections: Mutex<HashMap<SocketAddr, OutboundPeerChannels>>,
    peer_bulk_outbound_bytes: Mutex<HashMap<SocketAddr, Arc<Semaphore>>>,
    serving_sessions: Mutex<HashMap<ServingSessionKey, Arc<ServingSessionState>>>,
}

#[derive(Debug)]
struct OutboundPeerChannels {
    lanes: HashMap<CollectiveLane, LanePeerChannels>,
}

#[derive(Debug)]
struct LanePeerChannels {
    streams: Vec<Arc<Mutex<TcpStream>>>,
    next_stream_index: usize,
    pinned_for_serving: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ServingSessionKey {
    left_peer: SocketAddr,
    right_peer: SocketAddr,
}

#[derive(Debug, Clone, Copy)]
struct ServingLanePlan {
    lane: CollectiveLane,
    traffic_class: TensorTrafficClass,
    desired_stream_count: usize,
}

#[derive(Debug)]
struct ServingSessionState {
    session_id: Uuid,
    left_peer: SocketAddr,
    right_peer: SocketAddr,
}

#[derive(Debug, Clone, Copy)]
pub struct ServingReceiveSpec {
    pub collective_id: Uuid,
    pub lane: CollectiveLane,
    pub layer_idx: u32,
    pub step: u32,
    pub slot: u32,
    pub stream_id: u32,
    pub expected_sender_position: u32,
}

#[derive(Clone)]
pub struct ServingSessionTransport {
    state: Arc<TensorPlaneState>,
    inbound: Arc<Mutex<ServingInboundState>>,
    inbound_notify: Arc<Notify>,
    session: Arc<ServingSessionState>,
    reduce_scatter_plan: ServingLanePlan,
    all_gather_plan: ServingLanePlan,
    control_plan: ServingLanePlan,
    bulk_transfer_plan: ServingLanePlan,
    checkpoint_plan: ServingLanePlan,
}

#[derive(Debug)]
pub struct ServingBackgroundTransfer {
    lane: CollectiveLane,
    collective_id: Uuid,
    step: u32,
    slot: u32,
    stream_id: u32,
    join_handle: JoinHandle<Result<()>>,
}

impl ServingBackgroundTransfer {
    pub fn lane(&self) -> CollectiveLane {
        self.lane
    }

    pub fn collective_id(&self) -> Uuid {
        self.collective_id
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn slot(&self) -> u32 {
        self.slot
    }

    pub fn stream_id(&self) -> u32 {
        self.stream_id
    }

    pub async fn wait(self) -> Result<()> {
        self.join_handle.await.map_err(|error| {
            AgentError::Execution(format!(
                "Background {:?} transfer task for collective {} step {} slot {} failed to join: {}",
                self.lane, self.collective_id, self.step, self.slot, error
            ))
        })?
    }
}

impl std::fmt::Debug for ServingSessionTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingSessionTransport")
            .field("session_id", &self.session.session_id)
            .field("left_peer", &self.session.left_peer)
            .field("right_peer", &self.session.right_peer)
            .finish()
    }
}

impl ServingSessionTransport {
    pub fn session_id(&self) -> Uuid {
        self.session.session_id
    }

    pub fn stream_id_for(&self, lane: CollectiveLane, step: u32, slot: u32) -> u32 {
        let plan = match lane {
            CollectiveLane::ReduceScatter => self.reduce_scatter_plan,
            CollectiveLane::AllGather => self.all_gather_plan,
            CollectiveLane::Control => self.control_plan,
            CollectiveLane::BulkTransfer => self.bulk_transfer_plan,
            CollectiveLane::Checkpoint => self.checkpoint_plan,
        };
        let stream_count = plan.desired_stream_count.max(1) as u32;
        if stream_count == 1 {
            0
        } else {
            step.wrapping_add(slot) % stream_count
        }
    }

    pub async fn send_reduce_scatter_chunk(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
        chunk_shape: &[usize],
    ) -> Result<()> {
        self.send_frame(
            self.session.right_peer,
            ServingFrameHeader::new(
                self.session.session_id,
                collective_id,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::ReduceScatter,
                chunk_data.len() as u32,
                chunk_shape.len() as u32,
            ),
            chunk_data,
            chunk_shape,
            self.reduce_scatter_plan,
        )
        .await
    }

    pub async fn send_all_gather_chunk(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
        chunk_shape: &[usize],
    ) -> Result<()> {
        self.send_frame(
            self.session.right_peer,
            ServingFrameHeader::new(
                self.session.session_id,
                collective_id,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::AllGather,
                chunk_data.len() as u32,
                chunk_shape.len() as u32,
            ),
            chunk_data,
            chunk_shape,
            self.all_gather_plan,
        )
        .await
    }

    pub async fn send_control(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
        chunk_shape: &[usize],
    ) -> Result<()> {
        self.send_frame(
            self.session.right_peer,
            ServingFrameHeader::new(
                self.session.session_id,
                collective_id,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::Control,
                chunk_data.len() as u32,
                chunk_shape.len() as u32,
            ),
            chunk_data,
            chunk_shape,
            self.control_plan,
        )
        .await
    }

    pub async fn send_bulk_transfer(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
        chunk_shape: &[usize],
    ) -> Result<()> {
        self.send_frame(
            self.session.right_peer,
            ServingFrameHeader::new(
                self.session.session_id,
                collective_id,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::BulkTransfer,
                chunk_data.len() as u32,
                chunk_shape.len() as u32,
            ),
            chunk_data,
            chunk_shape,
            self.bulk_transfer_plan,
        )
        .await
    }

    pub async fn send_checkpoint(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
        chunk_shape: &[usize],
    ) -> Result<()> {
        self.send_frame(
            self.session.right_peer,
            ServingFrameHeader::new(
                self.session.session_id,
                collective_id,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::Checkpoint,
                chunk_data.len() as u32,
                chunk_shape.len() as u32,
            ),
            chunk_data,
            chunk_shape,
            self.checkpoint_plan,
        )
        .await
    }

    pub fn spawn_bulk_transfer(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: Vec<f32>,
        chunk_shape: Vec<usize>,
    ) -> ServingBackgroundTransfer {
        self.spawn_background_frame(
            collective_id,
            layer_idx,
            step,
            slot,
            stream_id,
            sender_position,
            chunk_data,
            chunk_shape,
            self.bulk_transfer_plan,
        )
    }

    pub fn spawn_checkpoint(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: Vec<f32>,
        chunk_shape: Vec<usize>,
    ) -> ServingBackgroundTransfer {
        self.spawn_background_frame(
            collective_id,
            layer_idx,
            step,
            slot,
            stream_id,
            sender_position,
            chunk_data,
            chunk_shape,
            self.checkpoint_plan,
        )
    }

    pub async fn recv_frame(&self, spec: ServingReceiveSpec) -> Result<ServingFrame> {
        recv_slot(
            &self.state,
            &self.inbound,
            &self.inbound_notify,
            self.session.session_id,
            spec,
        )
        .await
    }

    async fn send_frame(
        &self,
        target: SocketAddr,
        header: ServingFrameHeader,
        chunk_data: &[f32],
        chunk_shape: &[usize],
        plan: ServingLanePlan,
    ) -> Result<()> {
        send_serving_frame_on_plan(&self.state, target, header, chunk_data, chunk_shape, plan).await
    }

    fn spawn_background_frame(
        &self,
        collective_id: Uuid,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: Vec<f32>,
        chunk_shape: Vec<usize>,
        plan: ServingLanePlan,
    ) -> ServingBackgroundTransfer {
        let target = self.session.right_peer;
        let session_id = self.session.session_id;
        let state = Arc::clone(&self.state);
        let join_handle = tokio::spawn(async move {
            send_serving_frame_on_plan(
                &state,
                target,
                ServingFrameHeader::new(
                    session_id,
                    collective_id,
                    sender_position,
                    layer_idx,
                    step,
                    slot,
                    stream_id,
                    plan.lane,
                    chunk_data.len() as u32,
                    chunk_shape.len() as u32,
                ),
                &chunk_data,
                &chunk_shape,
                plan,
            )
            .await
        });
        ServingBackgroundTransfer {
            lane: plan.lane,
            collective_id,
            step,
            slot,
            stream_id,
            join_handle,
        }
    }
}

pub struct TensorPlane {
    state: Arc<TensorPlaneState>,
    inbound: Arc<Mutex<ServingInboundState>>,
    inbound_notify: Arc<Notify>,
    _accept_task: tokio::task::JoinHandle<()>,
}

impl TensorPlane {
    pub async fn bind(config: TensorPlaneConfig) -> Result<Self> {
        let config = sanitized_config(config);
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
            reserved_priority_outbound_budget(config.max_outbound_inflight_bytes);
        let bulk_outbound_inflight_byte_capacity = config
            .max_outbound_inflight_bytes
            .saturating_sub(reserved_priority_outbound_inflight_bytes)
            .max(config.max_message_bytes);
        let priority_outbound_inflight_bytes = Arc::new(Semaphore::new(
            reserved_priority_outbound_inflight_bytes
                .try_into()
                .map_err(|_| {
                    AgentError::Config(
                        "tensor priority outbound byte budget exceeds u32".to_string(),
                    )
                })?,
        ));
        let bulk_outbound_inflight_bytes = Arc::new(Semaphore::new(
            bulk_outbound_inflight_byte_capacity
                .try_into()
                .map_err(|_| {
                    AgentError::Config("tensor bulk outbound byte budget exceeds u32".to_string())
                })?,
        ));
        let bulk_send_bandwidth_bytes_per_sec =
            reserved_bulk_send_bandwidth(config.max_send_bandwidth_bytes_per_sec);
        let peer_bulk_outbound_byte_capacity =
            per_peer_bulk_outbound_budget(bulk_outbound_inflight_byte_capacity);
        let state = Arc::new(TensorPlaneState {
            profile: config.profile,
            local_addr,
            advertised_addr,
            connect_timeout: config.connect_timeout,
            io_timeout: config.io_timeout,
            max_message_bytes: config.max_message_bytes,
            inbound_queue_message_capacity: config.max_inbound_messages,
            inbound_queue_byte_capacity: config.max_inbound_queued_bytes,
            outbound_inflight_byte_capacity: config.max_outbound_inflight_bytes,
            reserved_priority_outbound_inflight_bytes,
            bulk_outbound_inflight_byte_capacity,
            max_send_bandwidth_bytes_per_sec: config.max_send_bandwidth_bytes_per_sec,
            bulk_send_bandwidth_bytes_per_sec,
            peer_bulk_outbound_byte_capacity,
            inbound_queue_bytes,
            outbound_inflight_bytes,
            priority_outbound_inflight_bytes,
            bulk_outbound_inflight_bytes,
            max_concurrent_outbound_streams_per_peer: config
                .max_concurrent_outbound_streams_per_peer,
            metrics: TensorPlaneMetrics::new(),
            outbound_connections: Mutex::new(HashMap::new()),
            peer_bulk_outbound_bytes: Mutex::new(HashMap::new()),
            serving_sessions: Mutex::new(HashMap::new()),
        });
        let inbound = Arc::new(Mutex::new(ServingInboundState::default()));
        let inbound_notify = Arc::new(Notify::new());
        let accept_state = Arc::clone(&state);
        let accept_inbound = Arc::clone(&inbound);
        let accept_notify = Arc::clone(&inbound_notify);
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((mut stream, remote_addr)) => {
                        let state = Arc::clone(&accept_state);
                        let inbound = Arc::clone(&accept_inbound);
                        let notify = Arc::clone(&accept_notify);
                        tokio::spawn(async move {
                            loop {
                                match tokio::time::timeout(
                                    state.io_timeout,
                                    read_serving_frame(&mut stream, state.max_message_bytes),
                                )
                                .await
                                {
                                    Ok(Ok(frame)) => {
                                        let queued_bytes = frame.header.size_bytes().max(1);
                                        let queued_bytes_u32: u32 = match queued_bytes.try_into() {
                                            Ok(value) => value,
                                            Err(_) => {
                                                state
                                                    .metrics
                                                    .oversized_message_rejections
                                                    .fetch_add(1, Ordering::Relaxed);
                                                return;
                                            }
                                        };
                                        let permit = match state
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
                                                continue;
                                            }
                                        };

                                        let mut inbound_guard = inbound.lock().await;
                                        if inbound_guard.queued_messages
                                            >= state.inbound_queue_message_capacity
                                        {
                                            state
                                                .metrics
                                                .inbound_queue_full_rejections
                                                .fetch_add(1, Ordering::Relaxed);
                                            continue;
                                        }
                                        let slot = frame.header.slot_key();
                                        inbound_guard
                                            .pending_slots
                                            .entry(slot)
                                            .or_default()
                                            .push_back(InboundServingFrame {
                                                frame,
                                                remote_addr,
                                                queued_at: Instant::now(),
                                                _queued_bytes_permit: Some(permit),
                                            });
                                        inbound_guard.queued_messages += 1;
                                        drop(inbound_guard);

                                        state
                                            .metrics
                                            .bytes_received
                                            .fetch_add(queued_bytes as u64, Ordering::Relaxed);
                                        record_phase_bytes(
                                            &state.metrics,
                                            slot.lane,
                                            queued_bytes as u64,
                                            false,
                                        );
                                        update_peak(
                                            &state.metrics.peak_inbound_queued_bytes,
                                            (state.inbound_queue_byte_capacity
                                                - state.inbound_queue_bytes.available_permits()
                                                    as usize)
                                                as u64,
                                        );
                                        notify.notify_waiters();
                                    }
                                    Ok(Err(error)) => {
                                        if !matches!(
                                            error.kind(),
                                            std::io::ErrorKind::UnexpectedEof
                                                | std::io::ErrorKind::ConnectionReset
                                                | std::io::ErrorKind::BrokenPipe
                                        ) {
                                            state
                                                .metrics
                                                .oversized_message_rejections
                                                .fetch_add(1, Ordering::Relaxed);
                                            warn!(
                                                error = %error,
                                                remote_addr = %remote_addr,
                                                "serving dataplane receive failed"
                                            );
                                        }
                                        return;
                                    }
                                    Err(_) => {
                                        state
                                            .metrics
                                            .receive_timeout_count
                                            .fetch_add(1, Ordering::Relaxed);
                                        warn!(
                                            remote_addr = %remote_addr,
                                            "serving dataplane receive timed out"
                                        );
                                        return;
                                    }
                                }
                            }
                        });
                    }
                    Err(error) => warn!(error = %error, "serving dataplane accept failed"),
                }
            }
        });

        Ok(Self {
            state,
            inbound,
            inbound_notify,
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
            max_concurrent_outbound_streams_per_peer: self
                .state
                .max_concurrent_outbound_streams_per_peer,
            peer_bulk_outbound_byte_capacity: self.state.peer_bulk_outbound_byte_capacity,
            concurrent_receive_waiters: true,
            prioritized_traffic_classes: true,
            persistent_serving_peer_channels: true,
            per_peer_bulk_fairness: true,
            runtime_mode_aware_fallbacks: true,
            provider_specialized_collectives: true,
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
            bulk_transfer_bytes_sent: self
                .state
                .metrics
                .bulk_transfer_bytes_sent
                .load(Ordering::Relaxed),
            bulk_transfer_bytes_received: self
                .state
                .metrics
                .bulk_transfer_bytes_received
                .load(Ordering::Relaxed),
            checkpoint_bytes_sent: self
                .state
                .metrics
                .checkpoint_bytes_sent
                .load(Ordering::Relaxed),
            checkpoint_bytes_received: self
                .state
                .metrics
                .checkpoint_bytes_received
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

    pub async fn prepare_serving_peer_channels(
        &self,
        peers: &[SocketAddr],
        runtime_mode: InferenceRuntimeMode,
        provider: ExecutionProviderKind,
    ) -> Result<()> {
        let hot_lane_plans = serving_lane_plans(
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        for &peer in peers {
            for plan in hot_lane_plans {
                self.ensure_connection_pool(peer, plan.lane, plan.desired_stream_count, true)
                    .await?;
            }
        }
        Ok(())
    }

    pub async fn serving_transport_for_neighbors(
        &self,
        left_peer: SocketAddr,
        right_peer: SocketAddr,
        runtime_mode: InferenceRuntimeMode,
        provider: ExecutionProviderKind,
    ) -> Result<ServingSessionTransport> {
        self.prepare_serving_peer_channels(&[left_peer, right_peer], runtime_mode, provider)
            .await?;
        let session = self.durable_serving_session(left_peer, right_peer).await;
        Ok(ServingSessionTransport {
            state: Arc::clone(&self.state),
            inbound: Arc::clone(&self.inbound),
            inbound_notify: Arc::clone(&self.inbound_notify),
            session,
            reduce_scatter_plan: lane_plan(
                CollectiveLane::ReduceScatter,
                runtime_mode,
                provider,
                self.state.max_concurrent_outbound_streams_per_peer,
            ),
            all_gather_plan: lane_plan(
                CollectiveLane::AllGather,
                runtime_mode,
                provider,
                self.state.max_concurrent_outbound_streams_per_peer,
            ),
            control_plan: lane_plan(
                CollectiveLane::Control,
                runtime_mode,
                provider,
                self.state.max_concurrent_outbound_streams_per_peer,
            ),
            bulk_transfer_plan: lane_plan(
                CollectiveLane::BulkTransfer,
                runtime_mode,
                provider,
                self.state.max_concurrent_outbound_streams_per_peer,
            ),
            checkpoint_plan: lane_plan(
                CollectiveLane::Checkpoint,
                runtime_mode,
                provider,
                self.state.max_concurrent_outbound_streams_per_peer,
            ),
        })
    }

    async fn ensure_connection_pool(
        &self,
        target: SocketAddr,
        lane: CollectiveLane,
        desired_stream_count: usize,
        mark_persistent: bool,
    ) -> Result<Arc<Mutex<TcpStream>>> {
        ensure_connection_pool(
            &self.state,
            target,
            lane,
            desired_stream_count,
            mark_persistent,
        )
        .await
    }

    async fn durable_serving_session(
        &self,
        left_peer: SocketAddr,
        right_peer: SocketAddr,
    ) -> Arc<ServingSessionState> {
        let key = ServingSessionKey {
            left_peer,
            right_peer,
        };
        let mut sessions = self.state.serving_sessions.lock().await;
        Arc::clone(sessions.entry(key).or_insert_with(|| {
            Arc::new(ServingSessionState {
                session_id: Uuid::new_v4(),
                left_peer,
                right_peer,
            })
        }))
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

fn sanitized_config(config: TensorPlaneConfig) -> TensorPlaneConfig {
    let max_message_bytes = config.max_message_bytes.max(1);
    let max_outbound_inflight_bytes = config.max_outbound_inflight_bytes.max(max_message_bytes);

    TensorPlaneConfig {
        max_message_bytes,
        max_inbound_messages: config.max_inbound_messages.max(1),
        max_inbound_queued_bytes: config.max_inbound_queued_bytes.max(1),
        max_outbound_inflight_bytes,
        max_concurrent_outbound_streams_per_peer: config
            .max_concurrent_outbound_streams_per_peer
            .max(1),
        ..config
    }
}

async fn send_serving_frame_on_plan(
    state: &Arc<TensorPlaneState>,
    target: SocketAddr,
    header: ServingFrameHeader,
    chunk_data: &[f32],
    chunk_shape: &[usize],
    plan: ServingLanePlan,
) -> Result<()> {
    let message_bytes = header.size_bytes().max(1);
    if message_bytes > state.max_message_bytes {
        state
            .metrics
            .oversized_message_rejections
            .fetch_add(1, Ordering::Relaxed);
        return Err(AgentError::Network(format!(
            "Serving frame size {} exceeds limit {}",
            message_bytes, state.max_message_bytes
        )));
    }

    let message_bytes_u32: u32 = message_bytes.try_into().map_err(|_| {
        AgentError::Network("Serving frame too large for byte accounting".to_string())
    })?;
    let send_started = Instant::now();
    let wait_started = Instant::now();
    let outbound_permit = match state
        .outbound_inflight_bytes
        .clone()
        .try_acquire_many_owned(message_bytes_u32)
    {
        Ok(permit) => permit,
        Err(_) => {
            state
                .metrics
                .outbound_backpressure_wait_count
                .fetch_add(1, Ordering::Relaxed);
            let permit = state
                .outbound_inflight_bytes
                .clone()
                .acquire_many_owned(message_bytes_u32)
                .await
                .map_err(|_| {
                    AgentError::Network(
                        "Serving dataplane outbound byte budget unexpectedly closed".to_string(),
                    )
                })?;
            state
                .metrics
                .outbound_backpressure_wait_ms
                .fetch_add(wait_started.elapsed().as_millis() as u64, Ordering::Relaxed);
            permit
        }
    };
    update_peak(
        &state.metrics.peak_outbound_inflight_bytes,
        (state.outbound_inflight_byte_capacity
            - state.outbound_inflight_bytes.available_permits() as usize) as u64,
    );

    let class_permit =
        acquire_lane_budget(state, target, plan.traffic_class, message_bytes).await?;
    let stream =
        ensure_connection_pool(state, target, plan.lane, plan.desired_stream_count, true).await?;
    let send_result = {
        let mut stream_guard = stream.lock().await;
        tokio::time::timeout(
            state.io_timeout,
            write_serving_frame(&mut *stream_guard, header, chunk_data, chunk_shape),
        )
        .await
    };

    match send_result {
        Ok(Ok(())) => {
            state
                .metrics
                .bytes_sent
                .fetch_add(message_bytes as u64, Ordering::Relaxed);
            record_phase_bytes(&state.metrics, header.lane, message_bytes as u64, true);
            state.metrics.send_count.fetch_add(1, Ordering::Relaxed);
            state
                .metrics
                .send_latency_ms
                .fetch_add(send_started.elapsed().as_millis() as u64, Ordering::Relaxed);
            record_traffic_class_send(&state.metrics, plan.traffic_class);
            drop(class_permit);
            drop(outbound_permit);
            Ok(())
        }
        Ok(Err(error)) => {
            evict_connection(state, target, plan.lane).await;
            drop(class_permit);
            drop(outbound_permit);
            Err(AgentError::Network(format!(
                "Failed to send serving frame to {}: {}",
                target, error
            )))
        }
        Err(_) => {
            state
                .metrics
                .send_timeout_count
                .fetch_add(1, Ordering::Relaxed);
            evict_connection(state, target, plan.lane).await;
            drop(class_permit);
            drop(outbound_permit);
            Err(AgentError::Network(format!(
                "Timed out sending serving frame to {}",
                target
            )))
        }
    }
}

async fn recv_slot(
    state: &Arc<TensorPlaneState>,
    inbound: &Arc<Mutex<ServingInboundState>>,
    notify: &Arc<Notify>,
    session_id: Uuid,
    spec: ServingReceiveSpec,
) -> Result<ServingFrame> {
    let slot_key = ServingSlotKey {
        session_id,
        collective_id: spec.collective_id,
        lane: spec.lane,
        layer_idx: spec.layer_idx,
        step: spec.step,
        slot: spec.slot,
        stream_id: spec.stream_id,
    };

    loop {
        let notified = notify.notified();
        {
            let mut inbound_guard = inbound.lock().await;
            if let Some(queue) = inbound_guard.pending_slots.get_mut(&slot_key) {
                let message = queue.pop_front();
                let queue_empty = queue.is_empty();
                if queue_empty {
                    inbound_guard.pending_slots.remove(&slot_key);
                }
                if let Some(message) = message {
                    inbound_guard.queued_messages = inbound_guard.queued_messages.saturating_sub(1);
                    drop(inbound_guard);

                    if message.frame.header.sender_position != spec.expected_sender_position {
                        return Err(AgentError::Network(format!(
                            "Serving frame sender mismatch for session {} lane {:?} step {}: got {}, expected {} from {}",
                            session_id,
                            spec.lane,
                            spec.step,
                            message.frame.header.sender_position,
                            spec.expected_sender_position,
                            message.remote_addr
                        )));
                    }

                    state.metrics.receive_count.fetch_add(1, Ordering::Relaxed);
                    let wait_ms = message.queued_at.elapsed().as_millis() as u64;
                    state
                        .metrics
                        .receive_latency_ms
                        .fetch_add(wait_ms, Ordering::Relaxed);
                    state
                        .metrics
                        .receive_queue_wait_ms
                        .fetch_add(wait_ms, Ordering::Relaxed);
                    return Ok(message.frame);
                }
            }
        }
        notified.await;
    }
}

async fn ensure_connection_pool(
    state: &Arc<TensorPlaneState>,
    target: SocketAddr,
    lane: CollectiveLane,
    desired_stream_count: usize,
    mark_persistent: bool,
) -> Result<Arc<Mutex<TcpStream>>> {
    let desired_stream_count = desired_stream_count
        .max(1)
        .min(state.max_concurrent_outbound_streams_per_peer);

    loop {
        {
            let mut connections = state.outbound_connections.lock().await;
            if let Some(pool) = connections.get_mut(&target) {
                let lane_pool = pool.lanes.entry(lane).or_insert_with(|| LanePeerChannels {
                    streams: Vec::new(),
                    next_stream_index: 0,
                    pinned_for_serving: false,
                });
                if mark_persistent {
                    lane_pool.pinned_for_serving = true;
                }
                if lane_pool.streams.len() >= desired_stream_count {
                    let idx = lane_pool.next_stream_index % lane_pool.streams.len();
                    lane_pool.next_stream_index =
                        (lane_pool.next_stream_index + 1) % lane_pool.streams.len();
                    return Ok(Arc::clone(&lane_pool.streams[idx]));
                }
            }
        }

        let stream = connect_stream(state, target).await?;
        let mut connections = state.outbound_connections.lock().await;
        let pool = connections
            .entry(target)
            .or_insert_with(|| OutboundPeerChannels {
                lanes: HashMap::new(),
            });
        let lane_pool = pool.lanes.entry(lane).or_insert_with(|| LanePeerChannels {
            streams: Vec::new(),
            next_stream_index: 0,
            pinned_for_serving: false,
        });
        if mark_persistent {
            lane_pool.pinned_for_serving = true;
        }
        if lane_pool.streams.len() < desired_stream_count {
            lane_pool.streams.push(Arc::clone(&stream));
            state
                .metrics
                .current_outbound_connections
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn connect_stream(
    state: &Arc<TensorPlaneState>,
    target: SocketAddr,
) -> Result<Arc<Mutex<TcpStream>>> {
    let stream = tokio::time::timeout(state.connect_timeout, TcpStream::connect(target))
        .await
        .map_err(|_| {
            state
                .metrics
                .send_timeout_count
                .fetch_add(1, Ordering::Relaxed);
            AgentError::Network(format!(
                "Timed out connecting to serving dataplane peer {}",
                target
            ))
        })?
        .map_err(|e| {
            AgentError::Network(format!(
                "Failed to connect to serving dataplane peer {}: {}",
                target, e
            ))
        })?;
    stream.set_nodelay(true).map_err(|e| {
        AgentError::Network(format!("Failed to set TCP_NODELAY on {}: {}", target, e))
    })?;
    Ok(Arc::new(Mutex::new(stream)))
}

async fn evict_connection(state: &Arc<TensorPlaneState>, target: SocketAddr, lane: CollectiveLane) {
    let mut connections = state.outbound_connections.lock().await;
    if let Some(pool) = connections.get_mut(&target) {
        if let Some(lane_pool) = pool.lanes.remove(&lane) {
            state
                .metrics
                .current_outbound_connections
                .fetch_sub(lane_pool.streams.len() as u64, Ordering::Relaxed);
        }
        if pool.lanes.is_empty() {
            connections.remove(&target);
        }
    }
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

fn preferred_serving_stream_count(
    runtime_mode: InferenceRuntimeMode,
    provider: ExecutionProviderKind,
    max_streams: usize,
) -> usize {
    let max_streams = max_streams.max(1);
    match (provider, runtime_mode) {
        (ExecutionProviderKind::Cuda, InferenceRuntimeMode::ThroughputFirst) => max_streams,
        (ExecutionProviderKind::Metal, InferenceRuntimeMode::ThroughputFirst) => 2.min(max_streams),
        (ExecutionProviderKind::Metal, InferenceRuntimeMode::LatencyFirst) => 2.min(max_streams),
        (ExecutionProviderKind::Cpu, _) | (_, InferenceRuntimeMode::FitFirst) => 1,
        _ => 1.max(max_streams / 2),
    }
}

fn lane_plan(
    lane: CollectiveLane,
    runtime_mode: InferenceRuntimeMode,
    provider: ExecutionProviderKind,
    max_streams: usize,
) -> ServingLanePlan {
    let bulk_streams = preferred_serving_stream_count(runtime_mode, provider, max_streams);
    let interactive_streams = bulk_streams.min(2).max(1);
    let control_streams = 1;
    let checkpoint_streams = match (provider, runtime_mode) {
        (ExecutionProviderKind::Cuda, InferenceRuntimeMode::ThroughputFirst) => interactive_streams,
        (ExecutionProviderKind::Metal, InferenceRuntimeMode::ThroughputFirst) => {
            interactive_streams
        }
        _ => 1,
    };
    let desired_stream_count = match lane {
        CollectiveLane::ReduceScatter => bulk_streams.max(1),
        CollectiveLane::AllGather => interactive_streams,
        CollectiveLane::Control => control_streams,
        CollectiveLane::BulkTransfer => interactive_streams,
        CollectiveLane::Checkpoint => checkpoint_streams,
    };
    ServingLanePlan {
        lane,
        traffic_class: lane.traffic_class(),
        desired_stream_count,
    }
}

fn serving_lane_plans(
    runtime_mode: InferenceRuntimeMode,
    provider: ExecutionProviderKind,
    max_streams: usize,
) -> [ServingLanePlan; 5] {
    [
        lane_plan(
            CollectiveLane::ReduceScatter,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(
            CollectiveLane::AllGather,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(CollectiveLane::Control, runtime_mode, provider, max_streams),
        lane_plan(
            CollectiveLane::BulkTransfer,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(
            CollectiveLane::Checkpoint,
            runtime_mode,
            provider,
            max_streams,
        ),
    ]
}

fn reserved_priority_outbound_budget(total: usize) -> usize {
    (total / 4).max(DEFAULT_MAX_MESSAGE_BYTES).min(total.max(1))
}

fn reserved_bulk_send_bandwidth(total: u64) -> u64 {
    (total / 2).max(DEFAULT_MAX_MESSAGE_BYTES as u64)
}

fn per_peer_bulk_outbound_budget(total: usize) -> usize {
    (total / 2).max(DEFAULT_MAX_MESSAGE_BYTES)
}

async fn acquire_lane_budget(
    state: &Arc<TensorPlaneState>,
    target: SocketAddr,
    traffic_class: TensorTrafficClass,
    message_bytes: usize,
) -> Result<Vec<OwnedSemaphorePermit>> {
    let mut permits = Vec::new();
    match traffic_class {
        TensorTrafficClass::LatencyCritical | TensorTrafficClass::Interactive => {
            permits.push(
                acquire_bytes_permit(
                    Arc::clone(&state.priority_outbound_inflight_bytes),
                    message_bytes,
                    "priority outbound byte budget",
                )
                .await?,
            );
        }
        TensorTrafficClass::Bulk => {
            permits.push(
                acquire_bytes_permit(
                    Arc::clone(&state.bulk_outbound_inflight_bytes),
                    message_bytes,
                    "bulk outbound byte budget",
                )
                .await?,
            );
            let per_peer_budget = {
                let mut budgets = state.peer_bulk_outbound_bytes.lock().await;
                Arc::clone(budgets.entry(target).or_insert_with(|| {
                    Arc::new(Semaphore::new(state.peer_bulk_outbound_byte_capacity))
                }))
            };
            permits.push(
                acquire_bytes_permit(
                    per_peer_budget,
                    message_bytes,
                    "per-peer bulk outbound byte budget",
                )
                .await?,
            );
        }
    }
    Ok(permits)
}

async fn acquire_bytes_permit(
    semaphore: Arc<Semaphore>,
    bytes: usize,
    budget_name: &str,
) -> Result<OwnedSemaphorePermit> {
    semaphore
        .acquire_many_owned(bytes.try_into().map_err(|_| {
            AgentError::Config(format!("{budget_name} exceeds supported permit size"))
        })?)
        .await
        .map_err(|_| AgentError::Execution(format!("{budget_name} semaphore was closed")))
}

fn record_phase_bytes(metrics: &TensorPlaneMetrics, lane: CollectiveLane, bytes: u64, sent: bool) {
    let metric = match (lane, sent) {
        (CollectiveLane::ReduceScatter, true) => &metrics.reduce_scatter_bytes_sent,
        (CollectiveLane::ReduceScatter, false) => &metrics.reduce_scatter_bytes_received,
        (CollectiveLane::AllGather, true) => &metrics.all_gather_bytes_sent,
        (CollectiveLane::AllGather, false) => &metrics.all_gather_bytes_received,
        (CollectiveLane::Control, true) => &metrics.barrier_bytes_sent,
        (CollectiveLane::Control, false) => &metrics.barrier_bytes_received,
        (CollectiveLane::BulkTransfer, true) => &metrics.bulk_transfer_bytes_sent,
        (CollectiveLane::BulkTransfer, false) => &metrics.bulk_transfer_bytes_received,
        (CollectiveLane::Checkpoint, true) => &metrics.checkpoint_bytes_sent,
        (CollectiveLane::Checkpoint, false) => &metrics.checkpoint_bytes_received,
    };
    metric.fetch_add(bytes, Ordering::Relaxed);
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

async fn write_serving_frame<W>(
    writer: &mut W,
    header: ServingFrameHeader,
    chunk_data: &[f32],
    chunk_shape: &[usize],
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    writer.write_all(&header.encode_binary()).await?;
    let payload = encode_f32_slice_be(chunk_data);
    let shape = encode_usize_slice_be(chunk_shape);
    writer.write_all(payload.as_ref()).await?;
    writer.write_all(shape.as_ref()).await?;
    Ok(())
}

async fn read_serving_frame<R>(
    reader: &mut R,
    max_message_bytes: usize,
) -> std::io::Result<ServingFrame>
where
    R: AsyncRead + Unpin,
{
    let mut header_buf = [0u8; ServingFrameHeader::fixed_size()];
    reader.read_exact(&mut header_buf).await?;
    let header = ServingFrameHeader::decode_binary(&header_buf)?;
    if header.size_bytes() > max_message_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "serving frame size {} exceeds limit {}",
                header.size_bytes(),
                max_message_bytes
            ),
        ));
    }
    let chunk_data = read_f32_vec_be(reader, header.element_count as usize).await?;
    let chunk_shape = read_usize_vec_be(reader, header.shape_len as usize).await?;
    Ok(ServingFrame {
        header,
        chunk_data,
        chunk_shape,
    })
}

fn encode_f32_slice_be(values: &[f32]) -> BytesMut {
    let mut buf = BytesMut::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        buf.put_u32(value.to_bits());
    }
    buf
}

fn encode_usize_slice_be(values: &[usize]) -> BytesMut {
    let mut buf = BytesMut::with_capacity(values.len() * std::mem::size_of::<u64>());
    for value in values {
        buf.put_u64(*value as u64);
    }
    buf
}

async fn read_f32_vec_be<R>(reader: &mut R, len: usize) -> std::io::Result<Vec<f32>>
where
    R: AsyncRead + Unpin,
{
    let mut buf = vec![0u8; len * std::mem::size_of::<f32>()];
    reader.read_exact(&mut buf).await?;
    let mut out = Vec::with_capacity(len);
    for chunk in buf.chunks_exact(std::mem::size_of::<f32>()) {
        out.push(f32::from_bits(u32::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
        ])));
    }
    Ok(out)
}

async fn read_usize_vec_be<R>(reader: &mut R, len: usize) -> std::io::Result<Vec<usize>>
where
    R: AsyncRead + Unpin,
{
    let mut buf = vec![0u8; len * std::mem::size_of::<u64>()];
    reader.read_exact(&mut buf).await?;
    let mut out = Vec::with_capacity(len);
    for chunk in buf.chunks_exact(std::mem::size_of::<u64>()) {
        out.push(u64::from_be_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]) as usize);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, timeout};

    async fn wait_for_metric<F>(plane: &TensorPlane, predicate: F)
    where
        F: Fn(TensorPlaneMetricsSnapshot) -> bool,
    {
        for _ in 0..40 {
            if predicate(plane.metrics_snapshot()) {
                return;
            }
            sleep(Duration::from_millis(25)).await;
        }
        panic!("metric predicate not satisfied");
    }

    #[tokio::test]
    async fn test_serving_slot_receive_is_direct_and_preserves_nonmatching_frames() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let session_a = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();
        let session_b = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();

        session_a
            .send_reduce_scatter_chunk(Uuid::new_v4(), 1, 0, 0, 0, 9, &[1.0, 2.0], &[2])
            .await
            .unwrap();
        let collective_b = Uuid::new_v4();
        session_b
            .send_reduce_scatter_chunk(collective_b, 1, 0, 0, 0, 7, &[3.0, 4.0], &[2])
            .await
            .unwrap();

        let frame_b = timeout(
            Duration::from_secs(1),
            session_b.recv_frame(ServingReceiveSpec {
                collective_id: collective_b,
                lane: CollectiveLane::ReduceScatter,
                layer_idx: 1,
                step: 0,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 7,
            }),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(frame_b.chunk_data, vec![3.0, 4.0]);

        let metrics = plane.metrics_snapshot();
        assert!(metrics.current_inbound_queued_bytes > 0);
    }

    #[tokio::test]
    async fn test_serving_session_supports_multiple_concurrent_waiters_on_distinct_slots() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let session = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();
        let collective_id = Uuid::new_v4();

        let waiter_one = {
            let session = session.clone();
            tokio::spawn(async move {
                session
                    .recv_frame(ServingReceiveSpec {
                        collective_id,
                        lane: CollectiveLane::ReduceScatter,
                        layer_idx: 3,
                        step: 1,
                        slot: 1,
                        stream_id: 0,
                        expected_sender_position: 11,
                    })
                    .await
                    .unwrap()
            })
        };
        let waiter_two = {
            let session = session.clone();
            tokio::spawn(async move {
                session
                    .recv_frame(ServingReceiveSpec {
                        collective_id,
                        lane: CollectiveLane::ReduceScatter,
                        layer_idx: 3,
                        step: 1,
                        slot: 2,
                        stream_id: 0,
                        expected_sender_position: 12,
                    })
                    .await
                    .unwrap()
            })
        };

        session
            .send_reduce_scatter_chunk(collective_id, 3, 1, 2, 0, 12, &[6.0], &[1])
            .await
            .unwrap();
        session
            .send_reduce_scatter_chunk(collective_id, 3, 1, 1, 0, 11, &[5.0], &[1])
            .await
            .unwrap();

        let one = waiter_one.await.unwrap();
        let two = waiter_two.await.unwrap();
        assert_eq!(one.chunk_data, vec![5.0]);
        assert_eq!(two.chunk_data, vec![6.0]);
    }

    #[tokio::test]
    async fn test_prepare_serving_peer_channels_opens_multiple_streams_for_throughput() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            max_concurrent_outbound_streams_per_peer: 3,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();

        plane
            .prepare_serving_peer_channels(
                &[plane.local_addr()],
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cuda,
            )
            .await
            .unwrap();

        wait_for_metric(&plane, |snapshot| {
            snapshot.current_outbound_connections >= 10
        })
        .await;

        let capabilities = plane.capabilities_snapshot();
        assert!(capabilities.persistent_serving_peer_channels);
        assert!(capabilities.provider_specialized_collectives);
        assert!(capabilities.reserved_priority_outbound_inflight_bytes > 0);
        assert!(capabilities.bulk_send_bandwidth_bytes_per_sec > 0);
        assert!(capabilities.per_peer_bulk_fairness);
    }

    #[tokio::test]
    async fn test_serving_transport_stream_ids_follow_lane_plan() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            max_concurrent_outbound_streams_per_peer: 3,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();
        let session = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cuda,
            )
            .await
            .unwrap();

        assert_eq!(session.stream_id_for(CollectiveLane::Control, 7, 9), 0);
        assert_eq!(session.stream_id_for(CollectiveLane::AllGather, 1, 0), 1);
        assert_eq!(session.stream_id_for(CollectiveLane::BulkTransfer, 1, 0), 1);
        assert_eq!(session.stream_id_for(CollectiveLane::Checkpoint, 2, 1), 1);
    }

    #[tokio::test]
    async fn test_serving_transport_reuses_durable_neighbor_pair_session() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let first = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();
        let second = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::LatencyFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();

        assert_eq!(first.session_id(), second.session_id());
    }

    #[tokio::test]
    async fn test_checkpoint_and_bulk_lanes_use_serving_transport() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let session = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();
        let checkpoint_collective = Uuid::new_v4();
        let bulk_collective = Uuid::new_v4();

        session
            .send_bulk_transfer(bulk_collective, 4, 0, 0, 0, 2, &[1.0, 2.0, 3.0], &[3])
            .await
            .unwrap();
        session
            .send_checkpoint(checkpoint_collective, 4, 1, 0, 0, 2, &[9.0], &[1])
            .await
            .unwrap();

        let bulk = session
            .recv_frame(ServingReceiveSpec {
                collective_id: bulk_collective,
                lane: CollectiveLane::BulkTransfer,
                layer_idx: 4,
                step: 0,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 2,
            })
            .await
            .unwrap();
        let checkpoint = session
            .recv_frame(ServingReceiveSpec {
                collective_id: checkpoint_collective,
                lane: CollectiveLane::Checkpoint,
                layer_idx: 4,
                step: 1,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 2,
            })
            .await
            .unwrap();

        assert_eq!(bulk.chunk_data, vec![1.0, 2.0, 3.0]);
        assert_eq!(checkpoint.chunk_data, vec![9.0]);
        let snapshot = plane.metrics_snapshot();
        assert!(snapshot.bulk_transfer_bytes_sent > 0);
        assert!(snapshot.bulk_transfer_bytes_received > 0);
        assert!(snapshot.checkpoint_bytes_sent > 0);
        assert!(snapshot.checkpoint_bytes_received > 0);
    }

    #[tokio::test]
    async fn test_control_lane_send_uses_latency_critical_accounting() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let session = plane
            .serving_transport_for_neighbors(
                plane.local_addr(),
                plane.local_addr(),
                InferenceRuntimeMode::LatencyFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();
        let collective_id = Uuid::new_v4();

        session
            .send_control(collective_id, 7, 0, 0, 0, 3, &[3.0], &[1])
            .await
            .unwrap();
        let _ = session
            .recv_frame(ServingReceiveSpec {
                collective_id,
                lane: CollectiveLane::Control,
                layer_idx: 7,
                step: 0,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 3,
            })
            .await
            .unwrap();

        let snapshot = plane.metrics_snapshot();
        assert_eq!(snapshot.latency_critical_send_count, 1);
        assert_eq!(snapshot.barrier_bytes_sent > 0, true);
        assert_eq!(snapshot.barrier_bytes_received > 0, true);
    }
}
