use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::io::IoSlice;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::slice;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
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
static NEXT_TENSOR_PLANE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

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
    pub connection_refresh_attempt_count: u64,
    pub connection_refresh_success_count: u64,
    pub connection_evict_count: u64,
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
    pub runtime_mode_aware_traffic_policies: bool,
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
    connection_refresh_attempt_count: AtomicU64,
    connection_refresh_success_count: AtomicU64,
    connection_evict_count: AtomicU64,
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
            connection_refresh_attempt_count: AtomicU64::new(0),
            connection_refresh_success_count: AtomicU64::new(0),
            connection_evict_count: AtomicU64::new(0),
            latency_critical_send_count: AtomicU64::new(0),
            interactive_send_count: AtomicU64::new(0),
            bulk_send_count: AtomicU64::new(0),
        }
    }
}

#[derive(Debug)]
struct InboundServingFrame {
    frame: ServingFrameBytes,
    queued_at: Instant,
    _queued_bytes_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug)]
pub struct ServingFrameBytes {
    header: ServingFrameHeader,
    payload_bytes: Vec<u8>,
}

impl ServingFrameBytes {
    pub fn header(&self) -> ServingFrameHeader {
        self.header
    }

    pub fn payload_bytes(&self) -> &[u8] {
        &self.payload_bytes
    }

    pub fn element_count(&self) -> usize {
        self.header.element_count as usize
    }

    pub fn decode_payload_vec(&self) -> Vec<f32> {
        decode_f32_slice_wire(&self.payload_bytes)
    }

    pub fn first_f32(&self) -> Result<f32> {
        if self.payload_bytes.len() < std::mem::size_of::<f32>() {
            return Err(AgentError::Network(
                "Serving frame payload missing first f32 value".to_string(),
            ));
        }
        Ok(f32::from_bits(u32::from_le_bytes([
            self.payload_bytes[0],
            self.payload_bytes[1],
            self.payload_bytes[2],
            self.payload_bytes[3],
        ])))
    }

    pub fn accumulate_into(&self, dst: &mut [f32]) -> Result<()> {
        if dst.len() != self.element_count() {
            return Err(AgentError::Network(format!(
                "Serving frame payload len {} did not match destination len {}",
                self.element_count(),
                dst.len()
            )));
        }
        for (slot, chunk) in dst
            .iter_mut()
            .zip(self.payload_bytes.chunks_exact(std::mem::size_of::<f32>()))
        {
            *slot += f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(())
    }

    pub fn copy_into(&self, dst: &mut [f32]) -> Result<()> {
        if dst.len() != self.element_count() {
            return Err(AgentError::Network(format!(
                "Serving frame payload len {} did not match destination len {}",
                self.element_count(),
                dst.len()
            )));
        }
        copy_wire_f32_bytes_into_slice(dst, &self.payload_bytes);
        Ok(())
    }
}

#[derive(Debug)]
struct ServingInboundState {
    slots: Mutex<HashMap<ServingSlotKey, VecDeque<InboundServingFrame>>>,
    notify: Notify,
    queued_messages: AtomicUsize,
}

impl Default for ServingInboundState {
    fn default() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            notify: Notify::new(),
            queued_messages: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct TensorPlaneState {
    instance_id: u64,
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
    inbound: Arc<ServingInboundState>,
    metrics: TensorPlaneMetrics,
    outbound_connections: Mutex<HashMap<SocketAddr, OutboundPeerChannels>>,
    peer_bulk_outbound_bytes: Mutex<HashMap<SocketAddr, Arc<Semaphore>>>,
}

#[derive(Debug)]
struct OutboundPeerChannels {
    lanes: HashMap<CollectiveLane, LanePeerChannels>,
}

#[derive(Debug)]
struct LanePeerChannels {
    streams: Vec<Arc<Mutex<TcpStream>>>,
    pinned_for_serving: bool,
}

#[derive(Debug, Clone, Copy)]
struct ServingLanePlan {
    lane: CollectiveLane,
    traffic_class: TensorTrafficClass,
    desired_stream_count: usize,
}

#[derive(Debug, Clone)]
struct BoundServingLane {
    target: SocketAddr,
    plan: ServingLanePlan,
    streams: Arc<Mutex<Vec<Arc<Mutex<TcpStream>>>>>,
}

#[derive(Debug, Clone, Copy)]
pub struct ServingReceiveSpec {
    pub collective_id: Uuid,
    pub collective_seq: u32,
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
    left_peer: SocketAddr,
    right_peer: SocketAddr,
    reduce_scatter_lane: BoundServingLane,
    all_gather_lane: BoundServingLane,
    control_lane: BoundServingLane,
    bulk_transfer_lane: BoundServingLane,
    checkpoint_lane: BoundServingLane,
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

    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
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
            .field("left_peer", &self.left_peer)
            .field("right_peer", &self.right_peer)
            .finish()
    }
}

impl ServingSessionTransport {
    pub fn instance_id(&self) -> u64 {
        self.state.instance_id
    }

    pub fn state_ptr(&self) -> usize {
        Arc::as_ptr(&self.state) as usize
    }

    pub fn inbound_id(&self) -> usize {
        Arc::as_ptr(&self.state.inbound) as usize
    }

    fn lane_binding(&self, lane: CollectiveLane) -> &BoundServingLane {
        match lane {
            CollectiveLane::ReduceScatter => &self.reduce_scatter_lane,
            CollectiveLane::AllGather => &self.all_gather_lane,
            CollectiveLane::Control => &self.control_lane,
            CollectiveLane::BulkTransfer => &self.bulk_transfer_lane,
            CollectiveLane::Checkpoint => &self.checkpoint_lane,
        }
    }

    pub fn stream_id_for(&self, lane: CollectiveLane, step: u32, slot: u32) -> u32 {
        let plan = self.lane_binding(lane).plan;
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
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
    ) -> Result<()> {
        self.send_frame(
            ServingFrameHeader::new(
                collective_id,
                collective_seq,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::ReduceScatter,
                chunk_data.len() as u32,
            ),
            chunk_data,
            &self.reduce_scatter_lane,
        )
        .await
    }

    pub async fn send_all_gather_chunk(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
    ) -> Result<()> {
        self.send_frame(
            ServingFrameHeader::new(
                collective_id,
                collective_seq,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::AllGather,
                chunk_data.len() as u32,
            ),
            chunk_data,
            &self.all_gather_lane,
        )
        .await
    }

    pub async fn send_control(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
    ) -> Result<()> {
        self.send_frame(
            ServingFrameHeader::new(
                collective_id,
                collective_seq,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::Control,
                chunk_data.len() as u32,
            ),
            chunk_data,
            &self.control_lane,
        )
        .await
    }

    pub async fn send_bulk_transfer(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
    ) -> Result<()> {
        self.send_frame(
            ServingFrameHeader::new(
                collective_id,
                collective_seq,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::BulkTransfer,
                chunk_data.len() as u32,
            ),
            chunk_data,
            &self.bulk_transfer_lane,
        )
        .await
    }

    pub async fn send_checkpoint(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: &[f32],
    ) -> Result<()> {
        self.send_frame(
            ServingFrameHeader::new(
                collective_id,
                collective_seq,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::Checkpoint,
                chunk_data.len() as u32,
            ),
            chunk_data,
            &self.checkpoint_lane,
        )
        .await
    }

    pub async fn send_checkpoint_bytes(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        payload_bytes: &[u8],
    ) -> Result<()> {
        let payload_bytes = wire_bytes_for_raw_payload(payload_bytes);
        send_serving_frame_bytes_on_bound_lane(
            &self.state,
            &self.checkpoint_lane,
            ServingFrameHeader::new(
                collective_id,
                collective_seq,
                sender_position,
                layer_idx,
                step,
                slot,
                stream_id,
                CollectiveLane::Checkpoint,
                (payload_bytes.len() / std::mem::size_of::<f32>()) as u32,
            ),
            &payload_bytes,
        )
        .await
    }

    pub fn spawn_bulk_transfer(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: Vec<f32>,
    ) -> ServingBackgroundTransfer {
        self.spawn_background_frame(
            collective_id,
            collective_seq,
            layer_idx,
            step,
            slot,
            stream_id,
            sender_position,
            chunk_data,
            self.bulk_transfer_lane.clone(),
        )
    }

    pub fn spawn_checkpoint(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: Vec<f32>,
    ) -> ServingBackgroundTransfer {
        self.spawn_background_frame(
            collective_id,
            collective_seq,
            layer_idx,
            step,
            slot,
            stream_id,
            sender_position,
            chunk_data,
            self.checkpoint_lane.clone(),
        )
    }

    pub async fn recv_frame(&self, spec: ServingReceiveSpec) -> Result<ServingFrame> {
        let frame = self.recv_frame_bytes(spec).await?;
        Ok(ServingFrame {
            header: frame.header(),
            chunk_data: frame.decode_payload_vec(),
        })
    }

    pub async fn recv_frame_bytes(&self, spec: ServingReceiveSpec) -> Result<ServingFrameBytes> {
        recv_slot(&self.state, &self.state.inbound, spec).await
    }

    async fn send_frame(
        &self,
        header: ServingFrameHeader,
        chunk_data: &[f32],
        lane: &BoundServingLane,
    ) -> Result<()> {
        let payload_bytes = wire_bytes_for_f32_slice(chunk_data);
        send_serving_frame_bytes_on_bound_lane(&self.state, lane, header, &payload_bytes).await
    }

    fn spawn_background_frame(
        &self,
        collective_id: Uuid,
        collective_seq: u32,
        layer_idx: u32,
        step: u32,
        slot: u32,
        stream_id: u32,
        sender_position: u32,
        chunk_data: Vec<f32>,
        lane: BoundServingLane,
    ) -> ServingBackgroundTransfer {
        let state = Arc::clone(&self.state);
        let transfer_lane = lane.plan.lane;
        let lane_binding = lane.clone();
        let join_handle = tokio::spawn(async move {
            let payload_bytes = wire_bytes_for_f32_slice(&chunk_data);
            send_serving_frame_bytes_on_bound_lane(
                &state,
                &lane_binding,
                ServingFrameHeader::new(
                    collective_id,
                    collective_seq,
                    sender_position,
                    layer_idx,
                    step,
                    slot,
                    stream_id,
                    transfer_lane,
                    chunk_data.len() as u32,
                ),
                &payload_bytes,
            )
            .await
        });
        ServingBackgroundTransfer {
            lane: transfer_lane,
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
    _accept_task: tokio::task::JoinHandle<()>,
}

async fn bind_serving_lane(
    state: &Arc<TensorPlaneState>,
    target: SocketAddr,
    plan: ServingLanePlan,
) -> Result<BoundServingLane> {
    let mut streams = Vec::with_capacity(plan.desired_stream_count.max(1));
    for stream_id in 0..plan.desired_stream_count.max(1) as u32 {
        streams.push(
            ensure_connection_pool(
                state,
                target,
                plan.lane,
                plan.desired_stream_count,
                stream_id,
                true,
            )
            .await?,
        );
    }
    Ok(BoundServingLane {
        target,
        plan,
        streams: Arc::new(Mutex::new(streams)),
    })
}

impl TensorPlane {
    pub fn instance_id(&self) -> u64 {
        self.state.instance_id
    }

    pub fn state_ptr(&self) -> usize {
        Arc::as_ptr(&self.state) as usize
    }

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
        let inbound = Arc::new(ServingInboundState::default());
        let instance_id = NEXT_TENSOR_PLANE_INSTANCE_ID.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(TensorPlaneState {
            instance_id,
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
            inbound: Arc::clone(&inbound),
            metrics: TensorPlaneMetrics::new(),
            outbound_connections: Mutex::new(HashMap::new()),
            peer_bulk_outbound_bytes: Mutex::new(HashMap::new()),
        });
        let accept_state = Arc::clone(&state);
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((mut stream, remote_addr)) => {
                        let state = Arc::clone(&accept_state);
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

                                        if !try_reserve_inbound_message_slot(
                                            &state.inbound,
                                            state.inbound_queue_message_capacity,
                                        ) {
                                            state
                                                .metrics
                                                .inbound_queue_full_rejections
                                                .fetch_add(1, Ordering::Relaxed);
                                            continue;
                                        }
                                        let slot = frame.header.slot_key();
                                        let header = frame.header;
                                        let queue_len = {
                                            let mut slots = state.inbound.slots.lock().await;
                                            let queue =
                                                slots.entry(slot).or_insert_with(VecDeque::new);
                                            queue.push_back(InboundServingFrame {
                                                frame,
                                                queued_at: Instant::now(),
                                                _queued_bytes_permit: Some(permit),
                                            });
                                            queue.len()
                                        };
                                        if matches!(
                                            header.lane,
                                            CollectiveLane::ReduceScatter
                                                | CollectiveLane::AllGather
                                        ) {
                                            info!(
                                                tensor_plane_instance = state.instance_id,
                                                tensor_plane_state = format_args!("{:p}", Arc::as_ptr(&state)),
                                                inbound_id = format_args!("{:p}", Arc::as_ptr(&state.inbound)),
                                                remote_addr = %remote_addr,
                                                collective_id = %header.collective_id,
                                                collective_seq = header.collective_seq,
                                                sender_position = header.sender_position,
                                                lane = ?header.lane,
                                                layer_idx = header.layer_idx,
                                                step = header.step,
                                                slot = header.slot,
                                                stream_id = header.stream_id,
                                                queued_elements = header.element_count,
                                                queue_len,
                                                "Enqueued serving collective frame"
                                            );
                                        } else {
                                            debug!(
                                                tensor_plane_instance = state.instance_id,
                                                tensor_plane_state = format_args!("{:p}", Arc::as_ptr(&state)),
                                                inbound_id = format_args!("{:p}", Arc::as_ptr(&state.inbound)),
                                                remote_addr = %remote_addr,
                                                collective_id = %header.collective_id,
                                                collective_seq = header.collective_seq,
                                                sender_position = header.sender_position,
                                                lane = ?header.lane,
                                                layer_idx = header.layer_idx,
                                                step = header.step,
                                                slot = header.slot,
                                                stream_id = header.stream_id,
                                                queued_elements = header.element_count,
                                                queue_len,
                                                "Enqueued serving dataplane frame"
                                            );
                                        }

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
                                        state.inbound.notify.notify_waiters();
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
                                        debug!(
                                            remote_addr = %remote_addr,
                                            "serving dataplane connection was idle past the receive timeout; keeping channel open"
                                        );
                                        continue;
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
            runtime_mode_aware_traffic_policies: true,
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
            connection_refresh_attempt_count: self
                .state
                .metrics
                .connection_refresh_attempt_count
                .load(Ordering::Relaxed),
            connection_refresh_success_count: self
                .state
                .metrics
                .connection_refresh_success_count
                .load(Ordering::Relaxed),
            connection_evict_count: self
                .state
                .metrics
                .connection_evict_count
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
            self.state.profile,
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        let mut unique_peers = Vec::with_capacity(peers.len());
        for &peer in peers {
            if unique_peers.contains(&peer) {
                continue;
            }
            unique_peers.push(peer);
        }
        for peer in unique_peers {
            for plan in hot_lane_plans {
                for stream_id in 0..plan.desired_stream_count as u32 {
                    self.ensure_connection_pool(
                        peer,
                        plan.lane,
                        plan.desired_stream_count,
                        stream_id,
                        true,
                    )
                    .await?;
                }
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
        reset_serving_connection_pool(&self.state, right_peer).await;
        let mut reduce_scatter_plan = lane_plan(
            CollectiveLane::ReduceScatter,
            self.state.profile,
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        let mut all_gather_plan = lane_plan(
            CollectiveLane::AllGather,
            self.state.profile,
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        let mut control_plan = lane_plan(
            CollectiveLane::Control,
            self.state.profile,
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        let mut bulk_transfer_plan = lane_plan(
            CollectiveLane::BulkTransfer,
            self.state.profile,
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        let mut checkpoint_plan = lane_plan(
            CollectiveLane::Checkpoint,
            self.state.profile,
            runtime_mode,
            provider,
            self.state.max_concurrent_outbound_streams_per_peer,
        );
        if left_peer == right_peer {
            // A two-worker ring collapses left/right onto the same peer. For that
            // degenerate topology, multiplexing a single collective phase over
            // multiple logical streams adds ambiguity without adding any real
            // parallel transport path.
            reduce_scatter_plan.desired_stream_count = 1;
            all_gather_plan.desired_stream_count = 1;
            control_plan.desired_stream_count = 1;
            bulk_transfer_plan.desired_stream_count = 1;
            checkpoint_plan.desired_stream_count = 1;
        }
        info!(
            tensor_plane_instance = self.state.instance_id,
            tensor_plane_state = format_args!("{:p}", Arc::as_ptr(&self.state)),
            left_peer = %left_peer,
            right_peer = %right_peer,
            runtime_mode = ?runtime_mode,
            provider = ?provider,
            inbound_id = format_args!("{:p}", Arc::as_ptr(&self.state.inbound)),
            reduce_scatter_streams = reduce_scatter_plan.desired_stream_count,
            all_gather_streams = all_gather_plan.desired_stream_count,
            control_streams = control_plan.desired_stream_count,
            bulk_streams = bulk_transfer_plan.desired_stream_count,
            checkpoint_streams = checkpoint_plan.desired_stream_count,
            "Binding serving transport for tensor-plane neighbors"
        );
        Ok(ServingSessionTransport {
            state: Arc::clone(&self.state),
            left_peer,
            right_peer,
            reduce_scatter_lane: bind_serving_lane(&self.state, right_peer, reduce_scatter_plan)
                .await?,
            all_gather_lane: bind_serving_lane(&self.state, right_peer, all_gather_plan).await?,
            control_lane: bind_serving_lane(&self.state, right_peer, control_plan).await?,
            bulk_transfer_lane: bind_serving_lane(&self.state, right_peer, bulk_transfer_plan)
                .await?,
            checkpoint_lane: bind_serving_lane(&self.state, right_peer, checkpoint_plan).await?,
        })
    }

    pub async fn recv_frame_bytes(&self, spec: ServingReceiveSpec) -> Result<ServingFrameBytes> {
        recv_slot(&self.state, &self.state.inbound, spec).await
    }

    pub async fn recv_frame(&self, spec: ServingReceiveSpec) -> Result<ServingFrame> {
        let frame = self.recv_frame_bytes(spec).await?;
        Ok(ServingFrame {
            header: frame.header(),
            chunk_data: frame.decode_payload_vec(),
        })
    }

    async fn ensure_connection_pool(
        &self,
        target: SocketAddr,
        lane: CollectiveLane,
        desired_stream_count: usize,
        selected_stream_id: u32,
        mark_persistent: bool,
    ) -> Result<Arc<Mutex<TcpStream>>> {
        ensure_connection_pool(
            &self.state,
            target,
            lane,
            desired_stream_count,
            selected_stream_id,
            mark_persistent,
        )
        .await
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

async fn send_serving_frame_bytes_on_bound_lane(
    state: &Arc<TensorPlaneState>,
    lane: &BoundServingLane,
    header: ServingFrameHeader,
    payload_bytes: &[u8],
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
        acquire_lane_budget(state, lane.target, lane.plan.traffic_class, message_bytes).await?;
    let selected_stream_index = if lane.plan.desired_stream_count <= 1 {
        0
    } else {
        header.stream_id as usize % lane.plan.desired_stream_count
    };
    let send_result = send_serving_frame_bytes_with_refresh(
        state,
        lane,
        selected_stream_index,
        header,
        payload_bytes,
    )
    .await;

    match send_result {
        Ok(()) => {
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
            record_traffic_class_send(&state.metrics, lane.plan.traffic_class);
            drop(class_permit);
            drop(outbound_permit);
            Ok(())
        }
        Err(error) => {
            drop(class_permit);
            drop(outbound_permit);
            Err(error)
        }
    }
}

async fn send_serving_frame_bytes_with_refresh(
    state: &Arc<TensorPlaneState>,
    lane: &BoundServingLane,
    stream_index: usize,
    header: ServingFrameHeader,
    payload_bytes: &[u8],
) -> Result<()> {
    let initial_stream = {
        let streams = lane.streams.lock().await;
        Arc::clone(&streams[stream_index])
    };
    let first_result = {
        let mut stream_guard = initial_stream.lock().await;
        tokio::time::timeout(
            state.io_timeout,
            write_serving_frame_bytes(&mut *stream_guard, header, payload_bytes),
        )
        .await
    };
    match first_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) | Err(_) => {
            state
                .metrics
                .connection_refresh_attempt_count
                .fetch_add(1, Ordering::Relaxed);
            evict_connection(state, lane.target, lane.plan.lane).await;
            let refreshed_stream = ensure_connection_pool(
                state,
                lane.target,
                lane.plan.lane,
                lane.plan.desired_stream_count,
                header.stream_id,
                true,
            )
            .await?;
            {
                let mut streams = lane.streams.lock().await;
                streams[stream_index] = Arc::clone(&refreshed_stream);
            }
            let mut stream_guard = refreshed_stream.lock().await;
            match tokio::time::timeout(
                state.io_timeout,
                write_serving_frame_bytes(&mut *stream_guard, header, payload_bytes),
            )
            .await
            {
                Ok(Ok(())) => {
                    state
                        .metrics
                        .connection_refresh_success_count
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
                Ok(Err(error)) => Err(AgentError::Network(format!(
                    "Failed to send serving frame to {}: {}",
                    lane.target, error
                ))),
                Err(_) => {
                    state
                        .metrics
                        .send_timeout_count
                        .fetch_add(1, Ordering::Relaxed);
                    Err(AgentError::Network(format!(
                        "Timed out sending serving frame to {}",
                        lane.target
                    )))
                }
            }
        }
    }
}

async fn recv_slot(
    state: &Arc<TensorPlaneState>,
    inbound: &Arc<ServingInboundState>,
    spec: ServingReceiveSpec,
) -> Result<ServingFrameBytes> {
    let slot_key = ServingSlotKey {
        collective_id: spec.collective_id,
        collective_seq: spec.collective_seq,
        sender_position: spec.expected_sender_position,
        lane: spec.lane,
        layer_idx: spec.layer_idx,
        step: spec.step,
        slot: spec.slot,
        stream_id: spec.stream_id,
    };

    loop {
        let notified = inbound.notify.notified();
        tokio::pin!(notified);
        let inbound_id = format!("{:p}", Arc::as_ptr(inbound));
        let (delivered, queued_slot_keys) = {
            let mut slots = inbound.slots.lock().await;
            let mut should_remove = false;
            let delivered = slots.get_mut(&slot_key).and_then(|queue| {
                let message = queue.pop_front()?;
                should_remove = queue.is_empty();
                Some(message)
            });
            if should_remove {
                slots.remove(&slot_key);
            }
            let queued_slot_keys = if delivered.is_some() {
                None
            } else {
                Some(
                    slots
                        .iter()
                        .map(|(key, queue)| {
                            format!(
                                "{}:{}:{}:{:?}:{}:{}:{}:{}(len={})",
                                key.collective_id,
                                key.collective_seq,
                                key.sender_position,
                                key.lane,
                                key.layer_idx,
                                key.step,
                                key.slot,
                                key.stream_id,
                                queue.len()
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            };
            (delivered, queued_slot_keys)
        };
        if let Some(message) = delivered {
            if matches!(
                slot_key.lane,
                CollectiveLane::ReduceScatter | CollectiveLane::AllGather
            ) {
                info!(
                    tensor_plane_instance = state.instance_id,
                    tensor_plane_state = format_args!("{:p}", Arc::as_ptr(state)),
                    inbound_id = %inbound_id,
                    collective_id = %slot_key.collective_id,
                    collective_seq = slot_key.collective_seq,
                    sender_position = slot_key.sender_position,
                    lane = ?slot_key.lane,
                    layer_idx = slot_key.layer_idx,
                    step = slot_key.step,
                    slot = slot_key.slot,
                    stream_id = slot_key.stream_id,
                    queued_elements = message.frame.element_count(),
                    "Delivered serving collective frame from mailbox"
                );
            } else {
                debug!(
                    tensor_plane_instance = state.instance_id,
                    tensor_plane_state = format_args!("{:p}", Arc::as_ptr(state)),
                    inbound_id = %inbound_id,
                    collective_id = %slot_key.collective_id,
                    collective_seq = slot_key.collective_seq,
                    sender_position = slot_key.sender_position,
                    lane = ?slot_key.lane,
                    layer_idx = slot_key.layer_idx,
                    step = slot_key.step,
                    slot = slot_key.slot,
                    stream_id = slot_key.stream_id,
                    queued_elements = message.frame.element_count(),
                    "Delivered serving dataplane frame from mailbox"
                );
            }
            inbound.queued_messages.fetch_sub(1, Ordering::Relaxed);
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
        if matches!(
            slot_key.lane,
            CollectiveLane::ReduceScatter | CollectiveLane::AllGather
        ) {
            info!(
                tensor_plane_instance = state.instance_id,
                tensor_plane_state = format_args!("{:p}", Arc::as_ptr(state)),
                inbound_id = %inbound_id,
                collective_id = %slot_key.collective_id,
                collective_seq = slot_key.collective_seq,
                sender_position = slot_key.sender_position,
                lane = ?slot_key.lane,
                layer_idx = slot_key.layer_idx,
                step = slot_key.step,
                slot = slot_key.slot,
                stream_id = slot_key.stream_id,
                queued_slot_keys = ?queued_slot_keys,
                "Waiting for serving collective frame"
            );
        } else {
            debug!(
                tensor_plane_instance = state.instance_id,
                tensor_plane_state = format_args!("{:p}", Arc::as_ptr(state)),
                inbound_id = %inbound_id,
                collective_id = %slot_key.collective_id,
                collective_seq = slot_key.collective_seq,
                sender_position = slot_key.sender_position,
                lane = ?slot_key.lane,
                layer_idx = slot_key.layer_idx,
                step = slot_key.step,
                slot = slot_key.slot,
                stream_id = slot_key.stream_id,
                queued_slot_keys = ?queued_slot_keys,
                "Waiting for serving dataplane frame"
            );
        }
        notified.await;
    }
}

fn try_reserve_inbound_message_slot(inbound: &ServingInboundState, capacity: usize) -> bool {
    let mut current = inbound.queued_messages.load(Ordering::Relaxed);
    loop {
        if current >= capacity {
            return false;
        }
        match inbound.queued_messages.compare_exchange(
            current,
            current + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return true,
            Err(observed) => current = observed,
        }
    }
}

async fn ensure_connection_pool(
    state: &Arc<TensorPlaneState>,
    target: SocketAddr,
    lane: CollectiveLane,
    desired_stream_count: usize,
    selected_stream_id: u32,
    mark_persistent: bool,
) -> Result<Arc<Mutex<TcpStream>>> {
    let desired_stream_count = desired_stream_count
        .max(1)
        .min(state.max_concurrent_outbound_streams_per_peer);
    let selected_stream_index = if desired_stream_count == 1 {
        0
    } else {
        selected_stream_id as usize % desired_stream_count
    };

    loop {
        {
            let mut connections = state.outbound_connections.lock().await;
            if let Some(pool) = connections.get_mut(&target) {
                let lane_pool = pool.lanes.entry(lane).or_insert_with(|| LanePeerChannels {
                    streams: Vec::new(),
                    pinned_for_serving: false,
                });
                if mark_persistent {
                    lane_pool.pinned_for_serving = true;
                }
                if lane_pool.streams.len() > selected_stream_index {
                    return Ok(Arc::clone(&lane_pool.streams[selected_stream_index]));
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
            state
                .metrics
                .connection_evict_count
                .fetch_add(1, Ordering::Relaxed);
        }
        if pool.lanes.is_empty() {
            connections.remove(&target);
        }
    }
}

async fn reset_serving_connection_pool(state: &Arc<TensorPlaneState>, target: SocketAddr) {
    for lane in [
        CollectiveLane::ReduceScatter,
        CollectiveLane::AllGather,
        CollectiveLane::Control,
        CollectiveLane::BulkTransfer,
        CollectiveLane::Checkpoint,
    ] {
        evict_connection(state, target, lane).await;
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
    profile: TensorPlaneProfile,
    runtime_mode: InferenceRuntimeMode,
    provider: ExecutionProviderKind,
    max_streams: usize,
) -> usize {
    let max_streams = max_streams.max(1);
    match profile {
        TensorPlaneProfile::Conservative => match (provider, runtime_mode) {
            (ExecutionProviderKind::Cuda, InferenceRuntimeMode::ThroughputFirst) => max_streams,
            (ExecutionProviderKind::Metal, InferenceRuntimeMode::ThroughputFirst) => {
                2.min(max_streams)
            }
            (ExecutionProviderKind::Metal, InferenceRuntimeMode::LatencyFirst) => {
                2.min(max_streams)
            }
            (ExecutionProviderKind::Cpu, _) | (_, InferenceRuntimeMode::FitFirst) => 1,
            _ => 1.max(max_streams / 2),
        },
        TensorPlaneProfile::Lan => match (provider, runtime_mode) {
            (ExecutionProviderKind::Cuda, InferenceRuntimeMode::ThroughputFirst) => max_streams,
            (ExecutionProviderKind::Cuda, InferenceRuntimeMode::LatencyFirst) => {
                max_streams.min(3).max(2)
            }
            (ExecutionProviderKind::Metal, InferenceRuntimeMode::ThroughputFirst) => {
                max_streams.min(3).max(2)
            }
            (ExecutionProviderKind::Metal, InferenceRuntimeMode::LatencyFirst) => {
                2.min(max_streams)
            }
            (ExecutionProviderKind::Cpu, InferenceRuntimeMode::ThroughputFirst) => {
                2.min(max_streams)
            }
            (ExecutionProviderKind::Cpu, _) | (_, InferenceRuntimeMode::FitFirst) => 1,
            _ => 1.max(max_streams / 2),
        },
    }
}

fn lane_plan(
    lane: CollectiveLane,
    profile: TensorPlaneProfile,
    runtime_mode: InferenceRuntimeMode,
    provider: ExecutionProviderKind,
    max_streams: usize,
) -> ServingLanePlan {
    let bulk_streams = preferred_serving_stream_count(profile, runtime_mode, provider, max_streams);
    let interactive_streams = match profile {
        TensorPlaneProfile::Conservative => bulk_streams.min(2).max(1),
        TensorPlaneProfile::Lan => bulk_streams.min(3).max(1),
    };
    let control_streams = 1;
    let checkpoint_streams = match (profile, provider, runtime_mode) {
        (
            TensorPlaneProfile::Lan,
            ExecutionProviderKind::Cuda,
            InferenceRuntimeMode::ThroughputFirst,
        ) => bulk_streams.min(3).max(1),
        (
            TensorPlaneProfile::Lan,
            ExecutionProviderKind::Metal,
            InferenceRuntimeMode::ThroughputFirst,
        ) => interactive_streams,
        (_, ExecutionProviderKind::Cuda, InferenceRuntimeMode::ThroughputFirst) => {
            interactive_streams
        }
        (_, ExecutionProviderKind::Metal, InferenceRuntimeMode::ThroughputFirst) => {
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
    profile: TensorPlaneProfile,
    runtime_mode: InferenceRuntimeMode,
    provider: ExecutionProviderKind,
    max_streams: usize,
) -> [ServingLanePlan; 5] {
    [
        lane_plan(
            CollectiveLane::ReduceScatter,
            profile,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(
            CollectiveLane::AllGather,
            profile,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(
            CollectiveLane::Control,
            profile,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(
            CollectiveLane::BulkTransfer,
            profile,
            runtime_mode,
            provider,
            max_streams,
        ),
        lane_plan(
            CollectiveLane::Checkpoint,
            profile,
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

#[cfg(test)]
async fn write_serving_frame<W>(
    writer: &mut W,
    header: ServingFrameHeader,
    chunk_data: &[f32],
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let payload_bytes = wire_bytes_for_f32_slice(chunk_data);
    write_serving_frame_bytes(writer, header, &payload_bytes).await
}

async fn write_serving_frame_bytes<W>(
    writer: &mut W,
    header: ServingFrameHeader,
    payload_bytes: &[u8],
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let header_bytes = header.encode_binary();
    let mut header_offset = 0usize;
    let mut payload_offset = 0usize;
    while header_offset < header_bytes.len() || payload_offset < payload_bytes.len() {
        let mut bufs = [IoSlice::new(&[]), IoSlice::new(&[])];
        let mut count = 0usize;
        if header_offset < header_bytes.len() {
            bufs[count] = IoSlice::new(&header_bytes[header_offset..]);
            count += 1;
        }
        if payload_offset < payload_bytes.len() {
            bufs[count] = IoSlice::new(&payload_bytes[payload_offset..]);
            count += 1;
        }
        let written = writer.write_vectored(&bufs[..count]).await?;
        if written == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "failed to write serving frame bytes",
            ));
        }
        let remaining_header = header_bytes.len().saturating_sub(header_offset);
        let consumed_header = written.min(remaining_header);
        header_offset += consumed_header;
        payload_offset += written.saturating_sub(consumed_header);
    }
    Ok(())
}

async fn read_serving_frame<R>(
    reader: &mut R,
    max_message_bytes: usize,
) -> std::io::Result<ServingFrameBytes>
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
    let payload_bytes = header.element_count as usize * std::mem::size_of::<f32>();
    let mut payload_buf = vec![0u8; payload_bytes];
    reader.read_exact(&mut payload_buf).await?;
    Ok(ServingFrameBytes {
        header,
        payload_bytes: payload_buf,
    })
}

fn wire_bytes_for_f32_slice(chunk_data: &[f32]) -> Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    unsafe {
        Cow::Borrowed(slice::from_raw_parts(
            chunk_data.as_ptr() as *const u8,
            std::mem::size_of_val(chunk_data),
        ))
    }
    #[cfg(target_endian = "big")]
    {
        let mut payload = Vec::with_capacity(std::mem::size_of_val(chunk_data));
        for value in chunk_data {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Cow::Owned(payload)
    }
}

fn wire_bytes_for_raw_payload(payload_bytes: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(4 + payload_bytes.len() + 3);
    framed.extend_from_slice(&(payload_bytes.len() as u32).to_le_bytes());
    framed.extend_from_slice(payload_bytes);
    let padding = (std::mem::size_of::<f32>() - (framed.len() % std::mem::size_of::<f32>()))
        % std::mem::size_of::<f32>();
    if padding > 0 {
        framed.resize(framed.len() + padding, 0);
    }
    framed
}

fn decode_f32_slice_wire(buf: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; buf.len() / std::mem::size_of::<f32>()];
    copy_wire_f32_bytes_into_slice(&mut out, buf);
    out
}

fn copy_wire_f32_bytes_into_slice(dst: &mut [f32], src: &[u8]) {
    let expected_bytes = dst.len().saturating_mul(std::mem::size_of::<f32>());
    assert_eq!(
        src.len(),
        expected_bytes,
        "wire payload byte length {} did not match destination byte length {}",
        src.len(),
        expected_bytes
    );
    #[cfg(target_endian = "little")]
    unsafe {
        let dst_bytes = slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, expected_bytes);
        dst_bytes.copy_from_slice(src);
    }
    #[cfg(target_endian = "big")]
    for (slot, chunk) in dst
        .iter_mut()
        .zip(src.chunks_exact(std::mem::size_of::<f32>()))
    {
        *slot = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
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
            .send_reduce_scatter_chunk(Uuid::new_v4(), 0, 1, 0, 0, 0, 9, &[1.0, 2.0])
            .await
            .unwrap();
        let collective_b = Uuid::new_v4();
        session_b
            .send_reduce_scatter_chunk(collective_b, 0, 1, 0, 0, 0, 7, &[3.0, 4.0])
            .await
            .unwrap();

        let frame_b = timeout(
            Duration::from_secs(1),
            session_b.recv_frame(ServingReceiveSpec {
                collective_id: collective_b,
                collective_seq: 0,
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
                        collective_seq: 0,
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
                        collective_seq: 0,
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
            .send_reduce_scatter_chunk(collective_id, 0, 3, 1, 2, 0, 12, &[6.0])
            .await
            .unwrap();
        session
            .send_reduce_scatter_chunk(collective_id, 0, 3, 1, 1, 0, 11, &[5.0])
            .await
            .unwrap();

        let one = waiter_one.await.unwrap();
        let two = waiter_two.await.unwrap();
        assert_eq!(one.chunk_data, vec![5.0]);
        assert_eq!(two.chunk_data, vec![6.0]);
    }

    #[tokio::test]
    async fn test_serving_session_supports_multiple_waiters_on_same_slot() {
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
                        collective_seq: 0,
                        lane: CollectiveLane::ReduceScatter,
                        layer_idx: 6,
                        step: 2,
                        slot: 0,
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
                        collective_seq: 0,
                        lane: CollectiveLane::ReduceScatter,
                        layer_idx: 6,
                        step: 2,
                        slot: 0,
                        stream_id: 0,
                        expected_sender_position: 11,
                    })
                    .await
                    .unwrap()
            })
        };

        session
            .send_reduce_scatter_chunk(collective_id, 0, 6, 2, 0, 0, 11, &[5.0])
            .await
            .unwrap();
        session
            .send_reduce_scatter_chunk(collective_id, 0, 6, 2, 0, 0, 11, &[6.0])
            .await
            .unwrap();

        let one = timeout(Duration::from_secs(1), waiter_one)
            .await
            .unwrap()
            .unwrap();
        let two = timeout(Duration::from_secs(1), waiter_two)
            .await
            .unwrap()
            .unwrap();
        let mut payloads = vec![one.chunk_data, two.chunk_data];
        payloads.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert_eq!(payloads, vec![vec![5.0], vec![6.0]]);
    }

    #[tokio::test]
    async fn test_serving_session_discriminates_waiters_by_expected_sender_position() {
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

        let waiter = {
            let session = session.clone();
            tokio::spawn(async move {
                session
                    .recv_frame(ServingReceiveSpec {
                        collective_id,
                        collective_seq: 0,
                        lane: CollectiveLane::ReduceScatter,
                        layer_idx: 5,
                        step: 2,
                        slot: 0,
                        stream_id: 0,
                        expected_sender_position: 11,
                    })
                    .await
                    .unwrap()
            })
        };

        session
            .send_reduce_scatter_chunk(collective_id, 0, 5, 2, 0, 0, 12, &[9.0])
            .await
            .unwrap();
        session
            .send_reduce_scatter_chunk(collective_id, 0, 5, 2, 0, 0, 11, &[7.0])
            .await
            .unwrap();

        let frame = timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(frame.chunk_data, vec![7.0]);
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
        let left_peer = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9);
        let session = plane
            .serving_transport_for_neighbors(
                left_peer,
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
    async fn test_serving_transport_deduplicates_same_neighbor_endpoint() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let target = plane.local_addr();

        plane
            .serving_transport_for_neighbors(
                target,
                target,
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();

        wait_for_metric(&plane, |snapshot| {
            snapshot.current_outbound_connections >= 5
        })
        .await;
        let snapshot = plane.metrics_snapshot();
        assert_eq!(snapshot.current_outbound_connections, 5);
    }

    #[tokio::test]
    async fn test_serving_transport_rebind_refreshes_existing_serving_channels() {
        let plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let target = plane.local_addr();

        plane
            .serving_transport_for_neighbors(
                target,
                target,
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();
        wait_for_metric(&plane, |snapshot| {
            snapshot.current_outbound_connections >= 5
        })
        .await;

        plane
            .serving_transport_for_neighbors(
                target,
                target,
                InferenceRuntimeMode::ThroughputFirst,
                ExecutionProviderKind::Cpu,
            )
            .await
            .unwrap();

        wait_for_metric(&plane, |snapshot| snapshot.connection_evict_count >= 5).await;
        let snapshot = plane.metrics_snapshot();
        assert_eq!(snapshot.current_outbound_connections, 5);
        assert!(snapshot.connection_evict_count >= 5);
    }

    #[tokio::test]
    async fn test_logical_serving_stream_ids_bind_to_stable_underlying_channels() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            max_concurrent_outbound_streams_per_peer: 3,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();
        let target = plane.local_addr();

        let stream_zero = plane
            .ensure_connection_pool(target, CollectiveLane::ReduceScatter, 3, 0, true)
            .await
            .unwrap();
        let stream_one = plane
            .ensure_connection_pool(target, CollectiveLane::ReduceScatter, 3, 1, true)
            .await
            .unwrap();
        let stream_zero_again = plane
            .ensure_connection_pool(target, CollectiveLane::ReduceScatter, 3, 0, true)
            .await
            .unwrap();

        assert!(
            Arc::ptr_eq(&stream_zero, &stream_zero_again),
            "same logical stream id should reuse the same channel"
        );
        assert!(
            !Arc::ptr_eq(&stream_zero, &stream_one),
            "different logical stream ids should use different channels when available"
        );
    }

    #[tokio::test]
    async fn test_lan_profile_opens_more_cuda_latency_streams() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            profile: TensorPlaneProfile::Lan,
            max_concurrent_outbound_streams_per_peer: 4,
            ..TensorPlaneConfig::default()
        })
        .await
        .unwrap();
        let left_peer = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9);
        let session = plane
            .serving_transport_for_neighbors(
                left_peer,
                plane.local_addr(),
                InferenceRuntimeMode::LatencyFirst,
                ExecutionProviderKind::Cuda,
            )
            .await
            .unwrap();

        assert_eq!(
            session.stream_id_for(CollectiveLane::ReduceScatter, 2, 0),
            2
        );
        assert_eq!(session.stream_id_for(CollectiveLane::AllGather, 2, 0), 2);
    }

    #[tokio::test]
    async fn test_closed_lane_connection_refreshes_on_next_send() {
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
        let stale_stream = {
            let streams = session.reduce_scatter_lane.streams.lock().await;
            Arc::clone(&streams[0])
        };
        {
            let mut guard = stale_stream.lock().await;
            guard.shutdown().await.unwrap();
        }

        session
            .send_reduce_scatter_chunk(collective_id, 0, 2, 0, 0, 0, 7, &[4.0, 5.0])
            .await
            .unwrap();
        let frame = session
            .recv_frame(ServingReceiveSpec {
                collective_id,
                collective_seq: 0,
                lane: CollectiveLane::ReduceScatter,
                layer_idx: 2,
                step: 0,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 7,
            })
            .await
            .unwrap();

        assert_eq!(frame.chunk_data, vec![4.0, 5.0]);
        let snapshot = plane.metrics_snapshot();
        assert_eq!(snapshot.connection_refresh_attempt_count, 1);
        assert_eq!(snapshot.connection_refresh_success_count, 1);
        assert_eq!(snapshot.connection_evict_count, 1);
    }

    #[tokio::test]
    async fn test_serving_frame_write_read_roundtrip_preserves_payload_and_shape() {
        let header = ServingFrameHeader::new(
            Uuid::new_v4(),
            4,
            3,
            7,
            2,
            1,
            0,
            CollectiveLane::AllGather,
            3,
        );
        let chunk_data = [1.5_f32, -2.25, 8.0];
        let (mut writer, mut reader) = tokio::io::duplex(1024);

        write_serving_frame(&mut writer, header, &chunk_data)
            .await
            .unwrap();
        let frame = read_serving_frame(&mut reader, DEFAULT_MAX_MESSAGE_BYTES)
            .await
            .unwrap();

        assert_eq!(frame.header, header);
        assert_eq!(decode_f32_slice_wire(&frame.payload_bytes), chunk_data);
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
            .send_bulk_transfer(bulk_collective, 0, 4, 0, 0, 0, 2, &[1.0, 2.0, 3.0])
            .await
            .unwrap();
        session
            .send_checkpoint(checkpoint_collective, 0, 4, 1, 0, 0, 2, &[9.0])
            .await
            .unwrap();

        let bulk = session
            .recv_frame(ServingReceiveSpec {
                collective_id: bulk_collective,
                collective_seq: 0,
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
                collective_seq: 0,
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
            .send_control(collective_id, 0, 7, 0, 0, 0, 3, &[3.0])
            .await
            .unwrap();
        let _ = session
            .recv_frame(ServingReceiveSpec {
                collective_id,
                collective_seq: 0,
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

    #[tokio::test]
    async fn test_serving_session_discriminates_same_layer_collectives_by_sequence() {
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

        session
            .send_reduce_scatter_chunk(collective_id, 1, 4, 0, 0, 0, 11, &[8.0])
            .await
            .unwrap();
        session
            .send_reduce_scatter_chunk(collective_id, 0, 4, 0, 0, 0, 11, &[5.0])
            .await
            .unwrap();

        let frame = session
            .recv_frame(ServingReceiveSpec {
                collective_id,
                collective_seq: 0,
                lane: CollectiveLane::ReduceScatter,
                layer_idx: 4,
                step: 0,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 11,
            })
            .await
            .unwrap();

        assert_eq!(frame.chunk_data, vec![5.0]);
    }

    #[tokio::test]
    async fn test_serving_transport_survives_idle_gap_before_first_frame() {
        let plane = TensorPlane::bind(TensorPlaneConfig {
            io_timeout: Duration::from_millis(50),
            ..TensorPlaneConfig::default()
        })
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

        tokio::time::sleep(Duration::from_millis(125)).await;

        session
            .send_reduce_scatter_chunk(collective_id, 0, 9, 0, 0, 0, 7, &[4.0, 5.0])
            .await
            .unwrap();
        let frame = session
            .recv_frame(ServingReceiveSpec {
                collective_id,
                collective_seq: 0,
                lane: CollectiveLane::ReduceScatter,
                layer_idx: 9,
                step: 0,
                slot: 0,
                stream_id: 0,
                expected_sender_position: 7,
            })
            .await
            .unwrap();

        assert_eq!(frame.chunk_data, vec![4.0, 5.0]);
    }
}
