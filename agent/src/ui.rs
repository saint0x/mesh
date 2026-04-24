use agent::{
    pki::{MembershipRole, PeerCache, PoolConfig, PoolId},
    resource_manager::ResourceManager,
    DeviceCapabilities, DeviceConfig, DeviceConnectivityState, DirectPeerCandidate,
    ExecutionProviderInfo, ShardRegistry,
};
use anyhow::{anyhow, Context, Result};
use axum::{
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use control_plane::{
    api::types::RingTopologyResponse,
    connectivity::InferenceSchedulingPolicy,
    consumption_policy::{quote_consumption, ConsumptionQuoteInput},
    credit_policy::{compute_credit_policy, AssignmentCreditInput, CreditPolicyInput},
    model_assets, Database,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::UNIX_EPOCH,
};
use tokio::{process::Child, sync::Mutex};
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct UiState {
    db: Database,
    mesh_home: PathBuf,
    client: reqwest::Client,
    daemon: Arc<Mutex<Option<Child>>>,
    control_plane: Arc<Mutex<Option<Child>>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiEnvelope<T> {
    ok: bool,
    data: T,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiErrorEnvelope {
    ok: bool,
    error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ApiErrorBody {
    code: String,
    message: String,
    hint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DashboardSnapshot {
    generated_at: String,
    mesh_home: String,
    local_device_id: Option<String>,
    networks: Vec<UiNetwork>,
    devices: Vec<UiDevice>,
    models: Vec<UiModel>,
    jobs: Vec<UiJob>,
    ledger_events: Vec<UiLedgerEvent>,
    topologies: Vec<UiTopology>,
    runtime_stats: Option<Value>,
    resource_lock: Option<UiResourceLockStatus>,
    pools: Vec<UiPoolSummary>,
    doctor: Option<UiDoctorReport>,
    settings: UiSettings,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiNetwork {
    id: String,
    name: String,
    owner: String,
    created_at: String,
    preferred_path: String,
    attachments: Vec<UiAttachment>,
    scheduling_policy: UiSchedulingPolicy,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiAttachment {
    kind: String,
    endpoint: String,
    priority: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiSchedulingPolicy {
    submitter_active_job_soft_cap: u32,
    model_active_job_soft_cap_divisor: u32,
    capacity_unit_soft_cap_divisor: u32,
    tier_capacity_units: UiTierCapacityUnits,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiTierCapacityUnits {
    tier0: u32,
    tier1: u32,
    tier2: u32,
    tier3: u32,
    tier4: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiDevice {
    id: String,
    network_id: String,
    name: String,
    peer_id: Option<String>,
    status: String,
    health: String,
    last_seen: Option<String>,
    ring_position: Option<u32>,
    left_neighbor_id: Option<String>,
    right_neighbor_id: Option<String>,
    shard_model_id: Option<String>,
    shard_column_start: Option<u32>,
    shard_column_end: Option<u32>,
    contributed_memory_bytes: Option<u64>,
    connectivity_state: Option<UiConnectivityState>,
    listen_addrs: Vec<String>,
    tensor_plane_endpoints: Vec<String>,
    direct_candidates: Vec<DirectPeerCandidate>,
    capabilities: UiCapabilities,
    certificate_status: String,
    identity_status: String,
    local_device: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiConnectivityState {
    active_path: String,
    active_endpoint: Option<String>,
    status: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiCapabilities {
    tier: String,
    cpu_cores: usize,
    ram_mb: usize,
    gpu_present: bool,
    gpu_vram_mb: Option<usize>,
    os: String,
    arch: String,
    execution_providers: Vec<ExecutionProviderInfo>,
    default_execution_provider: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiModel {
    id: String,
    network_ids: Vec<String>,
    total_model_bytes: Option<u64>,
    tensor_parallelism_dim: Option<u32>,
    artifact_ready: bool,
    tokenizer_ready: bool,
    manifest_count: usize,
    weights_count: usize,
    participant_count: usize,
    loaded_local_shard: bool,
    local_shard_range: Option<UiShardRange>,
    local_memory_bytes: Option<u64>,
    shard_status: Option<String>,
    provider_compatibility: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiShardRange {
    start: u32,
    end: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiJob {
    id: String,
    network_id: String,
    model_id: String,
    status: String,
    submitted_by_device_id: String,
    submitted_by_name: String,
    ring_worker_count: u32,
    created_at: String,
    started_at: Option<String>,
    completed_at: Option<String>,
    completion_tokens: u32,
    prompt_tokens: Option<u32>,
    execution_time_ms: u64,
    reserved_credits: f64,
    settled_credits: f64,
    released_credits: f64,
    available_completion_tokens: u32,
    model_size_factor: f64,
    accounted_completion_tokens: u32,
    prompt_credits_accounted: bool,
    error: Option<String>,
    credit_policy: Option<UiCreditPolicy>,
    assignments: Vec<UiAssignment>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiCreditPolicy {
    job_credit_budget: f64,
    assignments: Vec<UiAssignmentCreditBreakdown>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiAssignmentCreditBreakdown {
    device_id: String,
    credits: f64,
    compute_share: f64,
    throughput_multiplier: f64,
    resource_pressure_multiplier: f64,
    normalized_contribution_share: f64,
    measured_service_rate: f64,
    reference_service_rate: f64,
    memory_pressure: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiAssignment {
    assignment_id: String,
    device_id: String,
    device_name: String,
    ring_position: u32,
    status: String,
    lease_expires_at: Option<String>,
    assigned_at: String,
    acknowledged_at: Option<String>,
    completed_at: Option<String>,
    failure_reason: Option<String>,
    execution_time_ms: u64,
    shard_column_start: Option<u32>,
    shard_column_end: Option<u32>,
    assigned_capacity_units: u32,
    execution_provider: Option<String>,
    reported_completion_tokens: u32,
    credits_earned: Option<f64>,
    throughput_multiplier: Option<f64>,
    resource_pressure_multiplier: Option<f64>,
    normalized_contribution_share: Option<f64>,
    available_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiLedgerEvent {
    id: String,
    network_id: String,
    event_type: String,
    job_id: Option<String>,
    device_id: Option<String>,
    credits_amount: Option<f64>,
    detail: String,
    metadata: Value,
    created_at: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiTopology {
    network_id: String,
    source: String,
    ring_stable: bool,
    workers: Vec<UiTopologyWorker>,
    punch_plans: Vec<UiPunchPlan>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiTopologyWorker {
    device_id: String,
    device_name: String,
    peer_id: Option<String>,
    position: Option<u32>,
    status: String,
    contributed_memory_bytes: Option<u64>,
    shard_column_start: Option<u32>,
    shard_column_end: Option<u32>,
    left_neighbor_id: Option<String>,
    right_neighbor_id: Option<String>,
    active_path: Option<String>,
    active_endpoint: Option<String>,
    tensor_plane_endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiPunchPlan {
    source_device_id: String,
    target_device_id: String,
    target_peer_id: String,
    reason: String,
    strategy: String,
    relay_rendezvous_required: bool,
    attempt_window_ms: u64,
    issued_at_ms: u64,
    target_candidates: Vec<DirectPeerCandidate>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiResourceLockStatus {
    status: String,
    total_memory_bytes: u64,
    user_allocated_bytes: u64,
    locked_memory_bytes: u64,
    lock_timestamp_ms: Option<u64>,
    ready_to_unlock: bool,
    unlock_in_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiPoolSummary {
    id: String,
    name: String,
    role: String,
    created_at: String,
    expires_at: u64,
    days_until_expiry: Option<i64>,
    peer_count: usize,
    root_pubkey_hex: String,
    valid_cert: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiPoolPeer {
    node_id: String,
    lan_addr: String,
    discovery_method: String,
    last_seen: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiDoctorReport {
    generated_at: String,
    overall: String,
    checks: Vec<UiDoctorCheck>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiDoctorCheck {
    id: String,
    label: String,
    status: String,
    detail: String,
    hint: Option<String>,
    duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiSettings {
    control_plane_url: Option<String>,
    local_device_name: Option<String>,
    preferred_provider: Option<String>,
    governance: Value,
    relay: Value,
    config_paths: UiConfigPaths,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct UiConfigPaths {
    device_config: String,
    device_certificate: String,
    relay_config: String,
    control_plane_db: String,
    shard_registry: String,
}

#[derive(Debug, serde::Deserialize)]
struct ModelManifest {
    model_id: String,
    tensor_parallelism_dim: u32,
    total_model_bytes: u64,
}

#[derive(Debug, serde::Deserialize)]
struct ShardRegistryEntry {
    info: ShardRegistryInfo,
    status: String,
}

#[derive(Debug, serde::Deserialize)]
struct ShardRegistryInfo {
    assignment: ShardRegistryAssignment,
    is_loaded: bool,
    memory_bytes: u64,
}

#[derive(Debug, serde::Deserialize)]
struct ShardRegistryAssignment {
    column_start: u32,
    column_end: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DashboardQuery {
    include: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeviceInitRequest {
    network_id: String,
    name: String,
    control_plane_url: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeviceStartRequest {
    log_level: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResourceLockRequest {
    memory: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RingJoinRequest {
    model_id: String,
    memory: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JobSubmitRequest {
    prompt: String,
    model_id: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct QuoteQuery {
    prompt_tokens: u32,
    max_tokens: u32,
    network_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct QuoteResponse {
    model_id: String,
    network_id: String,
    model_size_factor: f64,
    prompt_tokens: u32,
    max_tokens: u32,
    prompt_credits: f64,
    completion_credits_cap: f64,
    total_credits_cap: f64,
    available_completion_tokens: u32,
    device_available_credits: f64,
    feasible: bool,
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LedgerQuery {
    job_id: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PoolCreateRequest {
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PoolJoinRequest {
    pool_id: String,
    pool_root_pubkey: String,
    name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DeviceStatusResponse {
    configured: bool,
    device_id: Option<String>,
    network_id: Option<String>,
    name: Option<String>,
    control_plane_url: Option<String>,
    preferred_provider: Option<String>,
    has_certificate: bool,
    daemon_running: bool,
    listen_addrs: Vec<String>,
    observed_addrs: Vec<String>,
    direct_candidate_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LedgerSummaryPayload {
    total_credits_earned: f64,
    total_credits_burned: f64,
}

pub async fn cmd_ui(port: u16, api_port: u16, api_only: bool) -> Result<()> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow!("Could not resolve workspace root"))?
        .to_path_buf();
    let ui_dir = workspace_root.join("mesh-ui");
    let mesh_home = dirs::home_dir()
        .ok_or_else(|| anyhow!("Could not determine home directory"))?
        .join(".meshnet");

    let db_path = Database::default_path()?;
    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| anyhow!("Invalid control-plane db path"))?;
    let db = Database::new(db_path_str)?;
    db.migrate()
        .context("Failed to migrate the local control-plane database for Mesh UI startup")?;

    let api_state = UiState {
        db,
        mesh_home,
        client: reqwest::Client::new(),
        daemon: Arc::new(Mutex::new(None)),
        control_plane: Arc::new(Mutex::new(None)),
    };

    ensure_local_control_plane_running(&api_state).await?;

    if !api_only {
        ensure_mesh_ui_dependencies(&ui_dir)?;
        build_mesh_ui(&ui_dir, api_port)?;
    }

    let api_addr = SocketAddr::from(([127, 0, 0, 1], api_port));
    let listener = tokio::net::TcpListener::bind(api_addr)
        .await
        .with_context(|| format!("Failed to bind Mesh UI API on {}", api_addr))?;
    let api_router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/local/dashboard", get(get_dashboard))
        .route("/api/local/device/init", post(init_device))
        .route("/api/local/device/start", post(start_device))
        .route("/api/local/device/stop", post(stop_device))
        .route("/api/local/device/status", get(get_device_status))
        .route("/api/local/resource/lock", post(lock_resources))
        .route("/api/local/resource/unlock", post(unlock_resources))
        .route("/api/local/resource/status", get(get_resource_status))
        .route("/api/local/ring/join", post(join_ring))
        .route("/api/local/ring/leave", post(leave_ring))
        .route("/api/local/ring/status", get(get_ring_status))
        .route("/api/local/ring/topology", get(get_ring_topology))
        .route("/api/local/ring/shard", get(get_ring_shard))
        .route("/api/local/jobs", post(submit_job))
        .route("/api/local/jobs/stats", get(get_job_stats))
        .route(
            "/api/local/jobs/:job_id",
            get(get_job_status).delete(cancel_job),
        )
        .route("/api/local/ledger/summary", get(get_ledger_summary))
        .route("/api/local/ledger/events", get(get_ledger_events))
        .route("/api/local/pools", get(list_pools).post(create_pool))
        .route("/api/local/pools/join", post(join_pool))
        .route("/api/local/pools/:pool_id/peers", get(get_pool_peers))
        .route("/api/local/pools/:pool_id/status", get(get_pool_status))
        .route("/api/local/doctor", post(run_doctor))
        .route("/api/local/models/:model_id/quote", get(get_model_quote))
        .with_state(api_state.clone())
        .layer(CorsLayer::permissive());
    let api_task = tokio::spawn(async move { axum::serve(listener, api_router).await });

    let ui_url = format!("http://127.0.0.1:{port}");
    let api_url = format!("http://127.0.0.1:{api_port}");
    println!("Mesh UI API: {api_url}");
    if !api_only {
        println!("Mesh UI: {ui_url}");

        let mut child = tokio::process::Command::new("pnpm")
            .arg("preview")
            .arg("--")
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--strictPort")
            .current_dir(&ui_dir)
            .spawn()
            .context("Failed to start mesh-ui preview server")?;

        tokio::select! {
            result = child.wait() => {
                let status = result.context("Failed while waiting for mesh-ui preview server")?;
                if !status.success() {
                    return Err(anyhow!("mesh-ui preview server exited with {}", status));
                }
            }
            result = api_task => {
                match result {
                    Ok(Ok(())) => return Err(anyhow!("Mesh UI API exited unexpectedly")),
                    Ok(Err(error)) => return Err(anyhow!("Mesh UI API failed: {}", error)),
                    Err(error) => return Err(anyhow!("Mesh UI API task failed: {}", error)),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                let _ = child.kill().await;
                kill_tracked_process(&api_state.control_plane).await;
                kill_tracked_process(&api_state.daemon).await;
            }
        }
    } else {
        tokio::select! {
            result = api_task => {
                match result {
                    Ok(Ok(())) => return Err(anyhow!("Mesh UI API exited unexpectedly")),
                    Ok(Err(error)) => return Err(anyhow!("Mesh UI API failed: {}", error)),
                    Err(error) => return Err(anyhow!("Mesh UI API task failed: {}", error)),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                kill_tracked_process(&api_state.control_plane).await;
                kill_tracked_process(&api_state.daemon).await;
            }
        }
    }

    Ok(())
}

async fn get_dashboard(
    State(state): State<UiState>,
    Query(query): Query<DashboardQuery>,
) -> Response {
    let includes = parse_includes(query.include.as_deref());
    match load_dashboard_snapshot(&state, &includes).await {
        Ok(snapshot) => ok_response(snapshot),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "dashboard_load_failed",
            error.to_string(),
            Some(
                "Start Mesh UI with `mesh ui` so the local API and control plane are available."
                    .into(),
            ),
        ),
    }
}

async fn init_device(
    State(state): State<UiState>,
    Json(body): Json<DeviceInitRequest>,
) -> Response {
    if let Err(error) = ensure_control_plane_running_for_url(&state, &body.control_plane_url).await
    {
        return error_response(
            StatusCode::BAD_GATEWAY,
            "control_plane_start_failed",
            error.to_string(),
            Some(
                "Mesh UI could not bring up the local control plane before initialization.".into(),
            ),
        );
    }

    match crate::cmd_init(body.network_id, body.name, body.control_plane_url).await {
        Ok(()) => ok_response(serde_json::json!({ "initialized": true })),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "device_init_failed",
            error.to_string(),
            Some("Check control-plane reachability and network identity fields.".into()),
        ),
    }
}

async fn start_device(
    State(state): State<UiState>,
    Json(body): Json<DeviceStartRequest>,
) -> Response {
    if let Err(error) = ensure_local_control_plane_running(&state).await {
        return error_response(
            StatusCode::BAD_GATEWAY,
            "control_plane_start_failed",
            error.to_string(),
            Some(
                "Mesh UI could not bring up the local control plane before starting the daemon."
                    .into(),
            ),
        );
    }

    let mut daemon = state.daemon.lock().await;
    if daemon.is_some() {
        return ok_response(serde_json::json!({ "running": true }));
    }

    let current_exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "device_start_failed",
                error.to_string(),
                None,
            )
        }
    };

    let child = match tokio::process::Command::new(current_exe)
        .arg("device")
        .arg("start")
        .arg("--log-level")
        .arg(body.log_level.unwrap_or_else(|| "info".to_string()))
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "device_start_failed",
                error.to_string(),
                Some("Ensure the agent binary is available and executable.".into()),
            )
        }
    };

    *daemon = Some(child);
    ok_response(serde_json::json!({ "running": true }))
}

async fn stop_device(State(state): State<UiState>) -> Response {
    let mut daemon = state.daemon.lock().await;
    if let Some(child) = daemon.as_mut() {
        let _ = child.kill().await;
    }
    *daemon = None;
    ok_response(serde_json::json!({ "running": false }))
}

async fn get_device_status(State(state): State<UiState>) -> Response {
    match build_device_status(&state).await {
        Ok(status) => ok_response(status),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "device_status_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn lock_resources(
    State(_state): State<UiState>,
    Json(body): Json<ResourceLockRequest>,
) -> Response {
    match crate::cmd_lock_resources(body.memory).await {
        Ok(()) => ok_response(serde_json::json!({ "locked": true })),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "resource_lock_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn unlock_resources(State(_state): State<UiState>) -> Response {
    match crate::cmd_unlock_resources().await {
        Ok(()) => ok_response(serde_json::json!({ "locked": false })),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "resource_unlock_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_resource_status(State(_state): State<UiState>) -> Response {
    match load_resource_lock_status() {
        Ok(status) => ok_response(status),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "resource_status_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn join_ring(State(_state): State<UiState>, Json(body): Json<RingJoinRequest>) -> Response {
    match crate::cmd_join_ring(body.model_id, body.memory).await {
        Ok(()) => ok_response(serde_json::json!({ "joined": true })),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "ring_join_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn leave_ring(State(_state): State<UiState>) -> Response {
    match crate::cmd_leave_ring().await {
        Ok(()) => ok_response(serde_json::json!({ "joined": false })),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "ring_leave_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_ring_status(State(state): State<UiState>) -> Response {
    match load_ring_status(&state).await {
        Ok(status) => ok_response(status),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "ring_status_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_ring_topology(State(state): State<UiState>) -> Response {
    match load_live_topology(&state).await {
        Ok(topology) => ok_response(topology),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "ring_topology_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_ring_shard(State(_state): State<UiState>) -> Response {
    match load_local_shards().await {
        Ok(shards) => ok_response(shards),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "ring_shard_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn submit_job(State(state): State<UiState>, Json(body): Json<JobSubmitRequest>) -> Response {
    let config = match load_local_device_config() {
        Ok(Some(config)) => config,
        Ok(None) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "device_not_initialized",
                "Device is not initialized.".into(),
                Some("Run device init before submitting jobs.".into()),
            )
        }
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "device_config_failed",
                error.to_string(),
                None,
            )
        }
    };

    let payload = serde_json::json!({
        "device_id": config.device_id.to_string(),
        "network_id": config.network_id,
        "model_id": body.model_id,
        "prompt": body.prompt,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
    });
    match proxy_json(
        &state.client,
        reqwest::Method::POST,
        control_plane_url(&config, "/api/inference/submit"),
        Some(payload),
    )
    .await
    {
        Ok(value) => ok_response(value),
        Err(error) => error_response(
            StatusCode::BAD_GATEWAY,
            "job_submit_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_job_status(
    State(state): State<UiState>,
    AxumPath(job_id): AxumPath<String>,
) -> Response {
    match with_device_config(|config| async move {
        proxy_json(
            &state.client,
            reqwest::Method::GET,
            control_plane_url(&config, &format!("/api/inference/jobs/{job_id}")),
            None,
        )
        .await
    })
    .await
    {
        Ok(value) => ok_response(value),
        Err(error) => error_response(
            StatusCode::BAD_GATEWAY,
            "job_status_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn cancel_job(State(state): State<UiState>, AxumPath(job_id): AxumPath<String>) -> Response {
    match with_device_config(|config| async move {
        proxy_json(
            &state.client,
            reqwest::Method::DELETE,
            control_plane_url(&config, &format!("/api/inference/jobs/{job_id}")),
            None,
        )
        .await
    })
    .await
    {
        Ok(value) => ok_response(value),
        Err(error) => error_response(
            StatusCode::BAD_GATEWAY,
            "job_cancel_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_job_stats(State(_state): State<UiState>) -> Response {
    match load_runtime_stats() {
        Ok(stats) => ok_response(stats),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "job_stats_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_ledger_summary(
    State(state): State<UiState>,
    Query(query): Query<LedgerQuery>,
) -> Response {
    match with_device_config(|config| async move {
        let mut url = control_plane_url(
            &config,
            &format!("/api/ledger/summary?network_id={}", config.network_id),
        );
        if let Some(job_id) = query.job_id {
            url.push_str("&job_id=");
            url.push_str(&job_id);
        }
        proxy_json(&state.client, reqwest::Method::GET, url, None).await
    })
    .await
    {
        Ok(value) => ok_response(value),
        Err(error) => error_response(
            StatusCode::BAD_GATEWAY,
            "ledger_summary_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_ledger_events(
    State(state): State<UiState>,
    Query(query): Query<LedgerQuery>,
) -> Response {
    match with_device_config(|config| async move {
        let limit = query.limit.unwrap_or(100);
        let mut url = control_plane_url(
            &config,
            &format!(
                "/api/ledger/events?network_id={}&limit={limit}",
                config.network_id
            ),
        );
        if let Some(job_id) = query.job_id {
            url.push_str("&job_id=");
            url.push_str(&job_id);
        }
        proxy_json(&state.client, reqwest::Method::GET, url, None).await
    })
    .await
    {
        Ok(value) => ok_response(value),
        Err(error) => error_response(
            StatusCode::BAD_GATEWAY,
            "ledger_events_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn create_pool(
    State(_state): State<UiState>,
    Json(body): Json<PoolCreateRequest>,
) -> Response {
    match crate::cmd_pool_create(body.name).await {
        Ok(()) => match load_pools() {
            Ok(pools) => ok_response(pools),
            Err(error) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "pool_load_failed",
                error.to_string(),
                None,
            ),
        },
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "pool_create_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn join_pool(State(_state): State<UiState>, Json(body): Json<PoolJoinRequest>) -> Response {
    match crate::cmd_pool_join(body.pool_id, body.pool_root_pubkey, body.name).await {
        Ok(()) => match load_pools() {
            Ok(pools) => ok_response(pools),
            Err(error) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "pool_load_failed",
                error.to_string(),
                None,
            ),
        },
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "pool_join_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn list_pools(State(_state): State<UiState>) -> Response {
    match load_pools() {
        Ok(pools) => ok_response(pools),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "pool_list_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_pool_peers(
    State(_state): State<UiState>,
    AxumPath(pool_id): AxumPath<String>,
) -> Response {
    match load_pool_peers(&pool_id) {
        Ok(peers) => ok_response(peers),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "pool_peers_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_pool_status(
    State(_state): State<UiState>,
    AxumPath(pool_id): AxumPath<String>,
) -> Response {
    match load_pool_status(&pool_id) {
        Ok(status) => ok_response(status),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "pool_status_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn run_doctor(State(_state): State<UiState>) -> Response {
    match build_doctor_report().await {
        Ok(report) => ok_response(report),
        Err(error) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "doctor_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn get_model_quote(
    State(state): State<UiState>,
    AxumPath(model_id): AxumPath<String>,
    Query(query): Query<QuoteQuery>,
) -> Response {
    let config = match load_local_device_config() {
        Ok(Some(config)) => config,
        Ok(None) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "device_not_initialized",
                "Device is not initialized.".into(),
                None,
            )
        }
        Err(error) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "device_config_failed",
                error.to_string(),
                None,
            )
        }
    };

    match build_quote_response(&state, &config, &model_id, query).await {
        Ok(response) => ok_response(response),
        Err(error) => error_response(
            StatusCode::BAD_REQUEST,
            "quote_failed",
            error.to_string(),
            None,
        ),
    }
}

async fn load_dashboard_snapshot(
    state: &UiState,
    includes: &HashSet<String>,
) -> Result<DashboardSnapshot> {
    let local_device_config = load_local_device_config()?;
    let networks = load_networks(&state.db)?;
    let devices = load_devices(&state.db, local_device_config.as_ref())?;
    let jobs = load_jobs(&state.db, &devices)?;
    let ledger_events = load_ledger_events(&state.db)?;
    let topologies = load_topologies(
        &networks,
        &devices,
        &state.client,
        local_device_config
            .as_ref()
            .map(|config| config.control_plane_url.as_str()),
    )
    .await;
    let models = load_models(&state.mesh_home, &devices)?;
    let settings = load_settings(
        &state.mesh_home,
        &local_device_config,
        &Database::default_path()?,
    )?;

    Ok(DashboardSnapshot {
        generated_at: chrono::Utc::now().to_rfc3339(),
        mesh_home: state.mesh_home.display().to_string(),
        local_device_id: local_device_config
            .as_ref()
            .map(|config| config.device_id.to_string()),
        networks,
        devices,
        models,
        jobs,
        ledger_events,
        topologies,
        runtime_stats: if includes.contains("runtimeStats") {
            Some(load_runtime_stats()?)
        } else {
            None
        },
        resource_lock: if includes.contains("resourceLock") {
            Some(load_resource_lock_status()?)
        } else {
            None
        },
        pools: if includes.contains("pools") {
            load_pools()?
        } else {
            Vec::new()
        },
        doctor: if includes.contains("doctor") {
            Some(build_doctor_report().await?)
        } else {
            None
        },
        settings,
    })
}

fn ensure_mesh_ui_dependencies(ui_dir: &Path) -> Result<()> {
    if !ui_dir.join("node_modules").exists() {
        let status = Command::new("pnpm")
            .arg("install")
            .arg("--frozen-lockfile")
            .current_dir(ui_dir)
            .status()
            .context("Failed to install mesh-ui dependencies with pnpm")?;
        if !status.success() {
            return Err(anyhow!("pnpm install failed with {}", status));
        }
    }
    Ok(())
}

fn build_mesh_ui(ui_dir: &Path, api_port: u16) -> Result<()> {
    let status = Command::new("pnpm")
        .arg("build")
        .env(
            "VITE_MESH_UI_API_BASE",
            format!("http://127.0.0.1:{api_port}"),
        )
        .current_dir(ui_dir)
        .status()
        .context("Failed to build mesh-ui")?;
    if !status.success() {
        return Err(anyhow!("pnpm build failed with {}", status));
    }
    Ok(())
}

fn load_networks(db: &Database) -> Result<Vec<UiNetwork>> {
    let records = control_plane::services::network_service::list_networks(db)?;
    Ok(records
        .into_iter()
        .map(|record| UiNetwork {
            id: record.network_id,
            name: record.name,
            owner: record.owner_user_id,
            created_at: record.created_at,
            preferred_path: format!("{:?}", record.connectivity.preferred_path)
                .to_ascii_lowercase(),
            attachments: record
                .connectivity
                .attachments
                .into_iter()
                .map(|attachment| UiAttachment {
                    kind: format!("{:?}", attachment.kind).to_ascii_lowercase(),
                    endpoint: attachment.endpoint,
                    priority: attachment.priority,
                })
                .collect(),
            scheduling_policy: scheduling_policy(record.scheduling_policy),
        })
        .collect())
}

fn load_devices(
    db: &Database,
    local_device_config: Option<&DeviceConfig>,
) -> Result<Vec<UiDevice>> {
    let conn = db.get_conn()?;
    let local_device_id = local_device_config.map(|config| config.device_id.to_string());
    let mut stmt = conn.prepare(
        r#"
        SELECT
            device_id,
            network_id,
            name,
            peer_id,
            status,
            last_seen,
            ring_position,
            left_neighbor_id,
            right_neighbor_id,
            shard_model_id,
            shard_column_start,
            shard_column_end,
            contributed_memory,
            connectivity_state,
            listen_addrs,
            direct_candidates,
            capabilities,
            certificate
        FROM devices
        ORDER BY created_at DESC, device_id DESC
        "#,
    )?;

    let rows = stmt.query_map([], |row| {
        let capabilities_json: String = row.get(16)?;
        let capabilities: DeviceCapabilities =
            serde_json::from_str(&capabilities_json).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    capabilities_json.len(),
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
        let connectivity_state =
            parse_optional_json::<DeviceConnectivityState>(row.get::<_, Option<String>>(13)?)
                .map_err(to_sql_error)?;
        let listen_addrs = parse_optional_json::<Vec<String>>(row.get::<_, Option<String>>(14)?)
            .map_err(to_sql_error)?
            .unwrap_or_default();
        let direct_candidates =
            parse_optional_json::<Vec<DirectPeerCandidate>>(row.get::<_, Option<String>>(15)?)
                .map_err(to_sql_error)?
                .unwrap_or_default();
        let certificate: Option<Vec<u8>> = row.get(17)?;
        let device_id: String = row.get(0)?;

        Ok(UiDevice {
            id: device_id.clone(),
            network_id: row.get(1)?,
            name: row.get(2)?,
            peer_id: row.get(3)?,
            status: row.get(4)?,
            health: health_label(
                row.get::<_, String>(4)?.as_str(),
                connectivity_state.as_ref(),
            ),
            last_seen: row.get(5)?,
            ring_position: row.get::<_, Option<i64>>(6)?.map(|value| value as u32),
            left_neighbor_id: row.get(7)?,
            right_neighbor_id: row.get(8)?,
            shard_model_id: row.get(9)?,
            shard_column_start: row.get::<_, Option<i64>>(10)?.map(|value| value as u32),
            shard_column_end: row.get::<_, Option<i64>>(11)?.map(|value| value as u32),
            contributed_memory_bytes: row.get::<_, Option<i64>>(12)?.map(|value| value as u64),
            connectivity_state: connectivity_state.map(|state| UiConnectivityState {
                active_path: format!("{:?}", state.active_path).to_ascii_lowercase(),
                active_endpoint: state.active_endpoint,
                status: format!("{:?}", state.status).to_ascii_lowercase(),
            }),
            tensor_plane_endpoints: listen_addrs
                .iter()
                .filter(|addr| addr.starts_with("dataplane://"))
                .cloned()
                .collect(),
            listen_addrs,
            direct_candidates,
            capabilities: UiCapabilities {
                tier: format!("{:?}", capabilities.tier),
                cpu_cores: capabilities.cpu_cores,
                ram_mb: capabilities.ram_mb,
                gpu_present: capabilities.gpu_present,
                gpu_vram_mb: capabilities.gpu_vram_mb,
                os: capabilities.os,
                arch: capabilities.arch,
                execution_providers: capabilities.execution_providers,
                default_execution_provider: capabilities
                    .default_execution_provider
                    .as_str()
                    .to_string(),
            },
            certificate_status: if certificate.is_some() {
                "present".to_string()
            } else {
                "missing".to_string()
            },
            identity_status: "configured".to_string(),
            local_device: local_device_id
                .as_ref()
                .map(|current_id| current_id == &device_id)
                .unwrap_or(false),
        })
    })?;

    let mut devices = Vec::new();
    for row in rows {
        devices.push(row?);
    }
    Ok(devices)
}

fn load_jobs(db: &Database, devices: &[UiDevice]) -> Result<Vec<UiJob>> {
    let conn = db.get_conn()?;
    let device_by_id: HashMap<&str, &UiDevice> = devices
        .iter()
        .map(|device| (device.id.as_str(), device))
        .collect();

    let mut job_stmt = conn.prepare(
        r#"
        SELECT
            job_id,
            network_id,
            submitted_by_device_id,
            model_id,
            status,
            ring_worker_count,
            created_at,
            started_at,
            completed_at,
            completion_tokens,
            prompt_tokens,
            execution_time_ms,
            reserved_credits,
            settled_credits,
            released_credits,
            available_completion_tokens,
            model_size_factor,
            accounted_completion_tokens,
            prompt_credits_accounted,
            error
        FROM inference_jobs
        ORDER BY created_at DESC, job_id DESC
        "#,
    )?;

    let jobs = job_stmt
        .query_map([], |row| {
            let id: String = row.get(0)?;
            let network_id: String = row.get(1)?;
            let submitted_by_device_id: String = row.get(2)?;
            let prompt_tokens_json: Option<String> = row.get(10)?;
            let prompt_tokens = prompt_tokens_json
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Vec<u32>>(raw).ok())
                .map(|tokens| tokens.len() as u32);
            let assignments = load_assignments(&conn, &id, &device_by_id).map_err(to_sql_error)?;
            let credit_policy = build_credit_policy(
                &row.get::<_, String>(3)?,
                prompt_tokens.unwrap_or(0),
                &assignments,
            );
            let assignment_credits = credit_policy
                .as_ref()
                .map(|policy| {
                    policy
                        .assignments
                        .iter()
                        .map(|assignment| (assignment.device_id.as_str(), assignment))
                        .collect::<HashMap<_, _>>()
                })
                .unwrap_or_default();
            let assignments = assignments
                .into_iter()
                .map(|mut assignment| {
                    if let Some(breakdown) = assignment_credits.get(assignment.device_id.as_str()) {
                        assignment.credits_earned = Some(breakdown.credits);
                        assignment.throughput_multiplier = Some(breakdown.throughput_multiplier);
                        assignment.resource_pressure_multiplier =
                            Some(breakdown.resource_pressure_multiplier);
                        assignment.normalized_contribution_share =
                            Some(breakdown.normalized_contribution_share);
                    }
                    assignment
                })
                .collect();

            Ok(UiJob {
                id,
                network_id,
                model_id: row.get(3)?,
                status: row.get(4)?,
                submitted_by_name: device_by_id
                    .get(submitted_by_device_id.as_str())
                    .map(|device| device.name.clone())
                    .unwrap_or_else(|| submitted_by_device_id.clone()),
                submitted_by_device_id,
                ring_worker_count: row.get::<_, i64>(5)? as u32,
                created_at: row.get(6)?,
                started_at: row.get(7)?,
                completed_at: row.get(8)?,
                completion_tokens: row.get::<_, i64>(9)? as u32,
                prompt_tokens,
                execution_time_ms: row.get::<_, i64>(11)? as u64,
                reserved_credits: row.get(12)?,
                settled_credits: row.get(13)?,
                released_credits: row.get(14)?,
                available_completion_tokens: row.get::<_, i64>(15)? as u32,
                model_size_factor: row.get(16)?,
                accounted_completion_tokens: row.get::<_, i64>(17)? as u32,
                prompt_credits_accounted: row.get::<_, i64>(18)? != 0,
                error: row.get(19)?,
                credit_policy,
                assignments,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(jobs)
}

fn load_assignments(
    conn: &rusqlite::Connection,
    job_id: &str,
    device_by_id: &HashMap<&str, &UiDevice>,
) -> Result<Vec<UiAssignment>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT
            assignment_id,
            device_id,
            ring_position,
            status,
            lease_expires_at,
            assigned_at,
            acknowledged_at,
            completed_at,
            failure_reason,
            execution_time_ms,
            shard_column_start,
            shard_column_end,
            assigned_capacity_units,
            execution_provider,
            reported_completion_tokens
        FROM inference_job_assignments
        WHERE job_id = ?
        ORDER BY ring_position ASC, assignment_id ASC
        "#,
    )?;

    let assignments = stmt
        .query_map([job_id], |row| {
            let device_id: String = row.get(1)?;
            let device = device_by_id.get(device_id.as_str()).copied();
            Ok(UiAssignment {
                assignment_id: row.get(0)?,
                device_name: device
                    .map(|device| device.name.clone())
                    .unwrap_or_else(|| device_id.clone()),
                device_id,
                ring_position: row.get::<_, i64>(2)? as u32,
                status: row.get(3)?,
                lease_expires_at: row.get(4)?,
                assigned_at: row.get(5)?,
                acknowledged_at: row.get(6)?,
                completed_at: row.get(7)?,
                failure_reason: row.get(8)?,
                execution_time_ms: row.get::<_, i64>(9)? as u64,
                shard_column_start: row.get::<_, Option<i64>>(10)?.map(|value| value as u32),
                shard_column_end: row.get::<_, Option<i64>>(11)?.map(|value| value as u32),
                assigned_capacity_units: row.get::<_, i64>(12)? as u32,
                execution_provider: row.get(13)?,
                reported_completion_tokens: row.get::<_, i64>(14)?.max(0) as u32,
                credits_earned: None,
                throughput_multiplier: None,
                resource_pressure_multiplier: None,
                normalized_contribution_share: None,
                available_memory_bytes: device.map(available_memory_bytes),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(assignments)
}

fn build_credit_policy(
    model_id: &str,
    prompt_tokens: u32,
    assignments: &[UiAssignment],
) -> Option<UiCreditPolicy> {
    if assignments.is_empty()
        || assignments
            .iter()
            .any(|assignment| assignment.completed_at.is_none())
    {
        return None;
    }
    let manifest = model_assets::load_model_manifest(model_id).ok()?;
    let completion_tokens = assignments
        .iter()
        .map(|assignment| assignment.reported_completion_tokens)
        .min()
        .unwrap_or_default();
    let input = CreditPolicyInput {
        prompt_tokens,
        completion_tokens,
        total_model_bytes: manifest.total_model_bytes,
        total_columns: manifest.tensor_parallelism_dim,
        assignments: assignments
            .iter()
            .map(|assignment| AssignmentCreditInput {
                device_id: assignment.device_id.clone(),
                execution_time_ms: assignment.execution_time_ms,
                assigned_capacity_units: assignment.assigned_capacity_units,
                shard_column_start: assignment.shard_column_start.unwrap_or_default(),
                shard_column_end: assignment
                    .shard_column_end
                    .unwrap_or(manifest.tensor_parallelism_dim),
                available_memory_bytes: assignment.available_memory_bytes.unwrap_or(1),
            })
            .collect(),
    };
    let output = compute_credit_policy(input);
    Some(UiCreditPolicy {
        job_credit_budget: output.job_credit_budget,
        assignments: output
            .assignments
            .into_iter()
            .map(|assignment| UiAssignmentCreditBreakdown {
                device_id: assignment.device_id,
                credits: assignment.credits,
                compute_share: assignment.compute_share,
                throughput_multiplier: assignment.throughput_multiplier,
                resource_pressure_multiplier: assignment.resource_pressure_multiplier,
                normalized_contribution_share: assignment.normalized_contribution_share,
                measured_service_rate: assignment.measured_service_rate,
                reference_service_rate: assignment.reference_service_rate,
                memory_pressure: assignment.memory_pressure,
            })
            .collect(),
    })
}

fn load_ledger_events(db: &Database) -> Result<Vec<UiLedgerEvent>> {
    let conn = db.get_conn()?;
    let mut stmt = conn.prepare(
        r#"
        SELECT
            event_id,
            network_id,
            event_type,
            job_id,
            device_id,
            credits_amount,
            metadata,
            timestamp
        FROM ledger_events
        ORDER BY timestamp DESC, event_id DESC
        LIMIT 500
        "#,
    )?;

    let events = stmt
        .query_map([], |row| {
            let metadata_raw: String = row.get(6)?;
            let metadata = serde_json::from_str(&metadata_raw).unwrap_or(Value::Null);
            Ok(UiLedgerEvent {
                id: row.get(0)?,
                network_id: row.get(1)?,
                event_type: row.get(2)?,
                job_id: row.get(3)?,
                device_id: row.get(4)?,
                credits_amount: row.get(5)?,
                detail: ledger_detail(row.get::<_, String>(2)?.as_str(), &metadata),
                metadata,
                created_at: row.get(7)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(events)
}

async fn load_topologies(
    networks: &[UiNetwork],
    devices: &[UiDevice],
    client: &reqwest::Client,
    control_plane_url: Option<&str>,
) -> Vec<UiTopology> {
    let device_by_network: HashMap<&str, Vec<&UiDevice>> = {
        let mut grouped: HashMap<&str, Vec<&UiDevice>> = HashMap::new();
        for device in devices {
            grouped
                .entry(device.network_id.as_str())
                .or_default()
                .push(device);
        }
        grouped
    };

    let mut topologies = Vec::new();
    for network in networks {
        if let Some(base_url) = control_plane_url {
            let url = format!("{base_url}/api/ring/topology?network_id={}", network.id);
            if let Ok(response) = client.get(&url).send().await {
                if let Ok(topology) = response.error_for_status() {
                    if let Ok(body) = topology.json::<RingTopologyResponse>().await {
                        topologies.push(from_live_topology(network.id.clone(), body, devices));
                        continue;
                    }
                }
            }
        }

        let workers = device_by_network
            .get(network.id.as_str())
            .into_iter()
            .flat_map(|devices| devices.iter().copied())
            .filter(|device| device.ring_position.is_some())
            .map(|device| UiTopologyWorker {
                device_id: device.id.clone(),
                device_name: device.name.clone(),
                peer_id: device.peer_id.clone(),
                position: device.ring_position,
                status: device.status.clone(),
                contributed_memory_bytes: device.contributed_memory_bytes,
                shard_column_start: device.shard_column_start,
                shard_column_end: device.shard_column_end,
                left_neighbor_id: device.left_neighbor_id.clone(),
                right_neighbor_id: device.right_neighbor_id.clone(),
                active_path: device
                    .connectivity_state
                    .as_ref()
                    .map(|state| state.active_path.clone()),
                active_endpoint: device
                    .connectivity_state
                    .as_ref()
                    .and_then(|state| state.active_endpoint.clone()),
                tensor_plane_endpoints: device.tensor_plane_endpoints.clone(),
            })
            .collect::<Vec<_>>();

        topologies.push(UiTopology {
            network_id: network.id.clone(),
            source: "local_db".to_string(),
            ring_stable: !workers.is_empty(),
            workers,
            punch_plans: Vec::new(),
        });
    }

    topologies
}

fn from_live_topology(
    network_id: String,
    topology: RingTopologyResponse,
    devices: &[UiDevice],
) -> UiTopology {
    let devices_by_id: HashMap<&str, &UiDevice> = devices
        .iter()
        .map(|device| (device.id.as_str(), device))
        .collect();
    UiTopology {
        network_id,
        source: "control_plane".to_string(),
        ring_stable: topology.ring_stable,
        workers: topology
            .workers
            .into_iter()
            .map(|worker| {
                let device = devices_by_id.get(worker.device_id.as_str()).copied();
                UiTopologyWorker {
                    device_id: worker.device_id.clone(),
                    device_name: device
                        .map(|device| device.name.clone())
                        .unwrap_or_else(|| worker.device_id.clone()),
                    peer_id: Some(worker.peer_id),
                    position: Some(worker.position),
                    status: worker.status,
                    contributed_memory_bytes: Some(worker.contributed_memory),
                    shard_column_start: Some(worker.shard.column_start),
                    shard_column_end: Some(worker.shard.column_end),
                    left_neighbor_id: Some(worker.left_neighbor),
                    right_neighbor_id: Some(worker.right_neighbor),
                    active_path: worker
                        .connectivity_state
                        .as_ref()
                        .map(|state| format!("{:?}", state.active_path).to_ascii_lowercase()),
                    active_endpoint: worker
                        .connectivity_state
                        .and_then(|state| state.active_endpoint),
                    tensor_plane_endpoints: device
                        .map(|device| device.tensor_plane_endpoints.clone())
                        .unwrap_or_default(),
                }
            })
            .collect(),
        punch_plans: topology
            .peer_punch_plans
            .into_iter()
            .map(|plan| UiPunchPlan {
                source_device_id: plan.source_device_id,
                target_device_id: plan.target_device_id,
                target_peer_id: plan.target_peer_id,
                reason: format!("{:?}", plan.reason).to_ascii_lowercase(),
                strategy: format!("{:?}", plan.strategy).to_ascii_lowercase(),
                relay_rendezvous_required: plan.relay_rendezvous_required,
                attempt_window_ms: plan.attempt_window_ms,
                issued_at_ms: plan.issued_at_ms,
                target_candidates: plan
                    .target_candidates
                    .into_iter()
                    .filter_map(map_control_candidate)
                    .collect(),
            })
            .collect(),
    }
}

fn load_models(mesh_home: &Path, devices: &[UiDevice]) -> Result<Vec<UiModel>> {
    let models_dir = mesh_home.join("models");
    let registry_path = mesh_home.join("shards").join("registry.json");
    let shard_registry: HashMap<String, ShardRegistryEntry> = if registry_path.exists() {
        serde_json::from_str(&fs::read_to_string(&registry_path)?).unwrap_or_default()
    } else {
        HashMap::new()
    };

    let mut network_ids_by_model: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut participants_by_model: HashMap<String, Vec<&UiDevice>> = HashMap::new();
    for device in devices {
        if let Some(model_id) = &device.shard_model_id {
            network_ids_by_model
                .entry(model_id.clone())
                .or_default()
                .insert(device.network_id.clone());
            participants_by_model
                .entry(model_id.clone())
                .or_default()
                .push(device);
        }
    }

    let mut models = Vec::new();
    if models_dir.exists() {
        for entry in fs::read_dir(models_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let model_id = entry.file_name().to_string_lossy().to_string();
            let model_dir = entry.path();
            let manifest = fs::read_to_string(model_dir.join("model.json"))
                .ok()
                .and_then(|content| serde_json::from_str::<ModelManifest>(&content).ok());
            let manifest_count = count_matching_files(&model_dir, ".manifest.json")?;
            let weights_count = count_matching_files(&model_dir, ".safetensors")?;
            let tokenizer_ready = model_dir.join("tokenizer.json").exists()
                && model_dir.join("tokenizer_config.json").exists();
            let registry = shard_registry.get(&model_id);
            let participants = participants_by_model
                .get(&model_id)
                .cloned()
                .unwrap_or_default();
            let provider_compatibility = participants
                .iter()
                .flat_map(|device| {
                    device
                        .capabilities
                        .execution_providers
                        .iter()
                        .filter(|provider| provider.available)
                        .map(|provider| provider.kind.as_str().to_string())
                        .collect::<Vec<_>>()
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();

            models.push(UiModel {
                id: manifest
                    .as_ref()
                    .map(|manifest| manifest.model_id.clone())
                    .unwrap_or(model_id.clone()),
                network_ids: network_ids_by_model
                    .get(&model_id)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .collect(),
                total_model_bytes: manifest.as_ref().map(|manifest| manifest.total_model_bytes),
                tensor_parallelism_dim: manifest
                    .as_ref()
                    .map(|manifest| manifest.tensor_parallelism_dim),
                artifact_ready: manifest.is_some()
                    && tokenizer_ready
                    && manifest_count > 0
                    && weights_count > 0,
                tokenizer_ready,
                manifest_count,
                weights_count,
                participant_count: participants.len(),
                loaded_local_shard: registry.map(|entry| entry.info.is_loaded).unwrap_or(false),
                local_shard_range: registry.map(|entry| UiShardRange {
                    start: entry.info.assignment.column_start,
                    end: entry.info.assignment.column_end,
                }),
                local_memory_bytes: registry.map(|entry| entry.info.memory_bytes),
                shard_status: registry.map(|entry| entry.status.clone()),
                provider_compatibility,
            });
        }
    }

    models.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(models)
}

fn load_settings(
    mesh_home: &Path,
    local_device_config: &Option<DeviceConfig>,
    db_path: &Path,
) -> Result<UiSettings> {
    let relay_path = mesh_home.join("relay.toml");
    let relay = if relay_path.exists() {
        toml::from_str::<toml::Value>(&fs::read_to_string(&relay_path)?)
            .map(serde_json::to_value)?
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    };

    Ok(UiSettings {
        control_plane_url: local_device_config
            .as_ref()
            .map(|config| config.control_plane_url.clone()),
        local_device_name: local_device_config
            .as_ref()
            .map(|config| config.name.clone()),
        preferred_provider: local_device_config.as_ref().and_then(|config| {
            config
                .execution
                .preferred_provider
                .map(|provider| provider.as_str().to_string())
        }),
        governance: local_device_config
            .as_ref()
            .map(|config| serde_json::to_value(&config.governance))
            .transpose()?
            .unwrap_or(Value::Null),
        relay,
        config_paths: UiConfigPaths {
            device_config: DeviceConfig::default_path()?.display().to_string(),
            device_certificate: DeviceConfig::default_certificate_path()?
                .display()
                .to_string(),
            relay_config: relay_path.display().to_string(),
            control_plane_db: db_path.display().to_string(),
            shard_registry: mesh_home
                .join("shards")
                .join("registry.json")
                .display()
                .to_string(),
        },
    })
}

fn scheduling_policy(policy: InferenceSchedulingPolicy) -> UiSchedulingPolicy {
    UiSchedulingPolicy {
        submitter_active_job_soft_cap: policy.submitter_active_job_soft_cap,
        model_active_job_soft_cap_divisor: policy.model_active_job_soft_cap_divisor,
        capacity_unit_soft_cap_divisor: policy.capacity_unit_soft_cap_divisor,
        tier_capacity_units: UiTierCapacityUnits {
            tier0: policy.tier_capacity_units.tier0,
            tier1: policy.tier_capacity_units.tier1,
            tier2: policy.tier_capacity_units.tier2,
            tier3: policy.tier_capacity_units.tier3,
            tier4: policy.tier_capacity_units.tier4,
        },
    }
}

fn parse_optional_json<T>(raw: Option<String>) -> Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    match raw {
        Some(raw) => serde_json::from_str(&raw)
            .map(Some)
            .with_context(|| format!("Failed to parse JSON payload: {}", raw)),
        None => Ok(None),
    }
}

fn health_label(status: &str, connectivity: Option<&DeviceConnectivityState>) -> String {
    if status == "offline" {
        return "offline".to_string();
    }
    match connectivity.map(|state| format!("{:?}", state.status).to_ascii_lowercase()) {
        Some(value) if value == "degraded" => "degraded".to_string(),
        Some(value) if value == "disconnected" => "offline".to_string(),
        _ => "healthy".to_string(),
    }
}

fn ledger_detail(event_type: &str, metadata: &Value) -> String {
    let model = metadata
        .get("model_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown-model");
    match event_type {
        "job_started" => format!("Job started for {}", model),
        "job_completed" => format!("Job completed for {}", model),
        "job_cancelled" => format!("Job cancelled for {}", model),
        "credits_burned" => format!("Credits burned while serving {}", model),
        "credits_earned" => format!("Credits earned while serving {}", model),
        "credits_reserved" => format!("Credits reserved for {}", model),
        "credits_released" => format!("Credits released for {}", model),
        _ => format!("{} event", event_type.replace('_', " ")),
    }
}

fn count_matching_files(dir: &Path, suffix: &str) -> Result<usize> {
    Ok(fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(suffix))
        .count())
}

fn to_sql_error(error: anyhow::Error) -> rusqlite::Error {
    rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::other(error.to_string())))
}

fn map_control_candidate(
    candidate: control_plane::connectivity::DirectPeerCandidate,
) -> Option<DirectPeerCandidate> {
    serde_json::to_value(candidate)
        .ok()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn ok_response<T: Serialize>(data: T) -> Response {
    (StatusCode::OK, Json(ApiEnvelope { ok: true, data })).into_response()
}

fn error_response(
    status: StatusCode,
    code: &str,
    message: String,
    hint: Option<String>,
) -> Response {
    (
        status,
        Json(ApiErrorEnvelope {
            ok: false,
            error: ApiErrorBody {
                code: code.to_string(),
                message,
                hint,
            },
        }),
    )
        .into_response()
}

fn parse_includes(include: Option<&str>) -> HashSet<String> {
    include
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn load_local_device_config() -> Result<Option<DeviceConfig>> {
    match DeviceConfig::default_path() {
        Ok(path) if path.exists() => Ok(Some(DeviceConfig::load(&path)?)),
        Ok(_) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

async fn with_device_config<F, Fut>(f: F) -> Result<Value>
where
    F: FnOnce(DeviceConfig) -> Fut,
    Fut: std::future::Future<Output = Result<Value>>,
{
    let config = load_local_device_config()?
        .ok_or_else(|| anyhow!("Device is not initialized. Run device init first."))?;
    f(config).await
}

fn control_plane_url(config: &DeviceConfig, path: &str) -> String {
    format!("{}{}", config.control_plane_url.trim_end_matches('/'), path)
}

async fn proxy_json(
    client: &reqwest::Client,
    method: reqwest::Method,
    url: String,
    body: Option<Value>,
) -> Result<Value> {
    let request = client.request(method, url);
    let request = if let Some(body) = body {
        request.json(&body)
    } else {
        request
    };
    let response = request.send().await?.error_for_status()?;
    Ok(response.json::<Value>().await?)
}

async fn build_device_status(state: &UiState) -> Result<DeviceStatusResponse> {
    let config = load_local_device_config()?;
    let candidate_seed_records = agent::load_direct_candidate_seed_records().unwrap_or_default();
    let direct_candidates = if let Some(config) = config.as_ref() {
        let local_peer_id = agent::device::keypair::to_libp2p_keypair(&config.keypair)
            .public()
            .to_peer_id();
        agent::build_direct_peer_candidates_from_records(local_peer_id, &candidate_seed_records)
    } else {
        Vec::new()
    };
    let daemon_running = state.daemon.lock().await.is_some();
    Ok(DeviceStatusResponse {
        configured: config.is_some(),
        device_id: config.as_ref().map(|value| value.device_id.to_string()),
        network_id: config.as_ref().map(|value| value.network_id.clone()),
        name: config.as_ref().map(|value| value.name.clone()),
        control_plane_url: config.as_ref().map(|value| value.control_plane_url.clone()),
        preferred_provider: config.as_ref().and_then(|value| {
            value
                .execution
                .preferred_provider
                .map(|provider| provider.as_str().to_string())
        }),
        has_certificate: config
            .as_ref()
            .map(|value| value.has_certificate())
            .unwrap_or(false),
        daemon_running,
        listen_addrs: load_json_vec(&state.mesh_home.join("listen_addrs.json")).unwrap_or_default(),
        observed_addrs: load_json_vec(&state.mesh_home.join("observed_addrs.json"))
            .unwrap_or_default(),
        direct_candidate_count: direct_candidates.len(),
    })
}

fn load_resource_lock_status() -> Result<UiResourceLockStatus> {
    let mut manager = ResourceManager::new().context("Failed to initialize resource manager")?;
    manager
        .load_config()
        .context("Failed to load resource config")?;
    let unlock_in_seconds = manager.time_until_unlock().map(|value| value.as_secs());
    let lock_timestamp_ms = manager.lock_timestamp().and_then(|value| {
        value
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|duration| duration.as_millis() as u64)
    });
    Ok(UiResourceLockStatus {
        status: if manager.is_locked() {
            "locked".into()
        } else {
            "unlocked".into()
        },
        total_memory_bytes: manager.total_memory(),
        user_allocated_bytes: manager.user_allocated(),
        locked_memory_bytes: manager.locked_memory(),
        lock_timestamp_ms,
        ready_to_unlock: manager.is_locked() && unlock_in_seconds.unwrap_or(0) == 0,
        unlock_in_seconds,
    })
}

async fn load_ring_status(state: &UiState) -> Result<Value> {
    let topology = load_live_topology(state).await?;
    let config =
        load_local_device_config()?.ok_or_else(|| anyhow!("Device is not initialized."))?;
    let worker = topology
        .workers
        .iter()
        .find(|worker| worker.device_id == config.device_id.to_string())
        .cloned();
    Ok(serde_json::to_value(serde_json::json!({
        "networkId": config.network_id,
        "ringStable": topology.ring_stable,
        "worker": worker,
    }))?)
}

async fn load_live_topology(state: &UiState) -> Result<RingTopologyResponse> {
    let config =
        load_local_device_config()?.ok_or_else(|| anyhow!("Device is not initialized."))?;
    let url = control_plane_url(
        &config,
        &format!("/api/ring/topology?network_id={}", config.network_id),
    );
    let response = state.client.get(url).send().await?.error_for_status()?;
    Ok(response.json::<RingTopologyResponse>().await?)
}

async fn load_local_shards() -> Result<Value> {
    let registry = ShardRegistry::with_defaults()?;
    let _ = registry.load().await;
    let shards = registry.list_shards().await;
    Ok(serde_json::to_value(
        shards
            .into_iter()
            .map(|(model_id, info, status)| {
                serde_json::json!({
                    "modelId": model_id,
                    "status": format!("{:?}", status).to_ascii_lowercase(),
                    "columnStart": info.assignment.column_start,
                    "columnEnd": info.assignment.column_end,
                    "workerPosition": info.assignment.worker_position,
                    "totalWorkers": info.assignment.total_workers,
                    "memoryBytes": info.memory_bytes,
                    "downloadProgress": info.download_progress,
                })
            })
            .collect::<Vec<_>>(),
    )?)
}

fn load_runtime_stats() -> Result<Value> {
    let stats_path = dirs::home_dir()
        .ok_or_else(|| anyhow!("Could not determine home directory"))?
        .join(".meshnet")
        .join("inference_stats.json");
    if !stats_path.exists() {
        return Ok(serde_json::json!({}));
    }
    Ok(serde_json::from_str(&fs::read_to_string(stats_path)?)?)
}

fn load_pools() -> Result<Vec<UiPoolSummary>> {
    let pools = PoolConfig::list_pools()?;
    let summaries = pools
        .into_iter()
        .map(|(pool_id, config, cert)| {
            let peer_count = PeerCache::load(&pool_id)
                .map(|cache| cache.get_peers(&pool_id).len())
                .unwrap_or(0);
            UiPoolSummary {
                id: pool_id.to_hex(),
                name: config.name.clone(),
                role: match config.role {
                    MembershipRole::Admin => "admin".into(),
                    MembershipRole::Member => "member".into(),
                },
                created_at: config.created_at.clone(),
                expires_at: config.expires_at,
                days_until_expiry: config.days_until_expiry(),
                peer_count,
                root_pubkey_hex: hex::encode(config.pool_root_pubkey),
                valid_cert: config.is_cert_valid(&cert),
            }
        })
        .collect();
    Ok(summaries)
}

fn load_pool_peers(pool_id_hex: &str) -> Result<Vec<UiPoolPeer>> {
    let pool_id = PoolId::from_hex(pool_id_hex)?;
    let peer_cache = PeerCache::load(&pool_id)?;
    Ok(peer_cache
        .get_peers(&pool_id)
        .into_iter()
        .map(|peer| UiPoolPeer {
            node_id: peer.node_id.to_hex(),
            lan_addr: peer.lan_addr.clone(),
            discovery_method: format!("{:?}", peer.discovery_method).to_ascii_lowercase(),
            last_seen: peer.last_seen,
        })
        .collect())
}

fn load_pool_status(pool_id_hex: &str) -> Result<Value> {
    let pool_id = PoolId::from_hex(pool_id_hex)?;
    let (config, cert) = PoolConfig::load(&pool_id)?;
    let peers = load_pool_peers(pool_id_hex)?;
    Ok(serde_json::json!({
        "poolId": pool_id_hex,
        "name": config.name,
        "role": format!("{:?}", config.role).to_ascii_lowercase(),
        "createdAt": config.created_at,
        "expiresAt": config.expires_at,
        "daysUntilExpiry": config.days_until_expiry(),
        "validCert": config.is_cert_valid(&cert),
        "rootPubkeyHex": hex::encode(config.pool_root_pubkey),
        "peers": peers,
    }))
}

async fn build_doctor_report() -> Result<UiDoctorReport> {
    let generated_at = chrono::Utc::now().to_rfc3339();
    let config_path = DeviceConfig::default_path()?;
    let mut checks = Vec::new();

    let config_check_start = std::time::Instant::now();
    let config = DeviceConfig::load(&config_path).ok();
    checks.push(UiDoctorCheck {
        id: "device_config".into(),
        label: "Device config".into(),
        status: if config.is_some() {
            "ok".into()
        } else {
            "fail".into()
        },
        detail: if config.is_some() {
            format!("Loaded device config from {}", config_path.display())
        } else {
            format!("Missing device config at {}", config_path.display())
        },
        hint: if config.is_some() {
            None
        } else {
            Some("Run device init to generate local identity and registration.".into())
        },
        duration_ms: config_check_start.elapsed().as_millis() as u64,
    });

    let cert_start = std::time::Instant::now();
    let has_cert = config
        .as_ref()
        .map(|value| value.has_certificate())
        .unwrap_or(false);
    checks.push(UiDoctorCheck {
        id: "device_certificate".into(),
        label: "Device certificate".into(),
        status: if has_cert { "ok".into() } else { "fail".into() },
        detail: if has_cert {
            "Device certificate is present.".into()
        } else {
            "Device certificate is missing.".into()
        },
        hint: if has_cert {
            None
        } else {
            Some("Re-run device init against the control plane.".into())
        },
        duration_ms: cert_start.elapsed().as_millis() as u64,
    });

    let reachability_start = std::time::Instant::now();
    let control_plane_result = if let Some(config) = config.as_ref() {
        Some(
            reqwest::Client::new()
                .get(control_plane_url(
                    config,
                    &format!("/api/ring/topology?network_id={}", config.network_id),
                ))
                .send()
                .await,
        )
    } else {
        None
    };
    checks.push(UiDoctorCheck {
        id: "control_plane_reachable".into(),
        label: "Control plane".into(),
        status: match &control_plane_result {
            Some(Ok(_)) => "ok".into(),
            Some(Err(_)) => "fail".into(),
            None => "fail".into(),
        },
        detail: match control_plane_result {
            Some(Ok(ref response)) => {
                format!("Control plane reachable (HTTP {}).", response.status())
            }
            Some(Err(ref error)) => format!("Control plane unreachable: {}", error),
            None => {
                "Device config missing, so control-plane reachability could not be checked.".into()
            }
        },
        hint: if config.is_some() {
            match &control_plane_result {
                Some(Err(_)) => {
                    Some("Start the control plane or verify the configured URL.".into())
                }
                _ => None,
            }
        } else {
            Some("Run device init first.".into())
        },
        duration_ms: reachability_start.elapsed().as_millis() as u64,
    });

    let overall = if checks.iter().any(|check| check.status == "fail") {
        "fail"
    } else if checks.iter().any(|check| check.status == "warn") {
        "warn"
    } else {
        "ok"
    };

    Ok(UiDoctorReport {
        generated_at,
        overall: overall.into(),
        checks,
    })
}

async fn build_quote_response(
    state: &UiState,
    config: &DeviceConfig,
    model_id: &str,
    query: QuoteQuery,
) -> Result<QuoteResponse> {
    let manifest = model_assets::load_model_manifest(model_id)?;
    let quote = quote_consumption(ConsumptionQuoteInput {
        prompt_tokens: query.prompt_tokens,
        requested_completion_tokens: query.max_tokens,
        total_model_bytes: manifest.total_model_bytes,
    });

    let network_id = query
        .network_id
        .unwrap_or_else(|| config.network_id.clone());
    let url = control_plane_url(
        config,
        &format!("/api/ledger/summary?network_id={network_id}"),
    );
    let ledger_summary = state
        .client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json::<LedgerSummaryPayload>()
        .await?;
    let device_available_credits =
        ledger_summary.total_credits_earned - ledger_summary.total_credits_burned;
    let feasible = device_available_credits + f64::EPSILON >= quote.total_credits;

    Ok(QuoteResponse {
        model_id: model_id.to_string(),
        network_id,
        model_size_factor: quote.model_size_factor,
        prompt_tokens: query.prompt_tokens,
        max_tokens: query.max_tokens,
        prompt_credits: quote.prompt_credits,
        completion_credits_cap: quote.completion_credits,
        total_credits_cap: quote.total_credits,
        available_completion_tokens: query.max_tokens,
        device_available_credits,
        feasible,
        reason: if feasible {
            None
        } else {
            Some("Available credits are below the requested reservation cap.".into())
        },
    })
}

fn load_json_vec(path: &Path) -> Option<Vec<String>> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn available_memory_bytes(device: &UiDevice) -> u64 {
    if let Some(bytes) = device.contributed_memory_bytes {
        return bytes.max(1);
    }
    (((device.capabilities.ram_mb + device.capabilities.gpu_vram_mb.unwrap_or_default()) as u64)
        * 1024
        * 1024)
        .max(1)
}

async fn kill_tracked_process(slot: &Arc<Mutex<Option<Child>>>) {
    let mut child = slot.lock().await;
    if let Some(process) = child.as_mut() {
        let _ = process.kill().await;
    }
    *child = None;
}

async fn ensure_local_control_plane_running(state: &UiState) -> Result<()> {
    let config = load_local_device_config()?;
    let url = config
        .as_ref()
        .map(|value| value.control_plane_url.as_str())
        .unwrap_or("http://127.0.0.1:8080");
    ensure_control_plane_running_for_url(state, url).await
}

async fn ensure_control_plane_running_for_url(state: &UiState, url: &str) -> Result<()> {
    let Some(port) = local_control_plane_port(url) else {
        return Ok(());
    };

    if control_plane_healthy(&state.client, url).await {
        return Ok(());
    }

    {
        let mut tracked = state.control_plane.lock().await;
        if let Some(child) = tracked.as_mut() {
            match child.try_wait() {
                Ok(Some(_)) => {
                    *tracked = None;
                }
                Ok(None) => {}
                Err(_) => {
                    *tracked = None;
                }
            }
        }

        if tracked.is_none() {
            *tracked = Some(spawn_control_plane_child(port)?);
        }
    }

    for _ in 0..40 {
        if control_plane_healthy(&state.client, url).await {
            return Ok(());
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
    }

    Err(anyhow!(
        "Local control plane did not become healthy at {} after startup",
        url
    ))
}

fn local_control_plane_port(url: &str) -> Option<u16> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let host = parsed.host_str()?;
    if !matches!(host, "127.0.0.1" | "localhost") {
        return None;
    }
    Some(parsed.port_or_known_default().unwrap_or(8080))
}

async fn control_plane_healthy(client: &reqwest::Client, base_url: &str) -> bool {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    match client.get(url).send().await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

fn spawn_control_plane_child(port: u16) -> Result<Child> {
    let mut attempts: Vec<(PathBuf, Vec<String>, Option<PathBuf>)> = Vec::new();
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(dir) = current_exe.parent() {
            attempts.push((
                dir.join("mesh-control-plane"),
                vec!["--port".into(), port.to_string()],
                None,
            ));
            attempts.push((
                dir.join("control-plane"),
                vec!["--port".into(), port.to_string()],
                None,
            ));
        }
    }

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow!("Could not resolve workspace root"))?
        .to_path_buf();
    attempts.push((
        workspace_root
            .join("target")
            .join("debug")
            .join("control-plane"),
        vec!["--port".into(), port.to_string()],
        None,
    ));
    attempts.push((
        workspace_root
            .join("target")
            .join("release")
            .join("control-plane"),
        vec!["--port".into(), port.to_string()],
        None,
    ));

    if let Ok(entries) = fs::read_dir(workspace_root.join("target")) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            attempts.push((
                path.join("debug").join("control-plane"),
                vec!["--port".into(), port.to_string()],
                None,
            ));
            attempts.push((
                path.join("release").join("control-plane"),
                vec!["--port".into(), port.to_string()],
                None,
            ));
        }
    }

    attempts.push((
        PathBuf::from("mesh-control-plane"),
        vec!["--port".into(), port.to_string()],
        None,
    ));
    attempts.push((
        PathBuf::from("control-plane"),
        vec!["--port".into(), port.to_string()],
        None,
    ));
    attempts.push((
        PathBuf::from("cargo"),
        vec![
            "run".into(),
            "-p".into(),
            "control-plane".into(),
            "--".into(),
            "--port".into(),
            port.to_string(),
        ],
        Some(workspace_root),
    ));

    let mut errors = Vec::new();
    for (program, args, cwd) in attempts {
        let mut command = tokio::process::Command::new(&program);
        command.kill_on_drop(true);
        if let Some(cwd) = cwd.as_ref() {
            command.current_dir(cwd);
        }
        for arg in &args {
            command.arg(arg);
        }
        match command.spawn() {
            Ok(child) => return Ok(child),
            Err(error) => errors.push(format!("{}: {}", program.display(), error)),
        }
    }

    Err(anyhow!(
        "Failed to spawn control plane. Tried: {}",
        errors.join(" | ")
    ))
}
