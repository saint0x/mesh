use agent::{
    DeviceCapabilities, DeviceConfig, DeviceConnectivityState, DirectPeerCandidate,
    ExecutionProviderInfo,
};
use anyhow::{anyhow, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use control_plane::{
    api::types::RingTopologyResponse, connectivity::InferenceSchedulingPolicy, Database,
};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::Command,
};
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct UiState {
    db: Database,
    mesh_home: PathBuf,
    local_device_config: Option<DeviceConfig>,
    client: reqwest::Client,
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
    execution_time_ms: u64,
    error: Option<String>,
    assignments: Vec<UiAssignment>,
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

pub async fn cmd_ui(port: u16, api_port: u16) -> Result<()> {
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

    let device_config = DeviceConfig::default_path()
        .ok()
        .and_then(|path| DeviceConfig::load(&path).ok());

    ensure_mesh_ui_dependencies(&ui_dir)?;
    build_mesh_ui(&ui_dir, api_port)?;

    let api_state = UiState {
        db,
        mesh_home,
        local_device_config: device_config,
        client: reqwest::Client::new(),
    };

    let api_addr = SocketAddr::from(([127, 0, 0, 1], api_port));
    let listener = tokio::net::TcpListener::bind(api_addr)
        .await
        .with_context(|| format!("Failed to bind Mesh UI API on {}", api_addr))?;
    let api_router = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/api/local/dashboard", get(get_dashboard))
        .with_state(api_state)
        .layer(CorsLayer::permissive());
    let api_task = tokio::spawn(async move {
        axum::serve(listener, api_router).await
    });

    let ui_url = format!("http://127.0.0.1:{port}");
    let api_url = format!("http://127.0.0.1:{api_port}");
    println!("Mesh UI API: {api_url}");
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
        }
    }

    Ok(())
}

async fn get_dashboard(State(state): State<UiState>) -> impl IntoResponse {
    match load_dashboard_snapshot(&state).await {
        Ok(snapshot) => (StatusCode::OK, Json(snapshot)).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": error.to_string() })),
        )
            .into_response(),
    }
}

async fn load_dashboard_snapshot(state: &UiState) -> Result<DashboardSnapshot> {
    let networks = load_networks(&state.db)?;
    let devices = load_devices(&state.db, state.local_device_config.as_ref())?;
    let jobs = load_jobs(&state.db, &devices, &networks)?;
    let ledger_events = load_ledger_events(&state.db)?;
    let topologies = load_topologies(
        &networks,
        &devices,
        &state.client,
        state
            .local_device_config
            .as_ref()
            .map(|config| config.control_plane_url.as_str()),
    )
    .await;
    let models = load_models(&state.mesh_home, &devices)?;
    let settings = load_settings(&state.mesh_home, &state.local_device_config, &Database::default_path()?)?;

    Ok(DashboardSnapshot {
        generated_at: chrono::Utc::now().to_rfc3339(),
        mesh_home: state.mesh_home.display().to_string(),
        local_device_id: state
            .local_device_config
            .as_ref()
            .map(|config| config.device_id.to_string()),
        networks,
        devices,
        models,
        jobs,
        ledger_events,
        topologies,
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
        .env("VITE_MESH_UI_API_BASE", format!("http://127.0.0.1:{api_port}"))
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
            preferred_path: format!("{:?}", record.connectivity.preferred_path).to_ascii_lowercase(),
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

fn load_devices(db: &Database, local_device_config: Option<&DeviceConfig>) -> Result<Vec<UiDevice>> {
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
        let capabilities: DeviceCapabilities = serde_json::from_str(&capabilities_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                capabilities_json.len(),
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
        let connectivity_state = parse_optional_json::<DeviceConnectivityState>(row.get::<_, Option<String>>(13)?)
            .map_err(to_sql_error)?;
        let listen_addrs = parse_optional_json::<Vec<String>>(row.get::<_, Option<String>>(14)?)
            .map_err(to_sql_error)?
            .unwrap_or_default();
        let direct_candidates =
            parse_optional_json::<Vec<DirectPeerCandidate>>(row.get::<_, Option<String>>(15)?)
                .map_err(to_sql_error)?
                .unwrap_or_default();
        let certificate: Option<Vec<u8>> = row.get(17)?;

        Ok(UiDevice {
            id: row.get(0)?,
            network_id: row.get(1)?,
            name: row.get(2)?,
            peer_id: row.get(3)?,
            status: row.get(4)?,
            health: health_label(row.get::<_, String>(4)?.as_str(), connectivity_state.as_ref()),
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
                default_execution_provider: capabilities.default_execution_provider.as_str().to_string(),
            },
            certificate_status: if certificate.is_some() {
                "present".to_string()
            } else {
                "missing".to_string()
            },
            identity_status: "configured".to_string(),
            local_device: local_device_id
                .as_ref()
                .map(|device_id| device_id == &row.get::<_, String>(0).unwrap_or_default())
                .unwrap_or(false),
        })
    })?;

    let mut devices = Vec::new();
    for row in rows {
        devices.push(row?);
    }
    Ok(devices)
}

fn load_jobs(db: &Database, devices: &[UiDevice], networks: &[UiNetwork]) -> Result<Vec<UiJob>> {
    let conn = db.get_conn()?;
    let device_by_id: HashMap<&str, &UiDevice> = devices.iter().map(|device| (device.id.as_str(), device)).collect();
    let network_by_id: HashMap<&str, &UiNetwork> =
        networks.iter().map(|network| (network.id.as_str(), network)).collect();

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
            execution_time_ms,
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
            let model_id: String = row.get(3)?;
            let status: String = row.get(4)?;
            let ring_worker_count = row.get::<_, i64>(5)? as u32;
            let assignments = load_assignments(
                &conn,
                &id,
                &network_id,
                &device_by_id,
                network_by_id.get(network_id.as_str()).map(|network| &network.scheduling_policy),
            )
            .map_err(to_sql_error)?;

            Ok(UiJob {
                id,
                network_id,
                model_id,
                status,
                submitted_by_name: device_by_id
                    .get(submitted_by_device_id.as_str())
                    .map(|device| device.name.clone())
                    .unwrap_or_else(|| submitted_by_device_id.clone()),
                submitted_by_device_id,
                ring_worker_count,
                created_at: row.get(6)?,
                started_at: row.get(7)?,
                completed_at: row.get(8)?,
                completion_tokens: row.get::<_, i64>(9)? as u32,
                execution_time_ms: row.get::<_, i64>(10)? as u64,
                error: row.get(11)?,
                assignments,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(jobs)
}

fn load_assignments(
    conn: &rusqlite::Connection,
    job_id: &str,
    network_id: &str,
    device_by_id: &HashMap<&str, &UiDevice>,
    policy: Option<&UiSchedulingPolicy>,
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
            execution_time_ms
        FROM inference_job_assignments
        WHERE job_id = ?
        ORDER BY ring_position ASC, assignment_id ASC
        "#,
    )?;

    let assignments = stmt
        .query_map([job_id], |row| {
            let device_id: String = row.get(1)?;
            let device = device_by_id.get(device_id.as_str()).copied();
            let assigned_capacity_units = device
                .and_then(|device| policy.map(|policy| capacity_units(policy, device.capabilities.tier.as_str())))
                .unwrap_or(1);

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
                shard_column_start: device.and_then(|device| device.shard_column_start),
                shard_column_end: device.and_then(|device| device.shard_column_end),
                assigned_capacity_units,
                execution_provider: device.map(|device| device.capabilities.default_execution_provider.clone()),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let _ = network_id;
    Ok(assignments)
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
            grouped.entry(device.network_id.as_str()).or_default().push(device);
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

fn from_live_topology(network_id: String, topology: RingTopologyResponse, devices: &[UiDevice]) -> UiTopology {
    let devices_by_id: HashMap<&str, &UiDevice> = devices.iter().map(|device| (device.id.as_str(), device)).collect();
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
                    active_endpoint: worker.connectivity_state.and_then(|state| state.active_endpoint),
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
            let tokenizer_ready =
                model_dir.join("tokenizer.json").exists() && model_dir.join("tokenizer_config.json").exists();
            let registry = shard_registry.get(&model_id);
            let participants = participants_by_model.get(&model_id).cloned().unwrap_or_default();
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
                tensor_parallelism_dim: manifest.as_ref().map(|manifest| manifest.tensor_parallelism_dim),
                artifact_ready: manifest.is_some() && tokenizer_ready && manifest_count > 0 && weights_count > 0,
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
        local_device_name: local_device_config.as_ref().map(|config| config.name.clone()),
        preferred_provider: local_device_config
            .as_ref()
            .and_then(|config| config.execution.preferred_provider.map(|provider| provider.as_str().to_string())),
        governance: local_device_config
            .as_ref()
            .map(|config| serde_json::to_value(&config.governance))
            .transpose()?
            .unwrap_or(Value::Null),
        relay,
        config_paths: UiConfigPaths {
            device_config: DeviceConfig::default_path()?.display().to_string(),
            device_certificate: DeviceConfig::default_certificate_path()?.display().to_string(),
            relay_config: relay_path.display().to_string(),
            control_plane_db: db_path.display().to_string(),
            shard_registry: mesh_home.join("shards").join("registry.json").display().to_string(),
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

fn capacity_units(policy: &UiSchedulingPolicy, tier: &str) -> u32 {
    match tier {
        "Tier0" => policy.tier_capacity_units.tier0,
        "Tier1" => policy.tier_capacity_units.tier1,
        "Tier2" => policy.tier_capacity_units.tier2,
        "Tier3" => policy.tier_capacity_units.tier3,
        "Tier4" => policy.tier_capacity_units.tier4,
        _ => 1,
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
        "credits_burned" => format!("Credits burned while serving {}", model),
        "credits_earned" => format!("Credits earned while serving {}", model),
        _ => format!("{} event", event_type.replace('_', " ")),
    }
}

fn count_matching_files(dir: &Path, suffix: &str) -> Result<usize> {
    Ok(fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(suffix)
        })
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
