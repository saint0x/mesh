use crate::api::types::{DeviceMemoryPressureLevel, DeviceMemoryTelemetry};
use crate::resource_manager::ResourceManager;
use std::path::PathBuf;
use sysinfo::{Pid, System};

fn runtime_memory_snapshot_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".meshnet")
        .join("device_memory_telemetry.json")
}

pub fn persist_runtime_memory_telemetry(snapshot: &DeviceMemoryTelemetry) -> std::io::Result<()> {
    let path = runtime_memory_snapshot_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(snapshot).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

pub fn load_runtime_memory_telemetry() -> Option<DeviceMemoryTelemetry> {
    let path = runtime_memory_snapshot_path();
    let contents = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn committed_mesh_memory_bytes() -> Option<u64> {
    let mut manager = ResourceManager::new().ok()?;
    manager.load_config().ok()?;
    if manager.is_locked() {
        Some(manager.locked_memory())
    } else {
        None
    }
}

fn pressure_level(score: f64) -> DeviceMemoryPressureLevel {
    if score >= 0.92 {
        DeviceMemoryPressureLevel::Critical
    } else if score >= 0.80 {
        DeviceMemoryPressureLevel::Hot
    } else if score >= 0.65 {
        DeviceMemoryPressureLevel::Warm
    } else {
        DeviceMemoryPressureLevel::Healthy
    }
}

fn compute_pressure_score(snapshot: &DeviceMemoryTelemetry) -> f64 {
    let system_ratio = if snapshot.total_system_memory_bytes == 0 {
        0.0
    } else {
        snapshot.used_system_memory_bytes as f64 / snapshot.total_system_memory_bytes as f64
    };
    let mesh_ratio = match (
        snapshot.mesh_committed_memory_bytes,
        snapshot.mesh_available_memory_bytes,
    ) {
        (Some(committed), Some(available)) if committed > 0 => {
            1.0 - (available as f64 / committed as f64)
        }
        _ => 0.0,
    };
    let runtime_ratio = match (
        snapshot.runtime_total_runtime_bytes,
        snapshot.runtime_max_total_runtime_bytes,
    ) {
        (Some(used), Some(limit)) if limit > 0 => used as f64 / limit as f64,
        _ => 0.0,
    };
    let kv_ratio = match (
        snapshot.runtime_live_kv_cache_bytes,
        snapshot.runtime_max_total_kv_cache_bytes,
    ) {
        (Some(used), Some(limit)) if limit > 0 => used as f64 / limit as f64,
        _ => 0.0,
    };

    system_ratio.max(mesh_ratio).max(runtime_ratio).max(kv_ratio).clamp(0.0, 1.0)
}

pub fn sample_device_memory_telemetry() -> DeviceMemoryTelemetry {
    let mut system = System::new_all();
    system.refresh_all();

    let total_system_memory_bytes = system.total_memory();
    let available_system_memory_bytes = system.available_memory();
    let used_system_memory_bytes = total_system_memory_bytes.saturating_sub(available_system_memory_bytes);

    let process = system.process(Pid::from_u32(std::process::id()));
    let process_resident_memory_bytes = process.map(|item| item.memory());
    let process_virtual_memory_bytes = process.map(|item| item.virtual_memory());

    let mut snapshot = load_runtime_memory_telemetry().unwrap_or(DeviceMemoryTelemetry {
        observed_at: chrono::Utc::now().to_rfc3339(),
        total_system_memory_bytes,
        available_system_memory_bytes,
        used_system_memory_bytes,
        process_resident_memory_bytes,
        process_virtual_memory_bytes,
        mesh_committed_memory_bytes: committed_mesh_memory_bytes(),
        mesh_available_memory_bytes: None,
        runtime_active_sessions: None,
        runtime_total_runtime_bytes: None,
        runtime_live_kv_cache_bytes: None,
        runtime_model_resident_bytes: None,
        runtime_logical_kv_tokens: None,
        runtime_max_total_runtime_bytes: None,
        runtime_max_total_kv_cache_bytes: None,
        tensor_inbound_queued_bytes: None,
        tensor_outbound_inflight_bytes: None,
        pressure_score: 0.0,
        pressure_level: DeviceMemoryPressureLevel::Healthy,
    });

    snapshot.observed_at = chrono::Utc::now().to_rfc3339();
    snapshot.total_system_memory_bytes = total_system_memory_bytes;
    snapshot.available_system_memory_bytes = available_system_memory_bytes;
    snapshot.used_system_memory_bytes = used_system_memory_bytes;
    snapshot.process_resident_memory_bytes = process_resident_memory_bytes;
    snapshot.process_virtual_memory_bytes = process_virtual_memory_bytes;
    snapshot.mesh_committed_memory_bytes = snapshot
        .mesh_committed_memory_bytes
        .or_else(committed_mesh_memory_bytes);

    if let Some(committed) = snapshot.mesh_committed_memory_bytes {
        let used_by_mesh = snapshot
            .runtime_total_runtime_bytes
            .or(snapshot.process_resident_memory_bytes)
            .unwrap_or(0);
        snapshot.mesh_available_memory_bytes = Some(committed.saturating_sub(used_by_mesh));
    }

    snapshot.pressure_score = compute_pressure_score(&snapshot);
    snapshot.pressure_level = pressure_level(snapshot.pressure_score);
    snapshot
}
