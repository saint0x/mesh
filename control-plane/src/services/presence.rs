use crate::connectivity::DeviceConnectivityState;
use crate::db::Database;
use crate::services::failover::reconcile_failover_state;
use crate::services::ring_manager::RingTopologyManager;
use rusqlite::params;
use std::collections::HashMap;
use std::sync::Arc;
use time::OffsetDateTime;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info};

fn presence_poll_interval() -> Duration {
    std::env::var("MESHNET_PRESENCE_POLL_INTERVAL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_millis(250))
}

fn stale_device_timeout() -> time::Duration {
    std::env::var("MESHNET_PRESENCE_STALE_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .map(time::Duration::milliseconds)
        .unwrap_or_else(|| time::Duration::seconds(5))
}

#[derive(Debug, Clone)]
struct StaleDeviceRecord {
    device_id: String,
    network_id: String,
    connectivity_state_json: Option<String>,
    in_ring: bool,
}

/// Background task that marks devices offline shortly after they stop heartbeating.
pub async fn presence_monitor(db: Database) {
    info!("Starting presence monitor task");

    let mut tick = interval(presence_poll_interval());

    loop {
        tick.tick().await;

        let db_clone = db.clone();
        match tokio::task::spawn_blocking(move || {
            let stale_devices = mark_offline_devices(&db_clone)?;
            let stale_count = stale_devices.len();
            if !stale_devices.is_empty() {
                let mut managers_by_network = HashMap::<String, RingTopologyManager>::new();
                let mut direct_failover = Vec::new();

                for device in stale_devices {
                    if device.in_ring {
                        let manager = managers_by_network
                            .entry(device.network_id.clone())
                            .or_insert_with(|| {
                                RingTopologyManager::new(Arc::new(db_clone.clone()))
                            });
                        manager.load_from_db(&device.network_id).map_err(
                            |e| -> Box<dyn std::error::Error + Send + Sync> {
                                Box::new(std::io::Error::other(e.to_string()))
                            },
                        )?;
                        manager
                            .handle_worker_failure(device.device_id.clone())
                            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                Box::new(std::io::Error::other(e.to_string()))
                            })?;
                    } else {
                        mark_device_offline(&db_clone, &device)?;
                        direct_failover.push(device.device_id);
                    }
                }

                if !direct_failover.is_empty() {
                    reconcile_failover_state(&db_clone, &direct_failover).map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            Box::new(std::io::Error::other(e.to_string()))
                        },
                    )?;
                }
            }
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(stale_count)
        })
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => error!(error = %e, "Failed to mark offline devices"),
            Err(e) => error!(error = %e, "Task join error"),
        }
    }
}

/// Mark devices as offline if last_seen is older than the configured timeout.
fn mark_offline_devices(
    db: &Database,
) -> Result<Vec<StaleDeviceRecord>, Box<dyn std::error::Error + Send + Sync>> {
    let threshold = OffsetDateTime::now_utc() - stale_device_timeout();
    let threshold_str = threshold
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let conn = db.get_conn()?;

    let mut stale_stmt = conn.prepare(
        r#"
        SELECT device_id, network_id, connectivity_state, ring_position
        FROM devices
        WHERE status = 'online'
          AND (last_seen IS NULL OR datetime(last_seen) < datetime(?))
        "#,
    )?;

    let stale_devices = stale_stmt
        .query_map(params![&threshold_str], |row| {
            Ok(StaleDeviceRecord {
                device_id: row.get(0)?,
                network_id: row.get(1)?,
                connectivity_state_json: row.get(2)?,
                in_ring: row.get::<_, Option<i64>>(3)?.is_some(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    let affected = stale_devices.len();
    if affected > 0 {
        debug!(count = affected, "Marked devices offline");
    }

    Ok(stale_devices)
}

fn mark_device_offline(
    db: &Database,
    device: &StaleDeviceRecord,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let conn = db.get_conn()?;
    let connectivity_state = device
        .connectivity_state_json
        .as_ref()
        .and_then(|json| serde_json::from_str::<DeviceConnectivityState>(json).ok())
        .map(|state| state.disconnected());
    let connectivity_state_json = connectivity_state
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;

    conn.execute(
        r#"
        UPDATE devices
        SET status = 'offline', connectivity_state = ?, listen_addrs = '[]'
        WHERE device_id = ?
        "#,
        params![connectivity_state_json, &device.device_id],
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::{
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath,
        InferenceSchedulingPolicy, NetworkConnectivity,
    };
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::provider::{
        BackendContractDescriptor, ExecutionProviderInfo, ExecutionProviderKind, MemoryModel,
    };
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::{device_service::register_device, network_service};

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            tier: Tier::Tier1,
            cpu_cores: 4,
            ram_mb: 8192,
            gpu_present: false,
            gpu_vram_mb: None,
            os: "linux".into(),
            arch: "x86_64".into(),
            execution_providers: vec![
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cpu,
                    available: true,
                    reason: None,
                    contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu),
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Metal,
                    available: false,
                    reason: Some("metal provider is only available on macOS".into()),
                    contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Metal),
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cuda,
                    available: true,
                    reason: None,
                    contract: BackendContractDescriptor::for_provider(ExecutionProviderKind::Cuda),
                },
            ],
            default_execution_provider: ExecutionProviderKind::Cuda,
            provider_contracts: vec![
                BackendContractDescriptor::for_provider(ExecutionProviderKind::Cpu),
                BackendContractDescriptor::for_provider(ExecutionProviderKind::Metal),
                BackendContractDescriptor::for_provider(ExecutionProviderKind::Cuda),
            ],
            default_provider_contract_hash: BackendContractDescriptor::for_provider(
                ExecutionProviderKind::Cuda,
            )
            .contract_hash,
            memory_model: MemoryModel::DiscreteVram,
        }
    }

    fn test_connectivity() -> NetworkConnectivity {
        NetworkConnectivity {
            preferred_path: ConnectivityPath::Relayed,
            attachments: vec![ConnectivityAttachment {
                kind: ConnectivityAttachmentKind::Libp2pRelay,
                endpoint: "/dns4/relay.mesh.example/tcp/4001".to_string(),
                priority: 0,
            }],
        }
    }

    #[test]
    fn test_mark_offline_devices() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "test-network".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
        )
        .unwrap();

        // Register a device
        let device_id = "test-device";
        register_device(
            &db,
            &keypair,
            device_id.to_string(),
            "test-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
            "12D3KooWQ6presence1111111111111111111111111111111".to_string(),
            test_capabilities(),
        )
        .unwrap();

        // Device should be online
        let conn = db.get_conn().unwrap();
        let status: String = conn
            .query_row(
                "SELECT status FROM devices WHERE device_id = ?",
                params![device_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "online");

        // Set last_seen to 30 seconds ago (old)
        let old_timestamp = (OffsetDateTime::now_utc() - time::Duration::seconds(30))
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap();

        conn.execute(
            "UPDATE devices SET last_seen = ? WHERE device_id = ?",
            params![&old_timestamp, device_id],
        )
        .unwrap();

        // Mark offline devices
        let affected = mark_offline_devices(&db).unwrap();
        assert_eq!(affected.len(), 1);
        assert_eq!(affected[0].device_id, device_id.to_string());
        mark_device_offline(&db, &affected[0]).unwrap();

        // Device should now be offline
        let status: String = conn
            .query_row(
                "SELECT status FROM devices WHERE device_id = ?",
                params![device_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "offline");
    }

    #[test]
    fn test_recent_heartbeat_stays_online() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "test-network".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
            InferenceSchedulingPolicy::default(),
        )
        .unwrap();

        // Register a device
        let device_id = "test-device-recent";
        register_device(
            &db,
            &keypair,
            device_id.to_string(),
            "test-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
            "12D3KooWQ6presence2222222222222222222222222222222".to_string(),
            test_capabilities(),
        )
        .unwrap();

        // Set last_seen to 1 second ago so it remains inside the tighter failover window.
        let recent_timestamp = (OffsetDateTime::now_utc() - time::Duration::seconds(1))
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap();

        let conn = db.get_conn().unwrap();
        conn.execute(
            "UPDATE devices SET last_seen = ? WHERE device_id = ?",
            params![&recent_timestamp, device_id],
        )
        .unwrap();

        // Mark offline devices
        let affected = mark_offline_devices(&db).unwrap();
        assert!(affected.is_empty()); // Should not mark this device offline

        // Device should still be online
        let status: String = conn
            .query_row(
                "SELECT status FROM devices WHERE device_id = ?",
                params![device_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "online");
    }
}
