use crate::connectivity::DeviceConnectivityState;
use crate::db::Database;
use crate::services::failover::reconcile_failover_state;
use rusqlite::params;
use time::OffsetDateTime;
use tokio::time::{interval, Duration};
use tracing::{debug, error, info};

/// Background task that marks devices offline after 20 seconds of inactivity
pub async fn presence_monitor(db: Database) {
    info!("Starting presence monitor task");

    let mut tick = interval(Duration::from_secs(10));

    loop {
        tick.tick().await;

        let db_clone = db.clone();
        match tokio::task::spawn_blocking(move || {
            let stale_devices = mark_offline_devices(&db_clone)?;
            reconcile_failover_state(&db_clone, &stale_devices).map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(std::io::Error::other(e.to_string()))
                },
            )?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(stale_devices.len())
        })
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => error!(error = %e, "Failed to mark offline devices"),
            Err(e) => error!(error = %e, "Task join error"),
        }
    }
}

/// Mark devices as offline if last_seen > 20 seconds ago
fn mark_offline_devices(
    db: &Database,
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    let threshold = OffsetDateTime::now_utc() - time::Duration::seconds(20);
    let threshold_str = threshold
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let conn = db.get_conn()?;

    let mut stale_stmt = conn.prepare(
        r#"
        SELECT device_id, connectivity_state
        FROM devices
        WHERE status = 'online'
          AND (last_seen IS NULL OR datetime(last_seen) < datetime(?))
        "#,
    )?;

    let stale_devices = stale_stmt
        .query_map(params![&threshold_str], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    for (device_id, connectivity_state_json) in &stale_devices {
        let connectivity_state = connectivity_state_json
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
            params![connectivity_state_json, device_id],
        )?;
    }

    let affected = stale_devices.len();
    let affected_ids = stale_devices
        .iter()
        .map(|(device_id, _)| device_id.clone())
        .collect::<Vec<_>>();

    if affected > 0 {
        debug!(count = affected, "Marked devices offline");
    }

    Ok(affected_ids)
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
    use crate::provider::{ExecutionProviderInfo, ExecutionProviderKind};
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
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Metal,
                    available: false,
                    reason: Some("metal provider is only available on macOS".into()),
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cuda,
                    available: true,
                    reason: None,
                },
            ],
            default_execution_provider: ExecutionProviderKind::Cuda,
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
        assert_eq!(affected, vec![device_id.to_string()]);

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

        // Set last_seen to 10 seconds ago (recent)
        let recent_timestamp = (OffsetDateTime::now_utc() - time::Duration::seconds(10))
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
