use crate::db::Database;
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
        match tokio::task::spawn_blocking(move || mark_offline_devices(&db_clone)).await {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => error!(error = %e, "Failed to mark offline devices"),
            Err(e) => error!(error = %e, "Task join error"),
        }
    }
}

/// Mark devices as offline if last_seen > 20 seconds ago
fn mark_offline_devices(db: &Database) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let threshold = OffsetDateTime::now_utc() - time::Duration::seconds(20);
    let threshold_str = threshold
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })?;

    let conn = db.get_conn()?;

    let affected = conn.execute(
        r#"
        UPDATE devices
        SET status = 'offline'
        WHERE status = 'online'
          AND (last_seen IS NULL OR datetime(last_seen) < datetime(?))
        "#,
        params![&threshold_str],
    )?;

    if affected > 0 {
        debug!(count = affected, "Marked devices offline");
    }

    Ok(affected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::create_test_db;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::services::certificate::ControlPlaneKeypair;
    use crate::services::device_service::register_device;

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            cpu_cores: 4,
            ram_mb: 8192,
            os: "linux".into(),
            arch: "x86_64".into(),
            has_gpu: false,
            tier: Tier::Tier1,
        }
    }

    #[test]
    fn test_mark_offline_devices() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();

        // Register a device
        let device_id = "test-device";
        register_device(
            &db,
            &keypair,
            device_id.to_string(),
            "test-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
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
        assert_eq!(affected, 1);

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

        // Register a device
        let device_id = "test-device-recent";
        register_device(
            &db,
            &keypair,
            device_id.to_string(),
            "test-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
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
        assert_eq!(affected, 0); // Should not mark this device offline

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
