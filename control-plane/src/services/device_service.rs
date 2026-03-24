use crate::api::error::{ApiError, ApiResult};
use crate::connectivity::NetworkConnectivity;
use crate::db::Database;
use crate::device::DeviceCapabilities;
use crate::services::certificate::ControlPlaneKeypair;
use crate::services::network_service;
use rusqlite::{params, OptionalExtension};
use time::OffsetDateTime;
use tracing::{debug, info};

/// Register a new device
pub fn register_device(
    db: &Database,
    keypair: &ControlPlaneKeypair,
    device_id: String,
    network_id: String,
    name: String,
    public_key: Vec<u8>,
    capabilities: DeviceCapabilities,
) -> ApiResult<(Vec<u8>, NetworkConnectivity)> {
    // Validate inputs
    if device_id.is_empty() {
        return Err(ApiError::BadRequest("device_id cannot be empty".into()));
    }
    if network_id.is_empty() {
        return Err(ApiError::BadRequest("network_id cannot be empty".into()));
    }
    if public_key.len() != 32 {
        return Err(ApiError::BadRequest(
            "public_key must be 32 bytes (Ed25519)".into(),
        ));
    }

    network_service::require_network_exists(db, &network_id)?;
    let connectivity = network_service::load_network_connectivity(db, &network_id)?;

    let conn = db.get_conn()?;

    // Check if device already registered
    let existing: Option<String> = conn
        .query_row(
            "SELECT device_id FROM devices WHERE device_id = ?",
            params![&device_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if existing.is_some() {
        return Err(ApiError::Conflict(format!(
            "Device {} already registered",
            device_id
        )));
    }

    // Generate certificate
    let certificate = keypair
        .generate_certificate(&device_id, &network_id, &public_key)
        .map_err(|e| ApiError::Internal(format!("Failed to generate certificate: {}", e)))?;

    // Serialize capabilities to JSON
    let capabilities_json = serde_json::to_string(&capabilities)
        .map_err(|e| ApiError::Internal(format!("Failed to serialize capabilities: {}", e)))?;

    // Insert device into database
    let now = OffsetDateTime::now_utc();
    let now_str = now
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))?;

    conn.execute(
        r#"
        INSERT INTO devices (
            device_id, network_id, name, public_key, capabilities,
            certificate, status, created_at, last_seen
        ) VALUES (?, ?, ?, ?, ?, ?, 'online', ?, ?)
        "#,
        params![
            &device_id,
            &network_id,
            &name,
            &public_key,
            &capabilities_json,
            &certificate,
            &now_str,
            &now_str
        ],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    info!(
        device_id = %device_id,
        network_id = %network_id,
        tier = ?capabilities.tier,
        "Device registered successfully"
    );

    Ok((certificate, connectivity))
}

/// Update device heartbeat
pub fn update_heartbeat(db: &Database, device_id: String) -> ApiResult<String> {
    // Validate device_id
    if device_id.is_empty() {
        return Err(ApiError::BadRequest("device_id cannot be empty".into()));
    }

    let now = OffsetDateTime::now_utc();
    let now_str = now
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))?;

    let conn = db.get_conn()?;

    // Update last_seen and set status to online
    let rows_affected = conn
        .execute(
            r#"
        UPDATE devices
        SET last_seen = ?, status = 'online'
        WHERE device_id = ?
        "#,
            params![&now_str, &device_id],
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if rows_affected == 0 {
        return Err(ApiError::NotFound(format!(
            "Device {} not found",
            device_id
        )));
    }

    debug!(device_id = %device_id, last_seen = %now_str, "Heartbeat updated");

    Ok(now_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::{
        ConnectivityAttachment, ConnectivityAttachmentKind, ConnectivityPath, NetworkConnectivity,
    };
    use crate::db::create_test_db;
    use crate::device::Tier;
    use crate::services::network_service;

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

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            cpu_cores: 8,
            ram_mb: 16384,
            os: "macos".into(),
            arch: "aarch64".into(),
            has_gpu: false,
            tier: Tier::Tier2,
        }
    }

    #[test]
    fn test_register_device() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
        )
        .unwrap();

        let device_id = "test-device-1";
        let network_id = "test-network";
        let public_key = vec![42u8; 32];

        let (certificate, connectivity) = register_device(
            &db,
            &keypair,
            device_id.to_string(),
            network_id.to_string(),
            "Test Device".to_string(),
            public_key.clone(),
            test_capabilities(),
        )
        .unwrap();

        assert!(!certificate.is_empty());
        assert_eq!(connectivity.preferred_path, ConnectivityPath::Relayed);

        // Verify device in database
        let conn = db.get_conn().unwrap();
        let device: (String, String, Vec<u8>, String) = conn
            .query_row(
                "SELECT device_id, network_id, public_key, status FROM devices WHERE device_id = ?",
                params![device_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();

        assert_eq!(device.0, device_id);
        assert_eq!(device.1, network_id);
        assert_eq!(device.2, public_key);
        assert_eq!(device.3, "online");
    }

    #[test]
    fn test_register_duplicate_device() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
        )
        .unwrap();

        let device_id = "test-device-2";
        let network_id = "test-network";

        // First registration should succeed
        register_device(
            &db,
            &keypair,
            device_id.to_string(),
            network_id.to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
            test_capabilities(),
        )
        .unwrap();

        // Second registration should fail
        let result = register_device(
            &db,
            &keypair,
            device_id.to_string(),
            network_id.to_string(),
            "Test Device".to_string(),
            vec![43u8; 32],
            test_capabilities(),
        );

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::Conflict(_)));
    }

    #[test]
    fn test_update_heartbeat() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
        )
        .unwrap();

        let device_id = "test-device-3";

        // Register device first
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

        // Update heartbeat
        let timestamp = update_heartbeat(&db, device_id.to_string()).unwrap();

        assert!(!timestamp.is_empty());

        // Verify in database
        let conn = db.get_conn().unwrap();
        let device: (String, String) = conn
            .query_row(
                "SELECT last_seen, status FROM devices WHERE device_id = ?",
                params![device_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(device.1, "online");
        assert_eq!(device.0, timestamp);
    }

    #[test]
    fn test_heartbeat_nonexistent_device() {
        let db = create_test_db();

        let result = update_heartbeat(&db, "nonexistent-device".to_string());

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::NotFound(_)));
    }

    #[test]
    fn test_invalid_public_key_length() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();
        network_service::create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            test_connectivity(),
        )
        .unwrap();

        let result = register_device(
            &db,
            &keypair,
            "test-device".to_string(),
            "test-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 16], // Wrong length
            test_capabilities(),
        );

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ApiError::BadRequest(_)));
    }

    #[test]
    fn test_register_device_requires_existing_network() {
        let db = create_test_db();
        let keypair = ControlPlaneKeypair::load_or_generate().unwrap();

        let result = register_device(
            &db,
            &keypair,
            "test-device-missing-network".to_string(),
            "missing-network".to_string(),
            "Test Device".to_string(),
            vec![42u8; 32],
            test_capabilities(),
        );

        assert!(matches!(result, Err(ApiError::NotFound(_))));
    }
}
