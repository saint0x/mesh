use control_plane::db::{models::*, Database};
use uuid::Uuid;

#[tokio::test]
async fn test_create_and_query_network() {
    let db = Database::new("sqlite::memory:")
        .await
        .expect("Failed to create database");
    db.migrate().await.expect("Failed to run migrations");

    let network_id = "test-network";
    let owner_id = "user-123";

    // Insert network
    sqlx::query(
        r#"
        INSERT INTO networks (network_id, name, owner_user_id, settings)
        VALUES (?, ?, ?, ?)
        "#,
    )
    .bind(network_id)
    .bind("Test Network")
    .bind(owner_id)
    .bind("{}")
    .execute(db.pool())
    .await
    .expect("Failed to insert network");

    // Query network
    let network: Network = sqlx::query_as("SELECT * FROM networks WHERE network_id = ?")
        .bind(network_id)
        .fetch_one(db.pool())
        .await
        .expect("Failed to query network");

    assert_eq!(network.network_id, network_id);
    assert_eq!(network.name, "Test Network");
    assert_eq!(network.owner_user_id, owner_id);
}

#[tokio::test]
async fn test_create_and_query_device() {
    let db = Database::new("sqlite::memory:")
        .await
        .expect("Failed to create database");
    db.migrate().await.expect("Failed to run migrations");

    // Create network first
    let network_id = "test-network";
    sqlx::query("INSERT INTO networks (network_id, name, owner_user_id) VALUES (?, ?, ?)")
        .bind(network_id)
        .bind("Test Network")
        .bind("user-123")
        .execute(db.pool())
        .await
        .expect("Failed to insert network");

    // Create device
    let device_id = Uuid::new_v4().to_string();
    let public_key = vec![1u8; 32]; // Mock Ed25519 public key
    let capabilities = r#"{"tier":"tier2","cpu_cores":8,"ram_mb":16384}"#;

    sqlx::query(
        r#"
        INSERT INTO devices (device_id, network_id, name, public_key, capabilities, status)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&device_id)
    .bind(network_id)
    .bind("Test Device")
    .bind(&public_key)
    .bind(capabilities)
    .bind("offline")
    .execute(db.pool())
    .await
    .expect("Failed to insert device");

    // Query device
    let device: Device = sqlx::query_as("SELECT * FROM devices WHERE device_id = ?")
        .bind(&device_id)
        .fetch_one(db.pool())
        .await
        .expect("Failed to query device");

    assert_eq!(device.device_id, device_id);
    assert_eq!(device.network_id, network_id);
    assert_eq!(device.name, "Test Device");
    assert_eq!(device.public_key, public_key);
    assert_eq!(device.status, "offline");
}

#[tokio::test]
async fn test_create_ledger_event() {
    let db = Database::new("sqlite::memory:")
        .await
        .expect("Failed to create database");
    db.migrate().await.expect("Failed to run migrations");

    // Create network first
    let network_id = "test-network";
    sqlx::query("INSERT INTO networks (network_id, name, owner_user_id) VALUES (?, ?, ?)")
        .bind(network_id)
        .bind("Test Network")
        .bind("user-123")
        .execute(db.pool())
        .await
        .expect("Failed to insert network");

    // Create ledger event
    let event_id = Uuid::new_v4().to_string();
    let job_id = Uuid::new_v4().to_string();

    sqlx::query(
        r#"
        INSERT INTO ledger_events (event_id, network_id, event_type, job_id, credits_amount, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        "#,
    )
    .bind(&event_id)
    .bind(network_id)
    .bind("job_started")
    .bind(&job_id)
    .bind(10.5)
    .bind(r#"{"model":"llama-3-8b"}"#)
    .execute(db.pool())
    .await
    .expect("Failed to insert ledger event");

    // Query ledger event
    let event: LedgerEvent = sqlx::query_as("SELECT * FROM ledger_events WHERE event_id = ?")
        .bind(&event_id)
        .fetch_one(db.pool())
        .await
        .expect("Failed to query ledger event");

    assert_eq!(event.event_id, event_id);
    assert_eq!(event.network_id, network_id);
    assert_eq!(event.event_type, "job_started");
    assert_eq!(event.job_id.unwrap(), job_id);
    assert_eq!(event.credits_amount.unwrap(), 10.5);
}

#[tokio::test]
async fn test_foreign_key_cascade_delete() {
    let db = Database::new("sqlite::memory:")
        .await
        .expect("Failed to create database");
    db.migrate().await.expect("Failed to run migrations");

    let network_id = "test-network";

    // Create network
    sqlx::query("INSERT INTO networks (network_id, name, owner_user_id) VALUES (?, ?, ?)")
        .bind(network_id)
        .bind("Test Network")
        .bind("user-123")
        .execute(db.pool())
        .await
        .expect("Failed to insert network");

    // Create device
    let device_id = Uuid::new_v4().to_string();
    sqlx::query(
        "INSERT INTO devices (device_id, network_id, name, public_key, capabilities) VALUES (?, ?, ?, ?, ?)"
    )
    .bind(&device_id)
    .bind(network_id)
    .bind("Test Device")
    .bind(vec![1u8; 32])
    .bind("{}")
    .execute(db.pool())
    .await
    .expect("Failed to insert device");

    // Delete network (should cascade to devices)
    sqlx::query("DELETE FROM networks WHERE network_id = ?")
        .bind(network_id)
        .execute(db.pool())
        .await
        .expect("Failed to delete network");

    // Verify device was deleted
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM devices WHERE device_id = ?")
        .bind(&device_id)
        .fetch_one(db.pool())
        .await
        .expect("Failed to count devices");

    assert_eq!(count.0, 0, "Device should be deleted via cascade");
}

#[tokio::test]
async fn test_unique_constraint_device_public_key() {
    let db = Database::new("sqlite::memory:")
        .await
        .expect("Failed to create database");
    db.migrate().await.expect("Failed to run migrations");

    let network_id = "test-network";

    // Create network
    sqlx::query("INSERT INTO networks (network_id, name, owner_user_id) VALUES (?, ?, ?)")
        .bind(network_id)
        .bind("Test Network")
        .bind("user-123")
        .execute(db.pool())
        .await
        .expect("Failed to insert network");

    let public_key = vec![1u8; 32];

    // Create first device
    sqlx::query(
        "INSERT INTO devices (device_id, network_id, name, public_key, capabilities) VALUES (?, ?, ?, ?, ?)"
    )
    .bind(Uuid::new_v4().to_string())
    .bind(network_id)
    .bind("Device 1")
    .bind(&public_key)
    .bind("{}")
    .execute(db.pool())
    .await
    .expect("Failed to insert first device");

    // Try to create second device with same public key in same network (should fail)
    let result = sqlx::query(
        "INSERT INTO devices (device_id, network_id, name, public_key, capabilities) VALUES (?, ?, ?, ?, ?)"
    )
    .bind(Uuid::new_v4().to_string())
    .bind(network_id)
    .bind("Device 2")
    .bind(&public_key)
    .bind("{}")
    .execute(db.pool())
    .await;

    assert!(result.is_err(), "Duplicate public key should fail");
}

#[tokio::test]
async fn test_ledger_indexes() {
    let db = Database::new("sqlite::memory:")
        .await
        .expect("Failed to create database");
    db.migrate().await.expect("Failed to run migrations");

    // Verify indexes exist
    let indexes: Vec<(String,)> = sqlx::query_as(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_ledger%'",
    )
    .fetch_all(db.pool())
    .await
    .expect("Failed to query indexes");

    let index_names: Vec<String> = indexes.into_iter().map(|(name,)| name).collect();

    assert!(index_names.contains(&"idx_ledger_network_time".to_string()));
    assert!(index_names.contains(&"idx_ledger_device_time".to_string()));
    assert!(index_names.contains(&"idx_ledger_job".to_string()));
    assert!(index_names.contains(&"idx_ledger_type".to_string()));
}
