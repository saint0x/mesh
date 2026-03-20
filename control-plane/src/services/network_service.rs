use crate::api::error::{ApiError, ApiResult};
use crate::db::models::Network;
use crate::db::Database;
use rusqlite::{params, OptionalExtension};
use time::OffsetDateTime;

pub fn create_network(
    db: &Database,
    network_id: String,
    name: String,
    owner_user_id: String,
    settings: Option<serde_json::Value>,
) -> ApiResult<Network> {
    if network_id.is_empty() {
        return Err(ApiError::BadRequest("network_id cannot be empty".into()));
    }
    if name.is_empty() {
        return Err(ApiError::BadRequest("name cannot be empty".into()));
    }
    if owner_user_id.is_empty() {
        return Err(ApiError::BadRequest("owner_user_id cannot be empty".into()));
    }

    let settings_json = serde_json::to_string(&settings.unwrap_or_else(|| serde_json::json!({})))
        .map_err(|e| ApiError::Internal(format!("Failed to serialize network settings: {}", e)))?;
    let now_str = OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|e| ApiError::Internal(format!("Failed to format timestamp: {}", e)))?;

    let conn = db.get_conn()?;
    let existing: Option<String> = conn
        .query_row(
            "SELECT network_id FROM networks WHERE network_id = ?",
            params![&network_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if existing.is_some() {
        return Err(ApiError::Conflict(format!(
            "Network {} already exists",
            network_id
        )));
    }

    conn.execute(
        r#"
        INSERT INTO networks (network_id, name, owner_user_id, created_at, settings)
        VALUES (?, ?, ?, ?, ?)
        "#,
        params![&network_id, &name, &owner_user_id, &now_str, &settings_json],
    )
    .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    Ok(Network {
        network_id,
        name,
        owner_user_id,
        created_at: now_str,
        settings: settings_json,
    })
}

pub fn list_networks(db: &Database) -> ApiResult<Vec<Network>> {
    let conn = db.get_conn()?;
    let mut stmt = conn
        .prepare(
            r#"
            SELECT network_id, name, owner_user_id, created_at, settings
            FROM networks
            ORDER BY created_at ASC, network_id ASC
            "#,
        )
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    let rows = stmt
        .query_map([], |row| {
            Ok(Network {
                network_id: row.get(0)?,
                name: row.get(1)?,
                owner_user_id: row.get(2)?,
                created_at: row.get(3)?,
                settings: row.get(4)?,
            })
        })
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))
}

pub fn require_network_exists(db: &Database, network_id: &str) -> ApiResult<()> {
    let conn = db.get_conn()?;
    let existing: Option<String> = conn
        .query_row(
            "SELECT network_id FROM networks WHERE network_id = ?",
            params![network_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| ApiError::Database(Box::new(crate::db::DbError::Rusqlite(e))))?;

    if existing.is_none() {
        return Err(ApiError::NotFound(format!(
            "Network {} not found",
            network_id
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::create_test_db;

    #[test]
    fn test_create_and_list_networks() {
        let db = create_test_db();
        let network = create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            Some(serde_json::json!({"region": "us-east-1"})),
        )
        .unwrap();

        assert_eq!(network.network_id, "test-network");

        let networks = list_networks(&db).unwrap();
        assert_eq!(networks.len(), 1);
        assert_eq!(networks[0].name, "Test Network");
    }

    #[test]
    fn test_require_network_exists() {
        let db = create_test_db();
        create_network(
            &db,
            "test-network".to_string(),
            "Test Network".to_string(),
            "owner-1".to_string(),
            None,
        )
        .unwrap();

        assert!(require_network_exists(&db, "test-network").is_ok());
        assert!(matches!(
            require_network_exists(&db, "missing-network"),
            Err(ApiError::NotFound(_))
        ));
    }
}
