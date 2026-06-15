use axum::{
    http::{header::HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;
use std::thread;
use std::time::Duration;
use tracing::Span;

pub(crate) const DB_LOCKED_HEADER: HeaderName = HeaderName::from_static("x-meshnet-db-locked");
const DB_LOCK_RETRY_MAX_WAIT_MS: u64 = 10_000;
const DB_LOCK_RETRY_BACKOFF_STEP_MS: u64 = 100;

/// API error types
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Database error: {0}")]
    Database(Box<dyn std::error::Error + Send + Sync>),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Resource conflict: {0}")]
    Conflict(String),

    #[error("Internal server error: {0}")]
    Internal(String),
}

impl From<crate::db::DbError> for ApiError {
    fn from(e: crate::db::DbError) -> Self {
        ApiError::Database(Box::new(e))
    }
}

fn is_locked_sqlite_error(error: &(dyn std::error::Error + 'static)) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    if message.contains("database is locked") || message.contains("database is busy") {
        return true;
    }

    if let Some(db_error) = error.downcast_ref::<crate::db::DbError>() {
        if matches!(
            db_error,
            crate::db::DbError::Rusqlite(rusqlite::Error::SqliteFailure(code, _))
                if code.code == rusqlite::ErrorCode::DatabaseBusy
                    || code.code == rusqlite::ErrorCode::DatabaseLocked
        ) {
            return true;
        }
    }

    if let Some(sqlite_error) = error.downcast_ref::<rusqlite::Error>() {
        if matches!(
            sqlite_error,
            rusqlite::Error::SqliteFailure(code, _)
                if code.code == rusqlite::ErrorCode::DatabaseBusy
                    || code.code == rusqlite::ErrorCode::DatabaseLocked
        ) {
            return true;
        }
    }

    false
}

fn is_locked_database_error(error: &(dyn std::error::Error + Send + Sync + 'static)) -> bool {
    let mut current = Some(error as &(dyn std::error::Error + 'static));
    while let Some(err) = current {
        if is_locked_sqlite_error(err) {
            return true;
        }
        current = err.source();
    }
    false
}

pub(crate) fn is_locked_api_error(error: &ApiError) -> bool {
    let ApiError::Database(db_error) = error else {
        return false;
    };
    is_locked_database_error(db_error.as_ref())
}

pub(crate) fn execute_with_db_lock_retry<T>(
    mut op: impl FnMut() -> Result<T, ApiError>,
) -> Result<T, ApiError> {
    let started_at = std::time::Instant::now();
    let mut attempt = 0usize;
    loop {
        match op() {
            Ok(value) => return Ok(value),
            Err(error) if is_locked_api_error(&error) => {
                let elapsed_ms = started_at.elapsed().as_millis() as u64;
                if elapsed_ms >= DB_LOCK_RETRY_MAX_WAIT_MS {
                    return Err(error);
                }
                attempt += 1;
                let remaining_ms = DB_LOCK_RETRY_MAX_WAIT_MS.saturating_sub(elapsed_ms);
                let backoff_ms = (DB_LOCK_RETRY_BACKOFF_STEP_MS * attempt as u64)
                    .min(DB_LOCK_RETRY_BACKOFF_STEP_MS * 10)
                    .min(remaining_ms);
                thread::sleep(Duration::from_millis(backoff_ms));
            }
            Err(error) => return Err(error),
        }
    }
}

pub(crate) fn log_locked_route_error(route: &'static str, error: &ApiError) {
    if is_locked_api_error(error) {
        tracing::error!(route, "SQLite lock surfaced through API route");
    }
}

/// Convert ApiError into HTTP response
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, is_locked) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg, false),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg, false),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, msg, false),
            ApiError::Database(e) => {
                let handler = Span::current()
                    .metadata()
                    .map(|metadata| metadata.name())
                    .unwrap_or("unknown");
                tracing::error!("Database error handler={}", handler);
                tracing::error!(error = %e, "Database error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                    is_locked_database_error(e.as_ref()),
                )
            }
            ApiError::Internal(msg) => {
                tracing::error!(error = %msg, "Internal error");
                (StatusCode::INTERNAL_SERVER_ERROR, msg, false)
            }
        };

        let body = Json(json!({
            "success": false,
            "message": message,
        }));

        let mut response = (status, body).into_response();
        if is_locked {
            response
                .headers_mut()
                .insert(DB_LOCKED_HEADER.clone(), HeaderValue::from_static("1"));
        }
        response
    }
}

/// Result type for API handlers
pub type ApiResult<T> = Result<T, ApiError>;
