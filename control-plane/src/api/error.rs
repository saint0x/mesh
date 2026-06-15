use axum::{
    http::{header::HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::future::Future;
use std::thread;
use std::time::Duration;
use thiserror::Error;
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
    if message_indicates_locked(&error.to_string())
        || message_indicates_locked(&format!("{error:?}"))
    {
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

fn message_indicates_locked(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("database is locked")
        || lowered.contains("database is busy")
        || lowered.contains("database table is locked")
        || lowered.contains("database schema is locked")
        || (lowered.contains("database") && lowered.contains("locked"))
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

pub(crate) async fn execute_with_db_lock_retry_async<T, Fut>(
    mut op: impl FnMut() -> Fut,
) -> Result<T, ApiError>
where
    Fut: Future<Output = Result<T, ApiError>>,
{
    let started_at = std::time::Instant::now();
    let mut attempt = 0usize;
    loop {
        match op().await {
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
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    fn locked_error() -> ApiError {
        ApiError::Database(Box::new(io::Error::other("database is locked")))
    }

    #[tokio::test]
    async fn async_retry_retries_locked_errors_until_success() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = execute_with_db_lock_retry_async({
            let attempts = attempts.clone();
            move || {
                let attempts = attempts.clone();
                async move {
                    let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                    if attempt < 2 {
                        Err(locked_error())
                    } else {
                        Ok::<_, ApiError>("ok")
                    }
                }
            }
        })
        .await
        .expect("locked retry should eventually succeed");

        assert_eq!(result, "ok");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn async_retry_does_not_retry_non_locked_errors() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = execute_with_db_lock_retry_async({
            let attempts = attempts.clone();
            move || {
                let attempts = attempts.clone();
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>(ApiError::Internal("boom".to_string()))
                }
            }
        })
        .await;

        assert!(matches!(result, Err(ApiError::Internal(message)) if message == "boom"));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
