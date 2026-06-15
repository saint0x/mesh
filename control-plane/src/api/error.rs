use axum::{
    http::{header::HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

pub(crate) const DB_LOCKED_HEADER: HeaderName = HeaderName::from_static("x-meshnet-db-locked");

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

fn is_locked_database_error(error: &(dyn std::error::Error + Send + Sync + 'static)) -> bool {
    let Some(db_error) = error.downcast_ref::<crate::db::DbError>() else {
        return false;
    };
    matches!(
        db_error,
        crate::db::DbError::Rusqlite(rusqlite::Error::SqliteFailure(code, _))
            if code.code == rusqlite::ErrorCode::DatabaseBusy
                || code.code == rusqlite::ErrorCode::DatabaseLocked
    )
}

/// Convert ApiError into HTTP response
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, is_locked) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg, false),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg, false),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, msg, false),
            ApiError::Database(e) => {
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
