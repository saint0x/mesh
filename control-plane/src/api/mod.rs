pub mod error;
pub mod ledger;
pub mod routes;
pub mod types;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Create the API router with all endpoints
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health check endpoint
        .route("/health", get(routes::health_check))
        // Device management endpoints
        .route("/api/devices/register", post(routes::register_device))
        .route("/api/devices/:id/heartbeat", post(routes::heartbeat))
        // Ledger endpoints
        .route("/api/ledger/events", post(ledger::create_ledger_event))
        // Attach application state
        .with_state(state)
        // Middleware
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}
