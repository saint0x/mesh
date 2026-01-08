pub mod error;
pub mod inference;
pub mod ledger;
pub mod ring;
pub mod routes;
pub mod types;

use axum::{
    routing::{delete, get, patch, post},
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
        // Ring topology endpoints
        .route("/api/ring/join", post(ring::join_ring))
        .route("/api/ring/topology", get(ring::get_topology))
        .route("/api/ring/leave/:device_id", delete(ring::leave_ring))
        // Handoff management endpoints
        .route("/api/ring/handoff", post(ring::create_handoff))
        .route("/api/ring/handoff/:handoff_id", get(ring::get_handoff))
        .route("/api/ring/handoff/:handoff_id", patch(ring::update_handoff))
        .route("/api/ring/handoff/:handoff_id", delete(ring::cancel_handoff))
        .route("/api/ring/handoffs", get(ring::list_handoffs))
        // Worker callback/notification endpoints
        .route("/api/ring/callback", post(ring::register_callback))
        .route("/api/ring/callback/:device_id", delete(ring::unregister_callback))
        .route("/api/ring/version", post(ring::check_topology_version))
        // Distributed inference endpoints
        .route("/api/inference/submit", post(inference::submit_inference))
        .route("/api/inference/poll/:network_id", get(inference::poll_inference_job))
        // Attach application state
        .with_state(state)
        // Middleware
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
}
