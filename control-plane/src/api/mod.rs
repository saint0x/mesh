pub mod error;
pub mod inference;
pub mod ledger;
pub mod ring;
pub mod routes;
pub mod status;
pub mod types;

use axum::{
    extract::Request,
    middleware::{self, Next},
    response::Response,
    routing::{delete, get, patch, post},
    Router,
};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Span;

use crate::state::AppState;

async fn log_locked_db_route(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let response = next.run(req).await;
    if response.status().is_server_error() {
        tracing::error!(
            %method,
            %path,
            status = %response.status(),
            db_locked = response
                .headers()
                .contains_key(&crate::api::error::DB_LOCKED_HEADER),
            "HTTP route returned server error"
        );
    }
    if response
        .headers()
        .contains_key(&crate::api::error::DB_LOCKED_HEADER)
    {
        tracing::error!(
            %method,
            %path,
            status = %response.status(),
            "SQLite lock response attributed to route"
        );
    }
    response
}

/// Create the API router with all endpoints
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health check endpoint
        .route("/health", get(routes::health_check))
        // Network management endpoints
        .route("/api/networks", post(routes::create_network))
        .route("/api/networks", get(routes::list_networks))
        .route(
            "/api/status/networks/:network_id/scheduler",
            get(status::get_network_scheduler_status),
        )
        .route(
            "/api/status/jobs/:job_id/scheduler",
            get(status::get_job_scheduler_status),
        )
        // Device management endpoints
        .route("/api/devices/register", post(routes::register_device))
        .route("/api/devices/:id/heartbeat", post(routes::heartbeat))
        // Ledger endpoints
        .route(
            "/api/ledger/events",
            post(ledger::create_ledger_event).get(ledger::list_ledger_events),
        )
        .route("/api/ledger/summary", get(ledger::get_ledger_summary))
        // Ring topology endpoints
        .route("/api/ring/join", post(ring::join_ring))
        .route("/api/ring/topology", get(ring::get_topology))
        .route("/api/ring/leave/:device_id", delete(ring::leave_ring))
        // Handoff management endpoints
        .route("/api/ring/handoff", post(ring::create_handoff))
        .route("/api/ring/handoff/:handoff_id", get(ring::get_handoff))
        .route("/api/ring/handoff/:handoff_id", patch(ring::update_handoff))
        .route(
            "/api/ring/handoff/:handoff_id",
            delete(ring::cancel_handoff),
        )
        .route("/api/ring/handoffs", get(ring::list_handoffs))
        // Worker callback/notification endpoints
        .route("/api/ring/callback", post(ring::register_callback))
        .route(
            "/api/ring/callback/:device_id",
            delete(ring::unregister_callback),
        )
        .route("/api/ring/version", post(ring::check_topology_version))
        // Distributed inference endpoints
        .route("/api/inference/submit", post(inference::submit_inference))
        .route(
            "/api/inference/assignments/claim",
            post(inference::claim_inference_assignment),
        )
        .route(
            "/api/inference/decode/claim",
            post(inference::claim_inference_assignment),
        )
        .route(
            "/api/inference/decode/queue",
            get(inference::observe_decode_queue_state),
        )
        .route(
            "/api/inference/decode/leases/:lease_id/renew",
            post(inference::renew_decode_lease),
        )
        .route(
            "/api/inference/decode/leases/:lease_id/release",
            post(inference::release_decode_lease),
        )
        .route(
            "/api/inference/jobs/:job_id/ack",
            post(inference::acknowledge_inference_assignment),
        )
        .route(
            "/api/inference/jobs/:job_id/result",
            post(inference::report_inference_result),
        )
        .route(
            "/api/inference/jobs/:job_id/progress",
            post(inference::report_inference_progress),
        )
        .route(
            "/api/inference/jobs/:job_id/session-checkpoints",
            post(inference::upload_inference_session_checkpoint),
        )
        .route(
            "/api/inference/jobs/:job_id/session-kv-transfers",
            post(inference::report_inference_session_kv_transfer),
        )
        .route(
            "/api/inference/session-kv-transfers/pending",
            get(inference::observe_pending_kv_transfers),
        )
        .route(
            "/api/inference/jobs/:job_id/session-kv-transfers/payloads",
            post(inference::upload_inference_session_kv_transfer_payload),
        )
        .route(
            "/api/inference/jobs/:job_id/session-kv-transfers/:transfer_id/payload",
            get(inference::download_inference_session_kv_transfer_payload),
        )
        .route(
            "/api/inference/jobs/:job_id/session-checkpoints/:session_id",
            get(inference::download_inference_session_checkpoint),
        )
        .route(
            "/api/inference/jobs/:job_id",
            get(inference::get_inference_job_status).delete(inference::cancel_inference_job),
        )
        // Attach application state
        .with_state(state)
        // Middleware
        .layer(CorsLayer::permissive())
        .layer(middleware::from_fn(log_locked_db_route))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request| {
                    tracing::info_span!(
                        "http_request",
                        method = %request.method(),
                        path = %request.uri().path()
                    )
                })
                .on_response(|response: &Response, latency: std::time::Duration, span: &Span| {
                    if response.status().is_server_error() {
                        tracing::error!(
                            parent: span,
                            status = %response.status(),
                            latency_ms = latency.as_millis(),
                            "HTTP request completed with server error"
                        );
                    }
                }),
        )
}
