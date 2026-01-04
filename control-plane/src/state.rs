use crate::db::Database;
use crate::services::certificate::ControlPlaneKeypair;
use std::sync::Arc;

/// Axum application state shared across all request handlers
#[derive(Clone)]
pub struct AppState {
    /// Database connection pool
    pub db: Database,
    /// Control plane keypair for signing certificates
    pub keypair: Arc<ControlPlaneKeypair>,
}

impl AppState {
    /// Create new application state
    pub fn new(db: Database, keypair: Arc<ControlPlaneKeypair>) -> Self {
        Self { db, keypair }
    }
}
