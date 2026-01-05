use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::device::Tier;
use crate::errors::{AgentError, Result};

/// Ledger event for job execution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub network_id: String,
    pub event_type: String,
    pub job_id: Option<Uuid>,
    pub device_id: Uuid,
    pub credits_amount: Option<f64>,
    pub metadata: serde_json::Value,
}

/// Response from ledger event creation
#[derive(Debug, Deserialize)]
pub struct LedgerEventResponse {
    pub event_id: i64,
    pub message: String,
}

/// Client for sending ledger events to control plane
pub struct LedgerClient {
    client: reqwest::Client,
    control_plane_url: String,
    max_retries: u32,
}

impl LedgerClient {
    /// Create a new ledger client
    pub fn new(control_plane_url: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            control_plane_url,
            max_retries: 3,
        }
    }

    /// Send a ledger event with retry logic
    ///
    /// Retries on network failures with exponential backoff.
    /// Returns Ok(()) even if all retries fail (logs warning instead of propagating error).
    pub async fn send_event(&self, event: &LedgerEvent) -> Result<()> {
        let url = format!("{}/api/ledger/events", self.control_plane_url);

        for attempt in 1..=self.max_retries {
            debug!(
                attempt = attempt,
                max_retries = self.max_retries,
                event_type = %event.event_type,
                "Sending ledger event"
            );

            match self.send_event_once(&url, event).await {
                Ok(response) => {
                    info!(
                        event_id = response.event_id,
                        event_type = %event.event_type,
                        job_id = ?event.job_id,
                        "Ledger event sent successfully"
                    );
                    return Ok(());
                }
                Err(e) => {
                    if attempt < self.max_retries {
                        let backoff = Duration::from_millis(500 * 2_u64.pow(attempt - 1));
                        warn!(
                            attempt = attempt,
                            error = %e,
                            backoff_ms = backoff.as_millis(),
                            "Failed to send ledger event, retrying"
                        );
                        sleep(backoff).await;
                    } else {
                        error!(
                            error = %e,
                            event_type = %event.event_type,
                            "Failed to send ledger event after all retries"
                        );
                        // Don't propagate error - we don't want to fail job execution
                        // just because ledger tracking failed
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    /// Send event once (no retry)
    async fn send_event_once(&self, url: &str, event: &LedgerEvent) -> Result<LedgerEventResponse> {
        let response = self
            .client
            .post(url)
            .json(event)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("HTTP request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "(failed to read response)".to_string());
            return Err(AgentError::Http(format!("HTTP {} - {}", status, body)));
        }

        let ledger_response = response
            .json::<LedgerEventResponse>()
            .await
            .map_err(|e| AgentError::Http(format!("Failed to parse response: {}", e)))?;

        Ok(ledger_response)
    }

    /// Send event without waiting for response (fire and forget)
    pub fn send_event_async(&self, event: LedgerEvent) {
        let client = self.clone();
        tokio::spawn(async move {
            let _ = client.send_event(&event).await;
        });
    }
}

impl Clone for LedgerClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            control_plane_url: self.control_plane_url.clone(),
            max_retries: self.max_retries,
        }
    }
}

/// Calculate credits earned for job execution
///
/// Credits = (execution_time_ms / 1000.0) * tier_multiplier
///
/// # Tier Multipliers
/// - Tier0 (Low-end): 1.0x
/// - Tier1 (Budget): 2.0x
/// - Tier2 (Mid-range): 4.0x
/// - Tier3 (High-end): 8.0x
/// - Tier4 (Workstation): 16.0x
pub fn calculate_credits(execution_time_ms: u64, tier: &Tier) -> f64 {
    let base_rate = match tier {
        Tier::Tier0 => 1.0,
        Tier::Tier1 => 2.0,
        Tier::Tier2 => 4.0,
        Tier::Tier3 => 8.0,
        Tier::Tier4 => 16.0,
    };

    // Convert milliseconds to seconds and multiply by tier rate
    (execution_time_ms as f64 / 1000.0) * base_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_credits_tier0() {
        let credits = calculate_credits(1000, &Tier::Tier0);
        assert_eq!(credits, 1.0); // 1 second * 1.0x
    }

    #[test]
    fn test_calculate_credits_tier2() {
        let credits = calculate_credits(2000, &Tier::Tier2);
        assert_eq!(credits, 8.0); // 2 seconds * 4.0x
    }

    #[test]
    fn test_calculate_credits_tier4() {
        let credits = calculate_credits(500, &Tier::Tier4);
        assert_eq!(credits, 8.0); // 0.5 seconds * 16.0x
    }

    #[test]
    fn test_ledger_event_serialization() {
        let event = LedgerEvent {
            network_id: "test-net".to_string(),
            event_type: "job_completed".to_string(),
            job_id: Some(Uuid::new_v4()),
            device_id: Uuid::new_v4(),
            credits_amount: Some(10.5),
            metadata: serde_json::json!({"duration_ms": 250}),
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: LedgerEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.network_id, "test-net");
        assert_eq!(deserialized.event_type, "job_completed");
        assert_eq!(deserialized.credits_amount, Some(10.5));
    }
}
