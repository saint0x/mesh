use crate::api::types::{
    HeartbeatRequest, HeartbeatResponse, RegisterDeviceRequest, RegisterDeviceResponse,
};
use crate::device::DeviceConfig;
use crate::errors::{AgentError, Result};
use reqwest::Client;
use std::time::Duration;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Client for registering and maintaining connection with control plane
pub struct RegistrationClient {
    client: Client,
    control_plane_url: String,
}

impl RegistrationClient {
    /// Create a new registration client
    pub fn new(control_plane_url: String) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .map_err(|e| AgentError::Http(format!("Failed to build HTTP client: {}", e)))?,
            control_plane_url,
        })
    }

    /// Register device with the control plane
    ///
    /// Returns the certificate blob on success
    pub async fn register(&self, config: &DeviceConfig) -> Result<Vec<u8>> {
        let request = RegisterDeviceRequest {
            device_id: config.device_id.to_string(),
            network_id: config.network_id.clone(),
            name: config.name.clone(),
            public_key: config.keypair.verifying_key().to_bytes().to_vec(),
            capabilities: config.capabilities.clone(),
        };

        info!(
            device_id = %config.device_id,
            network_id = %config.network_id,
            control_plane = %self.control_plane_url,
            "Registering device with control plane"
        );

        // Try registration with exponential backoff
        let mut retry_delay = Duration::from_secs(1);
        let max_retries = 5;

        for attempt in 1..=max_retries {
            match self.try_register(&request).await {
                Ok(cert) => {
                    info!(
                        device_id = %config.device_id,
                        attempt = attempt,
                        "Device registered successfully"
                    );
                    return Ok(cert);
                }
                Err(e) if attempt < max_retries => {
                    warn!(
                        device_id = %config.device_id,
                        attempt = attempt,
                        retry_in = ?retry_delay,
                        error = %e,
                        "Registration failed, retrying"
                    );
                    sleep(retry_delay).await;
                    retry_delay = std::cmp::min(retry_delay * 2, Duration::from_secs(60));
                }
                Err(e) => {
                    error!(
                        device_id = %config.device_id,
                        attempts = max_retries,
                        error = %e,
                        "Registration failed after all retries"
                    );
                    return Err(e);
                }
            }
        }

        unreachable!()
    }

    /// Try to register once (no retries)
    async fn try_register(&self, request: &RegisterDeviceRequest) -> Result<Vec<u8>> {
        let url = format!("{}/api/devices/register", self.control_plane_url);

        let response = self
            .client
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Registration request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Registration(format!(
                "HTTP {}: {}",
                status, error_text
            )));
        }

        let register_response: RegisterDeviceResponse = response
            .json()
            .await
            .map_err(|e| AgentError::Serialization(format!("Failed to parse response: {}", e)))?;

        if !register_response.success {
            return Err(AgentError::Registration(
                register_response
                    .message
                    .unwrap_or_else(|| "Registration failed".to_string()),
            ));
        }

        register_response
            .certificate
            .ok_or_else(|| AgentError::Registration("No certificate in response".to_string()))
    }

    /// Send a single heartbeat
    pub async fn heartbeat(&self, device_id: Uuid) -> Result<()> {
        let url = format!("{}/api/devices/{}/heartbeat", self.control_plane_url, device_id);

        let response = self
            .client
            .post(&url)
            .json(&HeartbeatRequest {})
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Heartbeat request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Heartbeat failed: HTTP {}: {}",
                status, error_text
            )));
        }

        let heartbeat_response: HeartbeatResponse = response
            .json()
            .await
            .map_err(|e| AgentError::Serialization(format!("Failed to parse response: {}", e)))?;

        if !heartbeat_response.success {
            return Err(AgentError::Network("Heartbeat response indicated failure".to_string()));
        }

        debug!(
            device_id = %device_id,
            last_seen = %heartbeat_response.last_seen,
            "Heartbeat sent successfully"
        );

        Ok(())
    }

    /// Run heartbeat loop indefinitely
    ///
    /// Sends heartbeats every 5 seconds. Does not fail on errors, just logs warnings.
    pub async fn heartbeat_loop(self, device_id: Uuid) {
        info!(device_id = %device_id, "Starting heartbeat loop");

        let mut tick = interval(Duration::from_secs(5));

        loop {
            tick.tick().await;

            if let Err(e) = self.heartbeat(device_id).await {
                warn!(
                    device_id = %device_id,
                    error = %e,
                    "Heartbeat failed (will retry)"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceCapabilities, Tier};

    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            cpu_cores: 8,
            ram_mb: 16384,
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            gpu_present: false,
            gpu_vram_mb: None,
            tier: Tier::Tier2,
        }
    }

    #[tokio::test]
    async fn test_registration_client_creation() {
        let client = RegistrationClient::new("http://localhost:8080".to_string()).unwrap();
        assert_eq!(client.control_plane_url, "http://localhost:8080");
    }

    // Note: Integration tests with mock server would go in tests/ directory
    // using wiremock crate for HTTP mocking
}
