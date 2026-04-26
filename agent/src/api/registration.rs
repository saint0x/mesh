use crate::api::types::{
    AcknowledgeInferenceAssignmentRequest, ClaimInferenceAssignmentRequest,
    ClaimInferenceAssignmentResponse, DownloadInferenceSessionCheckpointResponse, HeartbeatRequest,
    HeartbeatResponse, InferenceExecutionLease, ObserveDecodeQueueStateResponse,
    RegisterDeviceRequest, RegisterDeviceResponse, ReleaseDecodeLeaseRequest,
    ReleaseDecodeLeaseResponse, RenewDecodeLeaseRequest, RenewDecodeLeaseResponse,
    ReportInferenceAssignmentProgressRequest, ReportInferenceAssignmentRequest,
    UploadInferenceSessionCheckpointRequest, WorkClaimMode,
};
use crate::connectivity::{
    build_direct_peer_candidates_from_records, filter_peer_advertisable_addrs,
    load_direct_candidate_seed_records,
};
use crate::device::DeviceConfig;
use crate::errors::{AgentError, Result};
use reqwest::Client;
use serde::de::DeserializeOwned;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Client for registering and maintaining connection with control plane
#[derive(Clone)]
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
    /// Returns the certificate blob and relay addresses on success
    pub async fn register(&self, config: &DeviceConfig) -> Result<RegisterDeviceResponse> {
        let request = RegisterDeviceRequest {
            device_id: config.device_id.to_string(),
            network_id: config.network_id.clone(),
            name: config.name.clone(),
            public_key: config.keypair.verifying_key().to_bytes().to_vec(),
            peer_id: crate::device::keypair::to_libp2p_keypair(&config.keypair)
                .public()
                .to_peer_id()
                .to_string(),
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
                Ok(response) => {
                    info!(
                        device_id = %config.device_id,
                        attempt = attempt,
                        "Device registered successfully"
                    );
                    return Ok(response);
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
    async fn try_register(
        &self,
        request: &RegisterDeviceRequest,
    ) -> Result<RegisterDeviceResponse> {
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

        if register_response.certificate.is_none() {
            return Err(AgentError::Registration(
                "No certificate in response".to_string(),
            ));
        }

        Ok(register_response)
    }

    /// Send a single heartbeat
    pub async fn heartbeat(&self, config: &DeviceConfig) -> Result<()> {
        let url = format!(
            "{}/api/devices/{}/heartbeat",
            self.control_plane_url, config.device_id
        );

        let response = self
            .client
            .post(&url)
            .json(&{
                let listen_addrs = filter_peer_advertisable_addrs(
                    &load_advertised_listen_addrs().unwrap_or_default(),
                );
                let candidate_seed_records =
                    load_direct_candidate_seed_records().unwrap_or_else(|| {
                        let now_ms = current_epoch_ms();
                        listen_addrs
                            .iter()
                            .map(|endpoint| crate::connectivity::DirectCandidateSeed {
                                endpoint: endpoint.clone(),
                                source: crate::connectivity::DirectCandidateSource::LocalListen,
                                last_updated_ms: now_ms,
                            })
                            .collect()
                    });
                let peer_id = crate::device::keypair::to_libp2p_keypair(&config.keypair)
                    .public()
                    .to_peer_id();

                HeartbeatRequest {
                    connectivity_state: config.connectivity.current_state(),
                    listen_addrs,
                    direct_candidates: build_direct_peer_candidates_from_records(
                        peer_id,
                        &candidate_seed_records,
                    ),
                }
            })
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
            return Err(AgentError::Network(
                "Heartbeat response indicated failure".to_string(),
            ));
        }

        debug!(
            device_id = %config.device_id,
            last_seen = %heartbeat_response.last_seen,
            active_path = ?heartbeat_response.connectivity_state.active_path,
            connectivity_status = ?heartbeat_response.connectivity_state.status,
            listen_addr_count = heartbeat_response.listen_addrs.len(),
            direct_candidate_count = heartbeat_response.direct_candidates.len(),
            "Heartbeat sent successfully"
        );

        Ok(())
    }

    /// Run heartbeat loop indefinitely
    ///
    /// Sends heartbeats every 5 seconds. Does not fail on errors, just logs warnings.
    pub async fn heartbeat_loop(self, config: DeviceConfig) {
        info!(device_id = %config.device_id, "Starting heartbeat loop");

        let mut tick = interval(Duration::from_secs(5));

        loop {
            tick.tick().await;

            if let Err(e) = self.heartbeat(&config).await {
                warn!(
                    device_id = %config.device_id,
                    error = %e,
                    "Heartbeat failed (will retry)"
                );
            }
        }
    }

    pub async fn claim_inference_assignment(
        &self,
        device_id: Uuid,
        network_id: &str,
    ) -> Result<Option<InferenceExecutionLease>> {
        Ok(self
            .claim_inference_work(ClaimInferenceAssignmentRequest {
                device_id: device_id.to_string(),
                network_id: network_id.to_string(),
                claim_mode: WorkClaimMode::Any,
                include_queue_state: false,
                include_serving_session: false,
            })
            .await?
            .assignment)
    }

    pub async fn claim_inference_work(
        &self,
        request: ClaimInferenceAssignmentRequest,
    ) -> Result<ClaimInferenceAssignmentResponse> {
        let url = format!("{}/api/inference/assignments/claim", self.control_plane_url);
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Assignment claim failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Assignment claim failed: HTTP {}: {}",
                status, error_text
            )));
        }

        let body: ClaimInferenceAssignmentResponse = response.json().await.map_err(|e| {
            AgentError::Serialization(format!("Failed to parse assignment claim: {}", e))
        })?;

        Ok(body)
    }

    pub async fn claim_decode_work(
        &self,
        device_id: Uuid,
        network_id: &str,
    ) -> Result<Option<ClaimInferenceAssignmentResponse>> {
        let explicit_url = format!("{}/api/inference/decode/claim", self.control_plane_url);
        let request = ClaimInferenceAssignmentRequest {
            device_id: device_id.to_string(),
            network_id: network_id.to_string(),
            claim_mode: WorkClaimMode::Decode,
            include_queue_state: true,
            include_serving_session: true,
        };

        if let Some(response) = self
            .post_optional_json::<_, ClaimInferenceAssignmentResponse>(
                &explicit_url,
                &request,
                "Explicit decode claim failed",
            )
            .await?
        {
            return Ok(Some(response));
        }

        Ok(Some(self.claim_inference_work(request).await?))
    }

    pub async fn observe_decode_queue_state(
        &self,
        device_id: Uuid,
        network_id: &str,
    ) -> Result<Option<ObserveDecodeQueueStateResponse>> {
        let url = format!("{}/api/inference/decode/queue", self.control_plane_url);
        self.get_optional_json(
            &url,
            &[
                ("device_id", device_id.to_string()),
                ("network_id", network_id.to_string()),
            ],
            "Decode queue observation failed",
        )
        .await
    }

    pub async fn renew_decode_lease(
        &self,
        lease_id: &str,
        request: RenewDecodeLeaseRequest,
    ) -> Result<Option<RenewDecodeLeaseResponse>> {
        let url = format!(
            "{}/api/inference/decode/leases/{}/renew",
            self.control_plane_url, lease_id
        );
        self.post_optional_json(&url, &request, "Decode lease renew failed")
            .await
    }

    pub async fn release_decode_lease(
        &self,
        lease_id: &str,
        request: ReleaseDecodeLeaseRequest,
    ) -> Result<Option<ReleaseDecodeLeaseResponse>> {
        let url = format!(
            "{}/api/inference/decode/leases/{}/release",
            self.control_plane_url, lease_id
        );
        self.post_optional_json(&url, &request, "Decode lease release failed")
            .await
    }

    pub async fn acknowledge_inference_assignment(
        &self,
        job_id: Uuid,
        device_id: Uuid,
    ) -> Result<()> {
        let url = format!(
            "{}/api/inference/jobs/{}/ack",
            self.control_plane_url, job_id
        );
        let response = self
            .client
            .post(&url)
            .json(&AcknowledgeInferenceAssignmentRequest {
                device_id: device_id.to_string(),
            })
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Assignment acknowledge failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Assignment acknowledge failed: HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }

    pub async fn report_inference_result(
        &self,
        job_id: Uuid,
        request: ReportInferenceAssignmentRequest,
    ) -> Result<()> {
        let url = format!(
            "{}/api/inference/jobs/{}/result",
            self.control_plane_url, job_id
        );
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Inference result report failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Inference result report failed: HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }

    pub async fn report_inference_progress(
        &self,
        job_id: Uuid,
        request: ReportInferenceAssignmentProgressRequest,
    ) -> Result<()> {
        let url = format!(
            "{}/api/inference/jobs/{}/progress",
            self.control_plane_url, job_id
        );
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Inference progress report failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Inference progress report failed: HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }

    pub async fn upload_inference_session_checkpoint(
        &self,
        job_id: Uuid,
        request: UploadInferenceSessionCheckpointRequest,
    ) -> Result<()> {
        let url = format!(
            "{}/api/inference/jobs/{}/session-checkpoints",
            self.control_plane_url, job_id
        );
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("Session checkpoint upload failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Session checkpoint upload failed: HTTP {}: {}",
                status, error_text
            )));
        }

        Ok(())
    }

    pub async fn download_inference_session_checkpoint(
        &self,
        job_id: Uuid,
        session_id: Uuid,
    ) -> Result<Option<Vec<u8>>> {
        let url = format!(
            "{}/api/inference/jobs/{}/session-checkpoints/{}",
            self.control_plane_url, job_id, session_id
        );
        let response =
            self.client.get(&url).send().await.map_err(|e| {
                AgentError::Http(format!("Session checkpoint download failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AgentError::Network(format!(
                "Session checkpoint download failed: HTTP {}: {}",
                status, error_text
            )));
        }

        let body: DownloadInferenceSessionCheckpointResponse =
            response.json().await.map_err(|e| {
                AgentError::Serialization(format!(
                    "Failed to parse session checkpoint download: {}",
                    e
                ))
            })?;

        body.checkpoint
            .map(|checkpoint| {
                hex::decode(checkpoint.checkpoint_hex).map_err(|e| {
                    AgentError::Serialization(format!(
                        "Downloaded session checkpoint had invalid hex payload: {}",
                        e
                    ))
                })
            })
            .transpose()
    }
}

impl RegistrationClient {
    async fn post_optional_json<B: serde::Serialize, T: DeserializeOwned>(
        &self,
        url: &str,
        body: &B,
        error_context: &str,
    ) -> Result<Option<T>> {
        let response = self
            .client
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("{}: {}", error_context, e)))?;
        parse_optional_json_response(response, error_context).await
    }

    async fn get_optional_json<T: DeserializeOwned>(
        &self,
        url: &str,
        query: &[(&str, String)],
        error_context: &str,
    ) -> Result<Option<T>> {
        let response = self
            .client
            .get(url)
            .query(query)
            .send()
            .await
            .map_err(|e| AgentError::Http(format!("{}: {}", error_context, e)))?;
        parse_optional_json_response(response, error_context).await
    }
}

async fn parse_optional_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    error_context: &str,
) -> Result<Option<T>> {
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::METHOD_NOT_ALLOWED
    {
        return Ok(None);
    }

    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(AgentError::Network(format!(
            "{}: HTTP {}: {}",
            error_context, status, error_text
        )));
    }

    response
        .json::<T>()
        .await
        .map(Some)
        .map_err(|e| AgentError::Serialization(format!("{}: {}", error_context, e)))
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn load_advertised_listen_addrs() -> Option<Vec<String>> {
    let path = dirs::home_dir()?.join(".meshnet").join("listen_addrs.json");
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceCapabilities, Tier};
    use crate::provider::{ExecutionProviderInfo, ExecutionProviderKind};

    /// Test helper to create device capabilities
    /// Used by integration tests in tests/ directory
    #[allow(dead_code)]
    fn test_capabilities() -> DeviceCapabilities {
        DeviceCapabilities {
            cpu_cores: 8,
            ram_mb: 16384,
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            gpu_present: false,
            gpu_vram_mb: None,
            execution_providers: vec![
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cpu,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Metal,
                    available: true,
                    reason: None,
                },
                ExecutionProviderInfo {
                    kind: ExecutionProviderKind::Cuda,
                    available: false,
                    reason: Some("cuda provider is only available on Linux builds".to_string()),
                },
            ],
            default_execution_provider: ExecutionProviderKind::Metal,
            tier: Tier::Tier2,
        }
    }

    #[tokio::test]
    async fn test_registration_client_creation() {
        let client = RegistrationClient::new("http://localhost:8080".to_string()).unwrap();
        assert_eq!(client.control_plane_url, "http://localhost:8080");
    }

    // Integration tests for the HTTP registration surface belong in tests/.
}
