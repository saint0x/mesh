use serde::{Deserialize, Serialize};

use crate::api::error::{ApiError, ApiResult};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityPath {
    Direct,
    Relayed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityAttachmentKind {
    Libp2pRelay,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectivityAttachment {
    pub kind: ConnectivityAttachmentKind,
    pub endpoint: String,
    pub priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkConnectivity {
    pub preferred_path: ConnectivityPath,
    #[serde(default)]
    pub attachments: Vec<ConnectivityAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityStatus {
    Unknown,
    Connected,
    Degraded,
    Disconnected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceConnectivityState {
    pub active_path: ConnectivityPath,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_endpoint: Option<String>,
    pub status: ConnectivityStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DirectCandidateTransport {
    Quic,
    Tcp,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DirectCandidateScope {
    Public,
    Dns,
    Private,
    Loopback,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectPeerCandidate {
    pub endpoint: String,
    pub transport: DirectCandidateTransport,
    pub scope: DirectCandidateScope,
    pub priority: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkSettings {
    pub connectivity: NetworkConnectivity,
}

impl NetworkSettings {
    pub fn validate(&self) -> ApiResult<()> {
        self.connectivity.validate()
    }
}

impl NetworkConnectivity {
    pub fn validate(&self) -> ApiResult<()> {
        if self
            .attachments
            .iter()
            .any(|attachment| attachment.endpoint.trim().is_empty())
        {
            return Err(ApiError::BadRequest(
                "connectivity attachments must include a non-empty endpoint".to_string(),
            ));
        }

        match self.preferred_path {
            ConnectivityPath::Direct => Ok(()),
            ConnectivityPath::Relayed => {
                if self
                    .attachments
                    .iter()
                    .any(|attachment| attachment.kind == ConnectivityAttachmentKind::Libp2pRelay)
                {
                    Ok(())
                } else {
                    Err(ApiError::BadRequest(
                        "preferred_path=relayed requires at least one libp2p_relay attachment"
                            .to_string(),
                    ))
                }
            }
        }
    }

    pub fn preferred_attachment(&self) -> Option<&ConnectivityAttachment> {
        let expected_kind = match self.preferred_path {
            ConnectivityPath::Direct => return None,
            ConnectivityPath::Relayed => ConnectivityAttachmentKind::Libp2pRelay,
        };

        self.attachments
            .iter()
            .filter(|attachment| attachment.kind == expected_kind)
            .min_by_key(|attachment| attachment.priority)
    }
}

impl DeviceConnectivityState {
    pub fn unknown(connectivity: &NetworkConnectivity) -> Self {
        Self {
            active_path: connectivity.preferred_path.clone(),
            active_endpoint: connectivity
                .preferred_attachment()
                .map(|attachment| attachment.endpoint.clone()),
            status: ConnectivityStatus::Unknown,
        }
    }

    pub fn disconnected(&self) -> Self {
        Self {
            active_path: self.active_path.clone(),
            active_endpoint: self.active_endpoint.clone(),
            status: ConnectivityStatus::Disconnected,
        }
    }

    pub fn validate(&self) -> ApiResult<()> {
        if matches!(self.active_path, ConnectivityPath::Relayed)
            && self
                .active_endpoint
                .as_ref()
                .map(|endpoint| endpoint.trim().is_empty())
                .unwrap_or(true)
        {
            return Err(ApiError::BadRequest(
                "non-direct connectivity state requires an active endpoint".to_string(),
            ));
        }

        if self
            .active_endpoint
            .as_ref()
            .map(|endpoint| endpoint.trim().is_empty())
            .unwrap_or(false)
        {
            return Err(ApiError::BadRequest(
                "connectivity state endpoint must not be empty".to_string(),
            ));
        }

        Ok(())
    }
}

impl DirectPeerCandidate {
    pub fn validate(&self) -> ApiResult<()> {
        if self.endpoint.trim().is_empty() {
            return Err(ApiError::BadRequest(
                "direct candidate endpoint must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}
