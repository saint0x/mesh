use crate::errors::{AgentError, Result};
use libp2p::Multiaddr;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityPath {
    Direct,
    Relayed,
    Overlay,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityAttachmentKind {
    Libp2pRelay,
    UserspaceOverlay,
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

impl NetworkConnectivity {
    pub fn preferred_attachment(&self) -> Option<&ConnectivityAttachment> {
        let expected_kind = match self.preferred_path {
            ConnectivityPath::Direct => return None,
            ConnectivityPath::Relayed => ConnectivityAttachmentKind::Libp2pRelay,
            ConnectivityPath::Overlay => ConnectivityAttachmentKind::UserspaceOverlay,
        };

        self.attachments
            .iter()
            .filter(|attachment| attachment.kind == expected_kind)
            .min_by_key(|attachment| attachment.priority)
    }

    pub fn current_state(&self) -> DeviceConnectivityState {
        DeviceConnectivityState {
            active_path: self.preferred_path.clone(),
            active_endpoint: self
                .preferred_attachment()
                .map(|attachment| attachment.endpoint.clone()),
            status: ConnectivityStatus::Connected,
        }
    }

    pub fn resolve_primary_endpoint(&self) -> Result<Multiaddr> {
        let expected_kind = match self.preferred_path {
            ConnectivityPath::Direct => {
                return Err(AgentError::Config(
                    "preferred_path=direct is not yet supported by the production mesh runtime"
                        .to_string(),
                ));
            }
            ConnectivityPath::Relayed => ConnectivityAttachmentKind::Libp2pRelay,
            ConnectivityPath::Overlay => ConnectivityAttachmentKind::UserspaceOverlay,
        };

        let attachment = self
            .preferred_attachment()
            .filter(|attachment| attachment.kind == expected_kind)
            .ok_or_else(|| {
                AgentError::Config(format!(
                    "No connectivity attachment found for preferred path {:?}",
                    self.preferred_path
                ))
            })?;

        attachment
            .endpoint
            .parse()
            .map_err(|e| AgentError::Config(format!("Invalid connectivity endpoint: {}", e)))
    }
}
