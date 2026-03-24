use serde::{Deserialize, Serialize};

use crate::api::error::{ApiError, ApiResult};

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
        if self.attachments.iter().any(|attachment| attachment.endpoint.trim().is_empty()) {
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
            ConnectivityPath::Overlay => {
                if self
                    .attachments
                    .iter()
                    .any(|attachment| attachment.kind == ConnectivityAttachmentKind::UserspaceOverlay)
                {
                    Ok(())
                } else {
                    Err(ApiError::BadRequest(
                        "preferred_path=overlay requires at least one userspace_overlay attachment"
                            .to_string(),
                    ))
                }
            }
        }
    }
}
