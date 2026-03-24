// Event types for mesh network operations

use libp2p::request_response::ResponseChannel;
use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::job_protocol::{JobEnvelope, JobResult};

/// High-level events emitted by the mesh swarm
#[derive(Debug)]
pub enum MeshEvent {
    /// Successfully connected to relay server
    RelayConnected {
        relay_peer_id: PeerId,
        relay_addr: Multiaddr,
    },

    /// Connection to relay server failed
    RelayConnectionFailed {
        relay_addr: Multiaddr,
        error: String,
    },

    /// Relay connection was lost
    RelayDisconnected { relay_peer_id: PeerId },

    /// Successfully established connection to a peer
    PeerConnected {
        peer_id: PeerId,
        connection_info: ConnectionInfo,
    },

    /// Lost connection to a peer
    PeerDisconnected { peer_id: PeerId },

    /// Identify protocol received information about a peer
    PeerIdentified {
        peer_id: PeerId,
        protocol_version: String,
        agent_version: String,
    },

    /// Listening on a new address (including relay addresses)
    NewListenAddr { address: Multiaddr },

    /// DCUTR (direct connection upgrade) succeeded
    DirectConnectionUpgraded { peer_id: PeerId },

    /// DCUTR failed
    DirectConnectionUpgradeFailed { peer_id: PeerId, error: String },

    /// Relay reservation succeeded
    ReservationAccepted {
        relay_peer_id: PeerId,
        renewal_timeout: std::time::Duration,
    },

    /// Relay reservation was denied
    ReservationDenied { relay_peer_id: PeerId },

    /// Received a job request from a peer
    JobReceived {
        peer_id: PeerId,
        job: JobEnvelope,
        channel: ResponseChannel<JobResult>,
    },

    /// Received a job result from a peer
    JobCompleted { peer_id: PeerId, result: JobResult },

    /// Failed to send a job request
    JobSendFailed {
        peer_id: PeerId,
        job_id: Uuid,
        error: String,
    },
}

/// Information about a peer connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    /// Whether this is a direct connection or relayed
    pub connection_type: ConnectionType,

    /// Remote address of the peer
    pub remote_addr: Multiaddr,

    /// Number of established connections to this peer
    pub num_established: u32,
}

/// Type of connection to a peer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Direct P2P connection
    Direct,

    /// Connection through relay server
    Relayed,

    /// Connection type unknown
    Unknown,
}

impl ConnectionInfo {
    /// Check if this is a direct connection
    pub fn is_direct(&self) -> bool {
        self.connection_type == ConnectionType::Direct
    }

    /// Check if this is a relayed connection
    pub fn is_relayed(&self) -> bool {
        self.connection_type == ConnectionType::Relayed
    }
}
