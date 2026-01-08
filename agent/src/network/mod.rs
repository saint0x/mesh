// Network module for libp2p mesh connectivity
//
// This module provides the networking layer using libp2p with Circuit Relay v2
// for NAT traversal and DCUTR for direct connection upgrades.

mod events;
mod job_protocol;
mod mesh_swarm;
mod ring_gossip;
mod ring_gossip_service;
mod tensor_protocol;

pub use events::{ConnectionInfo, ConnectionType, MeshEvent};
pub use job_protocol::{JobEnvelope, JobProtocol, JobProtocolConfig, JobResult};
pub use mesh_swarm::{MeshSwarm, MeshSwarmBuilder, MeshSwarmConfig, RingConnections};
pub use ring_gossip::{MemberStatus, RingGossipMessage, RingMember, RingState, RingTopology};
pub use ring_gossip_service::RingGossipService;
pub use tensor_protocol::{
    AllReducePhase, TensorMessage, TensorProtocol, TensorProtocolConfig, TENSOR_PROTOCOL_ID,
};

// Re-export libp2p types needed for handling network events
pub use libp2p::request_response::{OutboundRequestId, ResponseChannel};
