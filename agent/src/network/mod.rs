// Network module for libp2p mesh connectivity
//
// This module provides the networking layer using libp2p with Circuit Relay v2
// for NAT traversal and DCUTR for direct connection upgrades.

mod events;
mod mesh_swarm;

pub use events::{MeshEvent, ConnectionInfo, ConnectionType};
pub use mesh_swarm::{MeshSwarm, MeshSwarmBuilder, MeshSwarmConfig};
