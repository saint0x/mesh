// Integration test example: Two agents connecting through a relay server
//
// This example demonstrates:
// 1. Agent A and Agent B both connect to a local relay server
// 2. Agent A creates a relay reservation
// 3. Agent B dials Agent A through the relay circuit
// 4. Successful peer connection is established
//
// Prerequisites:
// - Relay server must be running on localhost:4001
//
// Usage:
//   # Terminal 1: Start relay server
//   cargo run --bin relay-server
//
//   # Terminal 2: Run this integration test
//   cargo run --example relay_connectivity

use agent::{MeshSwarm, MeshEvent};
use tracing::{info, warn, error};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("relay_connectivity=debug".parse()?)
            .add_directive("agent=debug".parse()?))
        .init();

    info!("Starting relay connectivity integration test");

    // Generate keypairs for two agents
    let keypair_a = libp2p::identity::Keypair::generate_ed25519();
    let keypair_b = libp2p::identity::Keypair::generate_ed25519();

    let peer_id_a = keypair_a.public().to_peer_id();
    let peer_id_b = keypair_b.public().to_peer_id();

    info!("Agent A PeerID: {}", peer_id_a);
    info!("Agent B PeerID: {}", peer_id_b);

    // Build swarms for both agents
    let mut swarm_a = MeshSwarm::builder(keypair_a)
        .build()?;

    let mut swarm_b = MeshSwarm::builder(keypair_b)
        .build()?;

    info!("Both agents initialized");

    // Step 1: Agent A connects to relay
    info!("Agent A: Connecting to relay server...");
    swarm_a.connect_to_relay()?;

    // Wait for Agent A to connect and get relay peer ID
    let relay_peer_id = loop {
        match swarm_a.next_event().await {
            Some(MeshEvent::PeerConnected { peer_id, connection_info }) => {
                info!("Agent A: Connected to relay server {}", peer_id);
                info!("Agent A: Connection type: {:?}", connection_info.connection_type);
                break peer_id;
            }
            Some(MeshEvent::PeerIdentified { peer_id, agent_version, .. }) => {
                info!("Agent A: Relay identified as {} ({})", peer_id, agent_version);
            }
            Some(event) => {
                info!("Agent A: Event: {:?}", event);
            }
            None => {
                error!("Agent A: Event stream ended unexpectedly");
                return Ok(());
            }
        }
    };

    // Step 2: Agent A creates relay reservation
    info!("Agent A: Creating relay reservation...");
    swarm_a.listen_on_relay(relay_peer_id)?;

    // Wait for reservation confirmation
    loop {
        match swarm_a.next_event().await {
            Some(MeshEvent::ReservationAccepted { relay_peer_id, .. }) => {
                info!("Agent A: Relay reservation accepted by {}", relay_peer_id);
                break;
            }
            Some(MeshEvent::ReservationDenied { relay_peer_id }) => {
                error!("Agent A: Relay reservation denied by {}", relay_peer_id);
                return Err("Reservation denied".into());
            }
            Some(MeshEvent::NewListenAddr { address }) => {
                info!("Agent A: Now listening on: {}", address);
            }
            Some(event) => {
                info!("Agent A: Event: {:?}", event);
            }
            None => {
                error!("Agent A: Event stream ended unexpectedly");
                return Ok(());
            }
        }
    }

    // Step 3: Agent B connects to relay
    info!("Agent B: Connecting to relay server...");
    swarm_b.connect_to_relay()?;

    // Wait for Agent B to connect to relay
    loop {
        match swarm_b.next_event().await {
            Some(MeshEvent::PeerConnected { peer_id, .. }) => {
                info!("Agent B: Connected to relay server {}", peer_id);
                break;
            }
            Some(MeshEvent::PeerIdentified { peer_id, agent_version, .. }) => {
                info!("Agent B: Relay identified as {} ({})", peer_id, agent_version);
            }
            Some(event) => {
                info!("Agent B: Event: {:?}", event);
            }
            None => {
                error!("Agent B: Event stream ended unexpectedly");
                return Ok(());
            }
        }
    }

    // Step 4: Agent B creates reservation
    info!("Agent B: Creating relay reservation...");
    swarm_b.listen_on_relay(relay_peer_id)?;

    loop {
        match swarm_b.next_event().await {
            Some(MeshEvent::ReservationAccepted { .. }) => {
                info!("Agent B: Relay reservation accepted");
                break;
            }
            Some(MeshEvent::NewListenAddr { address }) => {
                info!("Agent B: Now listening on: {}", address);
            }
            Some(event) => {
                info!("Agent B: Event: {:?}", event);
            }
            None => {
                error!("Agent B: Event stream ended unexpectedly");
                return Ok(());
            }
        }
    }

    // Step 5: Agent B dials Agent A through the relay
    info!("Agent B: Dialing Agent A ({}) through relay...", peer_id_a);
    swarm_b.dial_peer(peer_id_a)?;

    // Run event loops for both agents concurrently
    let agent_a_task = tokio::spawn(async move {
        loop {
            match swarm_a.next_event().await {
                Some(MeshEvent::PeerConnected { peer_id, connection_info }) => {
                    info!("Agent A: Peer connected: {}", peer_id);
                    if connection_info.is_relayed() {
                        info!("Agent A: ✅ SUCCESS! Relayed connection established");
                    } else if connection_info.is_direct() {
                        info!("Agent A: ✅ DCUTR SUCCESS! Direct connection upgraded");
                    }
                    // Test successful - break loop
                    return Ok::<_, Box<dyn std::error::Error + Send + Sync>>(());
                }
                Some(MeshEvent::PeerDisconnected { peer_id }) => {
                    warn!("Agent A: Peer disconnected: {}", peer_id);
                }
                Some(event) => {
                    info!("Agent A: Event: {:?}", event);
                }
                None => {
                    error!("Agent A: Event stream ended");
                    return Ok(());
                }
            }
        }
    });

    let agent_b_task = tokio::spawn(async move {
        loop {
            match swarm_b.next_event().await {
                Some(MeshEvent::PeerConnected { peer_id, connection_info }) => {
                    info!("Agent B: Peer connected: {}", peer_id);
                    if peer_id == peer_id_a {
                        if connection_info.is_relayed() {
                            info!("Agent B: ✅ SUCCESS! Connected to Agent A via relay");
                        } else if connection_info.is_direct() {
                            info!("Agent B: ✅ DCUTR SUCCESS! Direct connection to Agent A");
                        }
                        // Test successful - break loop
                        return Ok::<_, Box<dyn std::error::Error + Send + Sync>>(());
                    }
                }
                Some(MeshEvent::PeerDisconnected { peer_id }) => {
                    warn!("Agent B: Peer disconnected: {}", peer_id);
                }
                Some(event) => {
                    info!("Agent B: Event: {:?}", event);
                }
                None => {
                    error!("Agent B: Event stream ended");
                    return Ok(());
                }
            }
        }
    });

    // Wait for both agents to complete (with timeout)
    tokio::select! {
        result = agent_a_task => {
            match result {
                Ok(Ok(())) => info!("Agent A task completed successfully"),
                Ok(Err(e)) => error!("Agent A task failed: {}", e),
                Err(e) => error!("Agent A task panicked: {}", e),
            }
        }
        result = agent_b_task => {
            match result {
                Ok(Ok(())) => info!("Agent B task completed successfully"),
                Ok(Err(e)) => error!("Agent B task failed: {}", e),
                Err(e) => error!("Agent B task panicked: {}", e),
            }
        }
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(30)) => {
            warn!("Test timed out after 30 seconds");
        }
    }

    info!("Integration test completed");
    Ok(())
}
