use libp2p::swarm::SwarmEvent;
use crate::relay::RelayBehaviourEvent;

/// Handle swarm events from the libp2p event loop
pub async fn handle_swarm_event(event: SwarmEvent<RelayBehaviourEvent>) {
    match event {
        SwarmEvent::NewListenAddr { address, .. } => {
            tracing::info!(address = %address, "Listening on address");
            println!("Listening on: {}", address);
        }

        SwarmEvent::ConnectionEstablished {
            peer_id,
            endpoint,
            connection_id,
            num_established,
            ..
        } => {
            tracing::info!(
                peer_id = %peer_id,
                endpoint = ?endpoint,
                connection_id = ?connection_id,
                num_established = num_established,
                "Connection established"
            );
        }

        SwarmEvent::ConnectionClosed {
            peer_id,
            cause,
            num_established,
            connection_id,
            ..
        } => {
            if let Some(error) = cause {
                tracing::warn!(
                    peer_id = %peer_id,
                    connection_id = ?connection_id,
                    cause = ?error,
                    remaining = num_established,
                    "Connection closed with error"
                );
            } else {
                tracing::info!(
                    peer_id = %peer_id,
                    connection_id = ?connection_id,
                    remaining = num_established,
                    "Connection closed normally"
                );
            }
        }

        SwarmEvent::IncomingConnection { send_back_addr, local_addr, connection_id } => {
            tracing::debug!(
                connection_id = ?connection_id,
                from = %send_back_addr,
                local = %local_addr,
                "Incoming connection"
            );
        }

        SwarmEvent::IncomingConnectionError { send_back_addr, error, connection_id, .. } => {
            tracing::warn!(
                connection_id = ?connection_id,
                from = %send_back_addr,
                error = %error,
                "Incoming connection error"
            );
        }

        SwarmEvent::OutgoingConnectionError { peer_id, error, connection_id } => {
            if let Some(peer) = peer_id {
                tracing::warn!(
                    peer_id = %peer,
                    connection_id = ?connection_id,
                    error = %error,
                    "Outgoing connection error"
                );
            } else {
                tracing::warn!(
                    connection_id = ?connection_id,
                    error = %error,
                    "Outgoing connection error (no peer_id)"
                );
            }
        }

        SwarmEvent::Behaviour(event) => {
            handle_behaviour_event(event).await;
        }

        other => {
            tracing::trace!("Swarm event: {:?}", other);
        }
    }
}

/// Handle behaviour-specific events (Identify, Relay)
async fn handle_behaviour_event(event: RelayBehaviourEvent) {
    use libp2p::{identify, relay};

    match event {
        // Identify protocol events
        RelayBehaviourEvent::Identify(identify::Event::Received { peer_id, info, .. }) => {
            tracing::debug!(
                peer_id = %peer_id,
                protocol_version = %info.protocol_version,
                agent_version = %info.agent_version,
                listen_addrs = ?info.listen_addrs,
                protocols = ?info.protocols,
                "Peer identified"
            );
        }

        RelayBehaviourEvent::Identify(identify::Event::Sent { .. }) => {
            tracing::trace!("Sent identify info");
        }

        RelayBehaviourEvent::Identify(identify::Event::Pushed { .. }) => {
            tracing::trace!("Pushed identify update");
        }

        RelayBehaviourEvent::Identify(identify::Event::Error { peer_id, error, .. }) => {
            tracing::warn!(peer_id = %peer_id, error = ?error, "Identify error");
        }

        // Relay protocol events - Reservations
        RelayBehaviourEvent::Relay(relay::Event::ReservationReqAccepted {
            src_peer_id,
            renewed,
        }) => {
            if renewed {
                tracing::info!(
                    peer_id = %src_peer_id,
                    "Reservation renewed"
                );
            } else {
                tracing::info!(
                    peer_id = %src_peer_id,
                    "Reservation accepted"
                );
            }
        }

        RelayBehaviourEvent::Relay(relay::Event::ReservationReqDenied {
            src_peer_id,
            ..
        }) => {
            tracing::warn!(
                peer_id = %src_peer_id,
                "Reservation denied (likely hit max limit)"
            );
        }

        RelayBehaviourEvent::Relay(relay::Event::ReservationTimedOut {
            src_peer_id,
        }) => {
            tracing::info!(
                peer_id = %src_peer_id,
                "Reservation timed out"
            );
        }

        // Relay protocol events - Circuits
        RelayBehaviourEvent::Relay(relay::Event::CircuitReqAccepted {
            src_peer_id,
            dst_peer_id,
        }) => {
            tracing::info!(
                src = %src_peer_id,
                dst = %dst_peer_id,
                "Circuit established"
            );
        }

        RelayBehaviourEvent::Relay(relay::Event::CircuitReqDenied {
            src_peer_id,
            dst_peer_id,
            ..
        }) => {
            tracing::warn!(
                src = %src_peer_id,
                dst = %dst_peer_id,
                "Circuit denied"
            );
        }

        RelayBehaviourEvent::Relay(relay::Event::CircuitClosed {
            src_peer_id,
            dst_peer_id,
            error,
        }) => {
            if let Some(err) = error {
                tracing::warn!(
                    src = %src_peer_id,
                    dst = %dst_peer_id,
                    error = ?err,
                    "Circuit closed with error"
                );
            } else {
                tracing::info!(
                    src = %src_peer_id,
                    dst = %dst_peer_id,
                    "Circuit closed normally"
                );
            }
        }

        other => {
            tracing::trace!("Behaviour event: {:?}", other);
        }
    }
}
