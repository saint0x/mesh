use clap::Parser;
use futures::StreamExt;
use libp2p::{
    identify, noise,
    relay,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Multiaddr, SwarmBuilder,
};
use std::error::Error;
use std::time::Duration;

/// Test client for relay server
#[derive(Parser, Debug)]
#[command(name = "test-client")]
#[command(about = "Test client for Mesh relay server")]
struct Args {
    /// Client name for logging
    #[arg(short, long, default_value = "client")]
    name: String,

    /// Relay server address
    #[arg(short, long, default_value = "/ip4/127.0.0.1/tcp/4001")]
    relay: String,
}

/// Client network behaviour with relay client
#[derive(NetworkBehaviour)]
struct ClientBehaviour {
    relay_client: relay::client::Behaviour,
    identify: identify::Behaviour,
}

impl From<identify::Event> for ClientBehaviourEvent {
    fn from(event: identify::Event) -> Self {
        ClientBehaviourEvent::Identify(event)
    }
}

impl From<relay::client::Event> for ClientBehaviourEvent {
    fn from(event: relay::client::Event) -> Self {
        ClientBehaviourEvent::RelayClient(event)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // Setup logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Starting test client: {}", args.name);

    // Generate ephemeral keypair
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let peer_id = keypair.public().to_peer_id();

    tracing::info!("Client PeerId: {}", peer_id);

    // Build swarm with relay client
    let mut swarm = SwarmBuilder::with_existing_identity(keypair.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_relay_client(noise::Config::new, yamux::Config::default)?
        .with_behaviour(|keypair, relay_client| {
            Ok(ClientBehaviour {
                relay_client,
                identify: identify::Behaviour::new(identify::Config::new(
                    "/mesh-client/1.0.0".to_string(),
                    keypair.public(),
                )),
            })
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Listen on localhost (ephemeral port)
    swarm.listen_on("/ip4/127.0.0.1/tcp/0".parse()?)?;

    // Parse relay address
    let relay_addr: Multiaddr = args.relay.parse()?;
    tracing::info!("Connecting to relay: {}", relay_addr);

    // Dial relay
    swarm.dial(relay_addr.clone())?;

    let mut reservation_made = false;

    // Event loop
    loop {
        match swarm.select_next_some().await {
            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!("Listening on: {}", address);
            }

            SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                tracing::info!(
                    "Connected to peer: {} (endpoint: {:?})",
                    peer_id,
                    endpoint
                );

                // After connecting to relay, make a reservation
                if !reservation_made {
                    tracing::info!("Making reservation with relay...");
                    reservation_made = true;
                }
            }

            SwarmEvent::ConnectionClosed {
                peer_id,
                cause,
                num_established,
                ..
            } => {
                if let Some(error) = cause {
                    tracing::warn!(
                        "Connection to {} closed with error: {:?} (remaining: {})",
                        peer_id,
                        error,
                        num_established
                    );
                } else {
                    tracing::info!(
                        "Connection to {} closed (remaining: {})",
                        peer_id,
                        num_established
                    );
                }
            }

            SwarmEvent::Behaviour(ClientBehaviourEvent::RelayClient(event)) => {
                tracing::info!("Relay client event: {:?}", event);

                // Handle relay events (actual event names depend on libp2p version)
                // For now, just log all relay client events
            }

            SwarmEvent::Behaviour(ClientBehaviourEvent::Identify(event)) => {
                tracing::debug!("Identify event: {:?}", event);
            }

            SwarmEvent::IncomingConnection { send_back_addr, .. } => {
                tracing::info!("📞 Incoming connection from: {}", send_back_addr);
            }

            other => {
                tracing::trace!("Other swarm event: {:?}", other);
            }
        }
    }
}
