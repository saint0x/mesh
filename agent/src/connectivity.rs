use crate::errors::{AgentError, Result};
use libp2p::{Multiaddr, PeerId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const OBSERVED_CANDIDATE_MAX_AGE_MS: u64 = 5 * 60 * 1000;

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
#[serde(rename_all = "snake_case")]
pub enum DirectCandidateSource {
    LocalListen,
    ObservedExternal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectPeerCandidate {
    pub endpoint: String,
    pub transport: DirectCandidateTransport,
    pub scope: DirectCandidateScope,
    pub source: DirectCandidateSource,
    pub priority: u32,
    pub last_updated_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectCandidateSeed {
    pub endpoint: String,
    pub source: DirectCandidateSource,
    pub last_updated_ms: u64,
}

impl NetworkConnectivity {
    fn configured_state(&self) -> DeviceConnectivityState {
        DeviceConnectivityState {
            active_path: self.preferred_path.clone(),
            active_endpoint: self
                .preferred_attachment()
                .map(|attachment| attachment.endpoint.clone()),
            status: ConnectivityStatus::Connected,
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

    pub fn current_state(&self) -> DeviceConnectivityState {
        load_runtime_connectivity_state().unwrap_or_else(|| self.configured_state())
    }

    pub fn runtime_endpoint(&self) -> Result<Option<Multiaddr>> {
        match self.preferred_path {
            ConnectivityPath::Direct => Ok(None),
            ConnectivityPath::Relayed => Ok(Some(self.resolve_primary_endpoint()?)),
        }
    }

    pub fn resolve_primary_endpoint(&self) -> Result<Multiaddr> {
        let expected_kind = match self.preferred_path {
            ConnectivityPath::Direct => {
                return Err(AgentError::Config(
                    "direct connectivity does not use a static endpoint".to_string(),
                ));
            }
            ConnectivityPath::Relayed => ConnectivityAttachmentKind::Libp2pRelay,
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

fn runtime_connectivity_state_path() -> Option<PathBuf> {
    let base = std::env::var_os("MESHNET_HOME")
        .map(PathBuf::from)
        .or_else(dirs::home_dir)?;
    Some(base.join(".meshnet").join("connectivity_state.json"))
}

fn meshnet_state_path(filename: &str) -> Option<PathBuf> {
    let base = std::env::var_os("MESHNET_HOME")
        .map(PathBuf::from)
        .or_else(dirs::home_dir)?;
    Some(base.join(".meshnet").join(filename))
}

pub fn persist_runtime_connectivity_state(state: &DeviceConnectivityState) -> Result<()> {
    let path = runtime_connectivity_state_path()
        .ok_or_else(|| AgentError::Config("Could not determine home directory".to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| AgentError::Serialization(format!("Failed to serialize state: {}", e)))?;
    fs::write(path, json)?;
    Ok(())
}

pub fn load_runtime_connectivity_state() -> Option<DeviceConnectivityState> {
    let path = runtime_connectivity_state_path()?;
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn persist_observed_reachability_addr(address: &Multiaddr) -> Result<()> {
    let path = meshnet_state_path("observed_addrs.json")
        .ok_or_else(|| AgentError::Config("Could not determine home directory".to_string()))?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut records = load_observed_reachability_records().unwrap_or_default();
    prune_stale_observed_records(&mut records);
    let address = address.to_string();
    let Some(candidate) = address
        .parse::<Multiaddr>()
        .ok()
        .and_then(|addr| classify_direct_addr(&addr))
    else {
        fs::write(path, serde_json::to_string_pretty(&records)?)?;
        return Ok(());
    };

    if !matches!(
        candidate.scope,
        DirectCandidateScope::Public | DirectCandidateScope::Dns
    ) {
        fs::write(path, serde_json::to_string_pretty(&records)?)?;
        return Ok(());
    }

    if let Some(existing) = records.iter_mut().find(|record| record.endpoint == address) {
        existing.last_updated_ms = now_epoch_ms();
    } else {
        records.push(DirectCandidateSeed {
            endpoint: address,
            source: DirectCandidateSource::ObservedExternal,
            last_updated_ms: now_epoch_ms(),
        });
    }
    fs::write(path, serde_json::to_string_pretty(&records)?)?;
    Ok(())
}

pub fn load_direct_candidate_seed_records() -> Option<Vec<DirectCandidateSeed>> {
    let listen_path = meshnet_state_path("listen_addrs.json")?;
    let mut combined = load_json_string_vec(&listen_path)
        .unwrap_or_default()
        .into_iter()
        .map(|endpoint| DirectCandidateSeed {
            endpoint,
            source: DirectCandidateSource::LocalListen,
            last_updated_ms: now_epoch_ms(),
        })
        .collect::<Vec<_>>();

    for record in load_observed_reachability_records().unwrap_or_default() {
        if let Some(existing) = combined
            .iter_mut()
            .find(|candidate| candidate.endpoint == record.endpoint)
        {
            if record.last_updated_ms > existing.last_updated_ms {
                existing.last_updated_ms = record.last_updated_ms;
            }
            existing.source = record.source.clone();
        } else {
            combined.push(record);
        }
    }

    Some(combined)
}

pub fn load_direct_candidate_seed_addrs() -> Option<Vec<String>> {
    Some(
        load_direct_candidate_seed_records()?
            .into_iter()
            .map(|record| record.endpoint)
            .collect(),
    )
}

pub fn load_observed_reachability_records() -> Option<Vec<DirectCandidateSeed>> {
    let observed_path = meshnet_state_path("observed_addrs.json")?;
    let mut records = load_json_seed_vec(&observed_path)?;
    prune_stale_observed_records(&mut records);
    Some(records)
}

pub fn load_observed_reachability_addrs() -> Option<Vec<String>> {
    Some(
        load_observed_reachability_records()?
            .into_iter()
            .map(|record| record.endpoint)
            .collect(),
    )
}

fn load_json_string_vec(path: &PathBuf) -> Option<Vec<String>> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn load_json_seed_vec(path: &PathBuf) -> Option<Vec<DirectCandidateSeed>> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn select_direct_dial_addrs(peer_id: PeerId, addrs: &[String]) -> Vec<Multiaddr> {
    let parsed = addrs
        .iter()
        .filter_map(|addr| addr.parse::<Multiaddr>().ok())
        .collect::<Vec<_>>();
    select_direct_dial_multiaddrs(peer_id, &parsed)
}

pub fn select_direct_dial_multiaddrs(peer_id: PeerId, addrs: &[Multiaddr]) -> Vec<Multiaddr> {
    let mut ranked = addrs
        .iter()
        .filter_map(|addr| {
            classify_direct_addr(addr)
                .map(|candidate| (candidate.priority, canonical_direct_addr(addr, peer_id)))
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|(left_rank, left_addr), (right_rank, right_addr)| {
        left_rank
            .cmp(right_rank)
            .then_with(|| left_addr.to_string().cmp(&right_addr.to_string()))
    });

    let mut seen = HashSet::new();
    ranked
        .into_iter()
        .map(|(_, addr)| addr)
        .filter(|addr| seen.insert(addr.to_string()))
        .collect()
}

pub fn build_direct_peer_candidates(peer_id: PeerId, addrs: &[String]) -> Vec<DirectPeerCandidate> {
    let seeds = addrs
        .iter()
        .map(|endpoint| DirectCandidateSeed {
            endpoint: endpoint.clone(),
            source: DirectCandidateSource::LocalListen,
            last_updated_ms: now_epoch_ms(),
        })
        .collect::<Vec<_>>();
    build_direct_peer_candidates_from_records(peer_id, &seeds)
}

pub fn build_direct_peer_candidates_from_records(
    peer_id: PeerId,
    seeds: &[DirectCandidateSeed],
) -> Vec<DirectPeerCandidate> {
    let now_ms = now_epoch_ms();
    let parsed = seeds
        .iter()
        .filter(|seed| direct_candidate_seed_is_fresh(seed, now_ms))
        .filter_map(|seed| {
            seed.endpoint
                .parse::<Multiaddr>()
                .ok()
                .map(|addr| (seed, addr))
        })
        .collect::<Vec<_>>();

    let mut candidates = parsed
        .iter()
        .filter_map(|(seed, addr)| {
            classify_direct_addr(addr).map(|mut candidate| {
                if matches!(candidate.scope, DirectCandidateScope::Loopback) {
                    return None;
                }
                candidate.endpoint = canonical_direct_addr(addr, peer_id).to_string();
                candidate.source = seed.source.clone();
                candidate.last_updated_ms = seed.last_updated_ms;
                Some(candidate)
            })
        })
        .flatten()
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        left.priority
            .cmp(&right.priority)
            .then_with(|| {
                candidate_source_rank(&left.source).cmp(&candidate_source_rank(&right.source))
            })
            .then_with(|| right.last_updated_ms.cmp(&left.last_updated_ms))
            .then_with(|| left.endpoint.cmp(&right.endpoint))
    });
    candidates.dedup_by(|left, right| left.endpoint == right.endpoint);
    candidates
}

pub fn select_direct_dial_addrs_from_candidates(
    peer_id: PeerId,
    candidates: &[DirectPeerCandidate],
) -> Vec<Multiaddr> {
    let mut seen = HashSet::new();
    candidates
        .iter()
        .filter_map(|candidate| candidate.endpoint.parse::<Multiaddr>().ok())
        .map(|addr| canonical_direct_addr(&addr, peer_id))
        .filter(|addr| seen.insert(addr.to_string()))
        .collect()
}

fn canonical_direct_addr(addr: &Multiaddr, peer_id: PeerId) -> Multiaddr {
    if addr
        .iter()
        .any(|protocol| matches!(protocol, libp2p::multiaddr::Protocol::P2p(_)))
    {
        addr.clone()
    } else {
        addr.clone().with(libp2p::multiaddr::Protocol::P2p(peer_id))
    }
}

fn classify_direct_addr(addr: &Multiaddr) -> Option<DirectPeerCandidate> {
    if addr
        .iter()
        .any(|protocol| matches!(protocol, libp2p::multiaddr::Protocol::P2pCircuit))
    {
        return None;
    }

    let mut ip_scope = AddrScope::Dns;
    let mut transport = None;

    for protocol in addr.iter() {
        match protocol {
            libp2p::multiaddr::Protocol::Ip4(ip) => ip_scope = classify_ip_scope(IpAddr::V4(ip)),
            libp2p::multiaddr::Protocol::Ip6(ip) => ip_scope = classify_ip_scope(IpAddr::V6(ip)),
            libp2p::multiaddr::Protocol::Dns(_)
            | libp2p::multiaddr::Protocol::Dns4(_)
            | libp2p::multiaddr::Protocol::Dns6(_)
            | libp2p::multiaddr::Protocol::Dnsaddr(_) => {
                ip_scope = AddrScope::Dns;
            }
            libp2p::multiaddr::Protocol::Quic | libp2p::multiaddr::Protocol::QuicV1 => {
                transport = Some(DirectCandidateTransport::Quic)
            }
            libp2p::multiaddr::Protocol::Tcp(_) => {
                transport.get_or_insert(DirectCandidateTransport::Tcp);
            }
            _ => {}
        }
    }

    let transport = transport?;
    let transport_rank = match transport {
        DirectCandidateTransport::Quic => 0,
        DirectCandidateTransport::Tcp => 1,
    };
    let scope = match ip_scope {
        AddrScope::Public => DirectCandidateScope::Public,
        AddrScope::Dns => DirectCandidateScope::Dns,
        AddrScope::Private => DirectCandidateScope::Private,
        AddrScope::Loopback => DirectCandidateScope::Loopback,
        AddrScope::Unsupported => return None,
    };

    let scope_rank = match scope {
        DirectCandidateScope::Public => 0,
        DirectCandidateScope::Dns => 10,
        DirectCandidateScope::Private => 20,
        DirectCandidateScope::Loopback => 30,
    };

    Some(DirectPeerCandidate {
        endpoint: addr.to_string(),
        transport,
        scope,
        source: DirectCandidateSource::LocalListen,
        priority: (scope_rank + transport_rank) as u32,
        last_updated_ms: now_epoch_ms(),
    })
}

fn candidate_source_rank(source: &DirectCandidateSource) -> u8 {
    match source {
        DirectCandidateSource::ObservedExternal => 0,
        DirectCandidateSource::LocalListen => 1,
    }
}

fn direct_candidate_seed_is_fresh(seed: &DirectCandidateSeed, now_ms: u64) -> bool {
    match seed.source {
        DirectCandidateSource::LocalListen => true,
        DirectCandidateSource::ObservedExternal => {
            now_ms.saturating_sub(seed.last_updated_ms) <= OBSERVED_CANDIDATE_MAX_AGE_MS
        }
    }
}

fn prune_stale_observed_records(records: &mut Vec<DirectCandidateSeed>) {
    let now_ms = now_epoch_ms();
    records.retain(|record| {
        record.source == DirectCandidateSource::ObservedExternal
            && direct_candidate_seed_is_fresh(record, now_ms)
    });
}

fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AddrScope {
    Public,
    Dns,
    Private,
    Loopback,
    Unsupported,
}

fn classify_ip_scope(ip: IpAddr) -> AddrScope {
    match ip {
        IpAddr::V4(ip) => classify_ipv4_scope(ip),
        IpAddr::V6(ip) => classify_ipv6_scope(ip),
    }
}

fn classify_ipv4_scope(ip: Ipv4Addr) -> AddrScope {
    if ip.is_loopback() {
        return AddrScope::Loopback;
    }
    if ip.is_unspecified() || ip.is_multicast() || ip.is_broadcast() || ip.is_documentation() {
        return AddrScope::Unsupported;
    }
    if ip.is_private() || ip.is_link_local() {
        return AddrScope::Private;
    }
    AddrScope::Public
}

fn classify_ipv6_scope(ip: Ipv6Addr) -> AddrScope {
    if ip.is_loopback() {
        return AddrScope::Loopback;
    }
    if ip.is_unspecified() || ip.is_multicast() || ipv6_is_documentation(ip) {
        return AddrScope::Unsupported;
    }
    if ip.is_unicast_link_local() || ipv6_is_unique_local(ip) {
        return AddrScope::Private;
    }
    AddrScope::Public
}

fn ipv6_is_unique_local(ip: Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xfe00) == 0xfc00
}

fn ipv6_is_documentation(ip: Ipv6Addr) -> bool {
    ip.segments()[0] == 0x2001 && ip.segments()[1] == 0x0db8
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn select_direct_dial_addrs_prefers_public_quic_then_private_tcp() {
        let peer_id = PeerId::random();
        let addrs = vec![
            "/ip4/192.168.1.44/tcp/4001".to_string(),
            "/ip4/34.120.0.10/udp/4001/quic-v1".to_string(),
            "/ip4/34.120.0.10/tcp/4001".to_string(),
            "/ip4/127.0.0.1/tcp/4001".to_string(),
            "/dns4/peer.mesh.example/udp/4001/quic-v1".to_string(),
            "/dns4/relay.mesh.example/tcp/4001/p2p-circuit".to_string(),
        ];

        let selected = select_direct_dial_addrs(peer_id, &addrs);
        let rendered = selected.iter().map(ToString::to_string).collect::<Vec<_>>();

        assert_eq!(
            rendered,
            vec![
                format!("/ip4/34.120.0.10/udp/4001/quic-v1/p2p/{}", peer_id),
                format!("/ip4/34.120.0.10/tcp/4001/p2p/{}", peer_id),
                format!("/dns4/peer.mesh.example/udp/4001/quic-v1/p2p/{}", peer_id),
                format!("/ip4/192.168.1.44/tcp/4001/p2p/{}", peer_id),
                format!("/ip4/127.0.0.1/tcp/4001/p2p/{}", peer_id),
            ]
        );
    }

    #[test]
    fn build_direct_peer_candidates_excludes_relay_and_sorts() {
        let peer_id = PeerId::random();
        let addrs = vec![
            "/dns4/peer.mesh.example/udp/4001/quic-v1".to_string(),
            "/ip4/10.0.0.2/tcp/4001".to_string(),
            "/dns4/relay.mesh.example/tcp/4001/p2p-circuit".to_string(),
        ];

        let candidates = build_direct_peer_candidates(peer_id, &addrs);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].scope, DirectCandidateScope::Dns);
        assert_eq!(candidates[0].transport, DirectCandidateTransport::Quic);
        assert_eq!(candidates[0].source, DirectCandidateSource::LocalListen);
        assert_eq!(candidates[1].scope, DirectCandidateScope::Private);
        assert_eq!(candidates[1].transport, DirectCandidateTransport::Tcp);
    }

    #[test]
    fn current_state_prefers_runtime_state_when_present() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("MESHNET_HOME");
        std::env::set_var("MESHNET_HOME", tempdir.path());

        let state = DeviceConnectivityState {
            active_path: ConnectivityPath::Direct,
            active_endpoint: Some("/ip4/10.0.0.8/tcp/4001".to_string()),
            status: ConnectivityStatus::Connected,
        };
        persist_runtime_connectivity_state(&state).unwrap();

        let connectivity = NetworkConnectivity {
            preferred_path: ConnectivityPath::Relayed,
            attachments: vec![ConnectivityAttachment {
                kind: ConnectivityAttachmentKind::Libp2pRelay,
                endpoint: "/dns4/relay.mesh.example/tcp/4001".to_string(),
                priority: 0,
            }],
        };

        assert_eq!(connectivity.current_state(), state);
        match previous {
            Some(value) => std::env::set_var("MESHNET_HOME", value),
            None => std::env::remove_var("MESHNET_HOME"),
        }
    }

    #[test]
    fn direct_candidate_seed_addrs_merge_listen_and_observed() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("MESHNET_HOME");
        std::env::set_var("MESHNET_HOME", tempdir.path());

        let listen_path = tempdir.path().join(".meshnet").join("listen_addrs.json");
        std::fs::create_dir_all(listen_path.parent().unwrap()).unwrap();
        std::fs::write(
            &listen_path,
            serde_json::to_string_pretty(&vec![
                "/ip4/10.0.0.2/tcp/4001".to_string(),
                "/dns4/peer.mesh.example/udp/4001/quic-v1".to_string(),
            ])
            .unwrap(),
        )
        .unwrap();

        persist_observed_reachability_addr(
            &"/ip4/34.120.0.10/tcp/4001".parse::<Multiaddr>().unwrap(),
        )
        .unwrap();

        let merged = load_direct_candidate_seed_records().unwrap();
        assert_eq!(merged.len(), 3);
        assert!(merged
            .iter()
            .any(|record| record.endpoint.contains("34.120.0.10")
                && record.source == DirectCandidateSource::ObservedExternal));

        match previous {
            Some(value) => std::env::set_var("MESHNET_HOME", value),
            None => std::env::remove_var("MESHNET_HOME"),
        }
    }

    #[test]
    fn build_direct_peer_candidates_prefers_observed_external_hints() {
        let peer_id = PeerId::random();
        let now = now_epoch_ms();
        let seeds = vec![
            DirectCandidateSeed {
                endpoint: "/ip4/34.120.0.10/tcp/4001".to_string(),
                source: DirectCandidateSource::LocalListen,
                last_updated_ms: now - 10_000,
            },
            DirectCandidateSeed {
                endpoint: "/ip4/34.120.0.10/tcp/4001".to_string(),
                source: DirectCandidateSource::ObservedExternal,
                last_updated_ms: now,
            },
        ];

        let candidates = build_direct_peer_candidates_from_records(peer_id, &seeds);
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].source,
            DirectCandidateSource::ObservedExternal
        );
        assert_eq!(candidates[0].last_updated_ms, now);
    }

    #[test]
    fn persist_observed_reachability_addr_ignores_private_and_loopback() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tempdir = tempfile::tempdir().unwrap();
        let previous = std::env::var_os("MESHNET_HOME");
        std::env::set_var("MESHNET_HOME", tempdir.path());

        persist_observed_reachability_addr(
            &"/ip4/127.0.0.1/tcp/4001".parse::<Multiaddr>().unwrap(),
        )
        .unwrap();
        persist_observed_reachability_addr(
            &"/ip4/192.168.1.5/tcp/4001".parse::<Multiaddr>().unwrap(),
        )
        .unwrap();
        persist_observed_reachability_addr(
            &"/ip4/34.120.0.10/tcp/4001".parse::<Multiaddr>().unwrap(),
        )
        .unwrap();

        let observed = load_observed_reachability_records().unwrap();
        assert_eq!(observed.len(), 1);
        assert_eq!(observed[0].endpoint, "/ip4/34.120.0.10/tcp/4001");

        match previous {
            Some(value) => std::env::set_var("MESHNET_HOME", value),
            None => std::env::remove_var("MESHNET_HOME"),
        }
    }

    #[test]
    fn build_direct_peer_candidates_excludes_loopback_and_stale_observed() {
        let peer_id = PeerId::random();
        let now = now_epoch_ms();
        let seeds = vec![
            DirectCandidateSeed {
                endpoint: "/ip4/127.0.0.1/tcp/4001".to_string(),
                source: DirectCandidateSource::LocalListen,
                last_updated_ms: now,
            },
            DirectCandidateSeed {
                endpoint: "/ip4/34.120.0.10/tcp/4001".to_string(),
                source: DirectCandidateSource::ObservedExternal,
                last_updated_ms: now - OBSERVED_CANDIDATE_MAX_AGE_MS - 1,
            },
            DirectCandidateSeed {
                endpoint: "/ip4/192.168.1.44/tcp/4001".to_string(),
                source: DirectCandidateSource::LocalListen,
                last_updated_ms: now,
            },
        ];

        let candidates = build_direct_peer_candidates_from_records(peer_id, &seeds);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].scope, DirectCandidateScope::Private);
        assert_eq!(
            candidates[0].endpoint,
            format!("/ip4/192.168.1.44/tcp/4001/p2p/{}", peer_id)
        );
    }
}
