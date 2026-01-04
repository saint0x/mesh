use serde::{Deserialize, Serialize};
use sqlx::FromRow;

/// Network model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Network {
    pub network_id: String,
    pub name: String,
    pub owner_user_id: String,
    pub created_at: String, // ISO 8601 timestamp
    pub settings: String,   // JSON string
}

/// Device model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Device {
    pub device_id: String, // UUID as string
    pub network_id: String,
    pub name: String,
    pub public_key: Vec<u8>,          // Ed25519 public key bytes
    pub capabilities: String,         // JSON string
    pub certificate: Option<Vec<u8>>, // mTLS certificate (optional)
    pub created_at: String,           // ISO 8601 timestamp
    pub last_seen: Option<String>,
    pub status: String, // 'online', 'offline', 'revoked'
}

/// Device status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceStatus {
    Online,
    Offline,
    Revoked,
}

impl DeviceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceStatus::Online => "online",
            DeviceStatus::Offline => "offline",
            DeviceStatus::Revoked => "revoked",
        }
    }
}

impl std::str::FromStr for DeviceStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "online" => Ok(DeviceStatus::Online),
            "offline" => Ok(DeviceStatus::Offline),
            "revoked" => Ok(DeviceStatus::Revoked),
            _ => Err(format!("Invalid device status: {}", s)),
        }
    }
}

/// Ledger event model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct LedgerEvent {
    pub event_id: String, // UUID as string
    pub network_id: String,
    pub event_type: String,
    pub job_id: Option<String>,
    pub device_id: Option<String>,
    pub user_id: Option<String>,
    pub credits_amount: Option<f64>,
    pub metadata: Option<String>, // JSON string
    pub timestamp: String,        // ISO 8601 timestamp
}

/// Ledger event type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedgerEventType {
    JobStarted,
    JobCompleted,
    CreditsBurned,
    CreditsEarned,
}

impl LedgerEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LedgerEventType::JobStarted => "job_started",
            LedgerEventType::JobCompleted => "job_completed",
            LedgerEventType::CreditsBurned => "credits_burned",
            LedgerEventType::CreditsEarned => "credits_earned",
        }
    }
}

impl std::str::FromStr for LedgerEventType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "job_started" => Ok(LedgerEventType::JobStarted),
            "job_completed" => Ok(LedgerEventType::JobCompleted),
            "credits_burned" => Ok(LedgerEventType::CreditsBurned),
            "credits_earned" => Ok(LedgerEventType::CreditsEarned),
            _ => Err(format!("Invalid ledger event type: {}", s)),
        }
    }
}
