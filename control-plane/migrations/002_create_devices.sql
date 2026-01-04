-- Create devices table
-- Stores device registrations with their capabilities and certificates
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,  -- UUID stored as TEXT
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    public_key BLOB NOT NULL,  -- Ed25519 public key (32 bytes)
    capabilities TEXT NOT NULL,  -- JSON stored as TEXT
    certificate BLOB,  -- mTLS certificate (optional, added after registration)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT,  -- Last heartbeat timestamp
    status TEXT NOT NULL DEFAULT 'offline',  -- 'online', 'offline', 'revoked'
    UNIQUE(network_id, public_key)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_devices_network ON devices(network_id);
CREATE INDEX IF NOT EXISTS idx_devices_status ON devices(status);
CREATE INDEX IF NOT EXISTS idx_devices_last_seen ON devices(last_seen);
