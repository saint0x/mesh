-- Create networks table
-- Stores network configurations for multi-tenant isolation
CREATE TABLE IF NOT EXISTS networks (
    network_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    settings TEXT NOT NULL DEFAULT '{}'  -- JSON stored as TEXT in SQLite
);

-- Index for looking up networks by owner
CREATE INDEX IF NOT EXISTS idx_networks_owner ON networks(owner_user_id);
