-- Add ring topology fields to devices table
-- Supports distributed worker ring management and shard assignments

-- Ring position and neighbor tracking
ALTER TABLE devices ADD COLUMN ring_position INTEGER;
ALTER TABLE devices ADD COLUMN left_neighbor_id TEXT;
ALTER TABLE devices ADD COLUMN right_neighbor_id TEXT;

-- Shard assignment (column ranges for model partitioning)
ALTER TABLE devices ADD COLUMN shard_column_start INTEGER;
ALTER TABLE devices ADD COLUMN shard_column_end INTEGER;

-- Resource contribution tracking
ALTER TABLE devices ADD COLUMN contributed_memory INTEGER;

-- Lock status for coordinated operations
ALTER TABLE devices ADD COLUMN lock_status TEXT DEFAULT 'unlocked';
ALTER TABLE devices ADD COLUMN lock_timestamp TEXT;
ALTER TABLE devices ADD COLUMN unlock_requested_at TEXT;

-- Create resource_locks table for tracking locked memory resources
CREATE TABLE IF NOT EXISTS resource_locks (
    device_id TEXT PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
    memory_bytes INTEGER NOT NULL,
    lock_timestamp TEXT NOT NULL,
    cooldown_hours INTEGER NOT NULL DEFAULT 24,
    unlock_requested_at TEXT,
    status TEXT NOT NULL DEFAULT 'locked'
);

-- Create pools table for managing worker pools per model
CREATE TABLE IF NOT EXISTS pools (
    pool_id TEXT PRIMARY KEY,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    ring_stable BOOLEAN DEFAULT FALSE,
    total_workers INTEGER,
    active_workers INTEGER,
    status TEXT DEFAULT 'initializing',
    last_checkpoint_token INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for ring topology queries
CREATE INDEX IF NOT EXISTS idx_devices_ring_position ON devices(ring_position);
CREATE INDEX IF NOT EXISTS idx_pools_network ON pools(network_id);
CREATE INDEX IF NOT EXISTS idx_pools_model ON pools(model_id);
CREATE INDEX IF NOT EXISTS idx_resource_locks_status ON resource_locks(status);
