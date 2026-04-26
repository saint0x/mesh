CREATE TABLE IF NOT EXISTS inference_sessions (
    session_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL,
    active_segment_id TEXT,
    kv_owner_device_id TEXT NOT NULL,
    kv_transfer_policy TEXT NOT NULL,
    kv_sequence_position INTEGER,
    kv_checkpoint_device_id TEXT,
    kv_checkpoint_created_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(job_id)
);

CREATE INDEX IF NOT EXISTS idx_inference_sessions_job
    ON inference_sessions(job_id);

CREATE INDEX IF NOT EXISTS idx_inference_sessions_network_status
    ON inference_sessions(network_id, status, updated_at DESC);
