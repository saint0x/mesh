CREATE TABLE IF NOT EXISTS inference_session_kv_transfers (
    transfer_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    batch_group_key TEXT,
    source_device_id TEXT NOT NULL,
    target_device_id TEXT NOT NULL,
    transfer_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    checkpoint_id TEXT,
    remote_access_uri TEXT,
    kv_sequence_position INTEGER,
    bytes_total INTEGER,
    bytes_transferred INTEGER,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_session_kv_transfers_unique_target
    ON inference_session_kv_transfers(session_id, segment_id, target_device_id);

CREATE INDEX IF NOT EXISTS idx_inference_session_kv_transfers_network_updated
    ON inference_session_kv_transfers(network_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_session_kv_transfers_job_updated
    ON inference_session_kv_transfers(job_id, updated_at DESC);
