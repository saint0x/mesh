CREATE TABLE IF NOT EXISTS inference_session_kv_transfer_payloads (
    transfer_id TEXT PRIMARY KEY
        REFERENCES inference_session_kv_transfers(transfer_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL
        REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL
        REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    source_device_id TEXT NOT NULL,
    target_device_id TEXT NOT NULL,
    kv_sequence_position INTEGER,
    payload_size_bytes INTEGER NOT NULL,
    payload_sha256 TEXT NOT NULL,
    payload_bytes BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inference_session_kv_transfer_payloads_job_updated
    ON inference_session_kv_transfer_payloads(job_id, updated_at DESC);
