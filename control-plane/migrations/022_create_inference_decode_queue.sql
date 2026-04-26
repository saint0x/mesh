CREATE TABLE IF NOT EXISTS inference_decode_queue (
    session_id TEXT PRIMARY KEY REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL,
    group_id TEXT NOT NULL,
    status TEXT NOT NULL,
    ready_at TEXT,
    lease_owner_device_id TEXT,
    lease_expires_at TEXT,
    last_error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inference_decode_queue_network_status
    ON inference_decode_queue(network_id, status, updated_at ASC);

CREATE INDEX IF NOT EXISTS idx_inference_decode_queue_job
    ON inference_decode_queue(job_id, status, updated_at DESC);
