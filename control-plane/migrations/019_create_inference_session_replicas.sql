CREATE TABLE IF NOT EXISTS inference_session_replicas (
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    active_segment_id TEXT,
    kv_sequence_position INTEGER,
    checkpoint_created_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_error TEXT,
    PRIMARY KEY (session_id, device_id)
);

CREATE INDEX IF NOT EXISTS idx_inference_session_replicas_job
    ON inference_session_replicas(job_id);

CREATE INDEX IF NOT EXISTS idx_inference_session_replicas_session_status
    ON inference_session_replicas(session_id, status, updated_at DESC);
