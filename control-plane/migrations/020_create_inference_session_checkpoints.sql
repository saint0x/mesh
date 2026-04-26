CREATE TABLE IF NOT EXISTS inference_session_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    source_device_id TEXT NOT NULL,
    source_segment_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    kv_sequence_position INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    checkpoint_bytes BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY(job_id) REFERENCES inference_jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_inference_session_checkpoints_session_created
    ON inference_session_checkpoints(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_session_checkpoints_job_source
    ON inference_session_checkpoints(job_id, source_device_id, created_at DESC);
