CREATE TABLE IF NOT EXISTS inference_serving_groups (
    group_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    ring_position INTEGER NOT NULL,
    shard_column_start INTEGER NOT NULL,
    shard_column_end INTEGER NOT NULL,
    assigned_capacity_units INTEGER NOT NULL,
    execution_provider TEXT NOT NULL,
    status TEXT NOT NULL,
    last_error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, device_id)
);

CREATE INDEX IF NOT EXISTS idx_inference_serving_groups_session_phase
    ON inference_serving_groups(session_id, phase, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_serving_groups_job
    ON inference_serving_groups(job_id, status, updated_at DESC);
