CREATE TABLE IF NOT EXISTS inference_session_kv_residency (
    residency_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    group_id TEXT NOT NULL,
    replica_device_id TEXT NOT NULL,
    shard_column_start INTEGER NOT NULL,
    shard_column_end INTEGER NOT NULL,
    owner_device_id TEXT NOT NULL,
    residency_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    sequence_first_position INTEGER,
    sequence_next_position INTEGER,
    cached_tokens INTEGER,
    payload_size_bytes INTEGER,
    remote_access_uri TEXT,
    checkpoint_id TEXT,
    prompt_cache_key TEXT,
    prompt_prefix_tokens INTEGER,
    eviction_eligible INTEGER NOT NULL DEFAULT 0,
    pinned_for_decode INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_session_kv_residency_unique_slice
    ON inference_session_kv_residency(session_id, phase, group_id, replica_device_id);

CREATE INDEX IF NOT EXISTS idx_inference_session_kv_residency_network_updated
    ON inference_session_kv_residency(network_id, updated_at DESC, residency_id DESC);

CREATE INDEX IF NOT EXISTS idx_inference_session_kv_residency_job_updated
    ON inference_session_kv_residency(job_id, updated_at DESC, residency_id DESC);

CREATE INDEX IF NOT EXISTS idx_inference_session_kv_residency_session_replica
    ON inference_session_kv_residency(session_id, replica_device_id, updated_at DESC);
