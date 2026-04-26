CREATE TABLE IF NOT EXISTS inference_decode_batch_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    completion_tokens INTEGER NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    batch_size INTEGER,
    active_decode_sessions INTEGER,
    batch_kv_tokens INTEGER,
    deferred_decode_sessions INTEGER,
    kv_cache_seq_len INTEGER,
    observed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_inference_decode_batch_events_session_observed
    ON inference_decode_batch_events(session_id, observed_at DESC, event_id DESC);

CREATE INDEX IF NOT EXISTS idx_inference_decode_batch_events_job_observed
    ON inference_decode_batch_events(job_id, observed_at DESC, event_id DESC);
