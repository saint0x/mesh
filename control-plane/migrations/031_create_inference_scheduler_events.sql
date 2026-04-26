CREATE TABLE IF NOT EXISTS inference_scheduler_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    job_id TEXT REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    session_id TEXT REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    device_id TEXT,
    segment_id TEXT,
    group_id TEXT,
    batch_group_key TEXT,
    event_kind TEXT NOT NULL,
    queue_status TEXT,
    detail TEXT,
    lease_target_session_count INTEGER,
    lease_target_batch_size INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_inference_scheduler_events_network_created
    ON inference_scheduler_events(network_id, created_at DESC, event_id DESC);

CREATE INDEX IF NOT EXISTS idx_inference_scheduler_events_job_created
    ON inference_scheduler_events(job_id, created_at DESC, event_id DESC);
