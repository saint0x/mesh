CREATE TABLE IF NOT EXISTS inference_jobs (
    job_id TEXT PRIMARY KEY,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    submitted_by_device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE RESTRICT,
    model_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_tokens TEXT NOT NULL,
    max_tokens INTEGER NOT NULL,
    temperature REAL NOT NULL,
    top_p REAL NOT NULL,
    status TEXT NOT NULL,
    ring_worker_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    completion TEXT,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    execution_time_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS inference_job_assignments (
    assignment_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    ring_position INTEGER NOT NULL,
    status TEXT NOT NULL,
    lease_expires_at TEXT,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    acknowledged_at TEXT,
    completed_at TEXT,
    failure_reason TEXT,
    UNIQUE(job_id, device_id)
);

CREATE INDEX IF NOT EXISTS idx_inference_jobs_network_created
    ON inference_jobs(network_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_jobs_status
    ON inference_jobs(status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_assignments_device_status
    ON inference_job_assignments(device_id, status, assigned_at ASC);

CREATE INDEX IF NOT EXISTS idx_inference_assignments_job
    ON inference_job_assignments(job_id, ring_position ASC);
