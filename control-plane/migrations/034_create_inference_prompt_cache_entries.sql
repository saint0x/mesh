CREATE TABLE IF NOT EXISTS inference_prompt_cache_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    prefix_token_count INTEGER NOT NULL,
    overlap_token_count INTEGER NOT NULL DEFAULT 0,
    prompt_tokens_json TEXT NOT NULL,
    owner_device_id TEXT NOT NULL,
    checkpoint_id TEXT,
    kv_sequence_position INTEGER,
    status TEXT NOT NULL,
    remote_access_uri TEXT,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_error TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_prompt_cache_entries_session
    ON inference_prompt_cache_entries(session_id);

CREATE INDEX IF NOT EXISTS idx_inference_prompt_cache_entries_network_model_status
    ON inference_prompt_cache_entries(network_id, model_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_prompt_cache_entries_cache_key
    ON inference_prompt_cache_entries(network_id, model_id, cache_key, status);
