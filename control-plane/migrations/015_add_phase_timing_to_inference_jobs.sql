ALTER TABLE inference_jobs
    ADD COLUMN time_to_first_token_ms INTEGER;

ALTER TABLE inference_jobs
    ADD COLUMN prefill_completed_at TEXT;
