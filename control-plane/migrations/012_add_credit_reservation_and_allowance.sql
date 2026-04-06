ALTER TABLE inference_jobs
    ADD COLUMN reserved_credits REAL NOT NULL DEFAULT 0;

ALTER TABLE inference_jobs
    ADD COLUMN settled_credits REAL NOT NULL DEFAULT 0;

ALTER TABLE inference_jobs
    ADD COLUMN released_credits REAL NOT NULL DEFAULT 0;

ALTER TABLE inference_jobs
    ADD COLUMN available_completion_tokens INTEGER NOT NULL DEFAULT 0;

ALTER TABLE inference_jobs
    ADD COLUMN model_size_factor REAL NOT NULL DEFAULT 1;
