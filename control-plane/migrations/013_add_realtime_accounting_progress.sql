ALTER TABLE inference_jobs
    ADD COLUMN accounted_completion_tokens INTEGER NOT NULL DEFAULT 0;

ALTER TABLE inference_jobs
    ADD COLUMN prompt_credits_accounted INTEGER NOT NULL DEFAULT 0;

ALTER TABLE inference_job_assignments
    ADD COLUMN reported_completion_tokens INTEGER NOT NULL DEFAULT 0;
