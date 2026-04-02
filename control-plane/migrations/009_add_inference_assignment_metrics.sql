ALTER TABLE inference_job_assignments
    ADD COLUMN execution_time_ms INTEGER NOT NULL DEFAULT 0;
