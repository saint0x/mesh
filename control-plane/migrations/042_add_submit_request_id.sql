ALTER TABLE inference_jobs
    ADD COLUMN request_id TEXT;

UPDATE inference_jobs
SET request_id = job_id
WHERE request_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_jobs_request_id
    ON inference_jobs(request_id);
