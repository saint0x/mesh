ALTER TABLE inference_job_assignments
    ADD COLUMN active_segment_id TEXT;

ALTER TABLE inference_job_assignments
    ADD COLUMN last_completed_segment_id TEXT;

ALTER TABLE inference_job_assignments
    ADD COLUMN segment_completed_at TEXT;

CREATE INDEX IF NOT EXISTS idx_inference_assignments_device_segment_status
    ON inference_job_assignments(device_id, network_id, active_segment_id, status, assigned_at ASC);
