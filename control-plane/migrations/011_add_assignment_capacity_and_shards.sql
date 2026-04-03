ALTER TABLE inference_job_assignments
    ADD COLUMN shard_column_start INTEGER NOT NULL DEFAULT 0;

ALTER TABLE inference_job_assignments
    ADD COLUMN shard_column_end INTEGER NOT NULL DEFAULT 0;

ALTER TABLE inference_job_assignments
    ADD COLUMN assigned_capacity_units INTEGER NOT NULL DEFAULT 1;

ALTER TABLE inference_job_assignments
    ADD COLUMN execution_provider TEXT;
