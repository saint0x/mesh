ALTER TABLE inference_job_assignments
    ADD COLUMN backend_contract_json TEXT;

ALTER TABLE inference_job_assignments
    ADD COLUMN backend_contract_hash TEXT;

ALTER TABLE inference_serving_groups
    ADD COLUMN execution_island_id TEXT NOT NULL DEFAULT '';

ALTER TABLE inference_serving_groups
    ADD COLUMN compatibility_class TEXT NOT NULL DEFAULT 'heterogeneous_portable';

ALTER TABLE inference_serving_groups
    ADD COLUMN backend_contract_hash TEXT;

ALTER TABLE inference_serving_groups
    ADD COLUMN fast_path_eligible INTEGER NOT NULL DEFAULT 0;

ALTER TABLE inference_serving_groups
    ADD COLUMN protocol_class TEXT NOT NULL DEFAULT 'provider_heterogeneous_portable_ring';

ALTER TABLE inference_serving_groups
    ADD COLUMN backend_contract_json TEXT;
