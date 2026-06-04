DROP TRIGGER IF EXISTS trg_decode_queue_scheduler_metadata_insert;
DROP TRIGGER IF EXISTS trg_decode_queue_scheduler_metadata_update;
DROP TRIGGER IF EXISTS trg_serving_groups_regroup_insert;
DROP TRIGGER IF EXISTS trg_serving_groups_regroup_update;
DROP TRIGGER IF EXISTS trg_serving_groups_regroup_delete;

CREATE TABLE inference_serving_groups_v2 (
    group_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    phase TEXT,
    support_role TEXT,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    ring_position INTEGER NOT NULL,
    shard_column_start INTEGER NOT NULL,
    shard_column_end INTEGER NOT NULL,
    assigned_capacity_units INTEGER NOT NULL,
    execution_provider TEXT NOT NULL,
    status TEXT NOT NULL,
    last_error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    lease_owner_device_id TEXT,
    lease_expires_at TEXT,
    execution_island_id TEXT,
    compatibility_class TEXT,
    backend_contract_hash TEXT,
    fast_path_eligible INTEGER NOT NULL DEFAULT 0,
    protocol_class TEXT,
    backend_contract_json TEXT,
    PRIMARY KEY (group_id, device_id),
    CHECK (phase IS NOT NULL OR support_role IS NOT NULL)
);

INSERT INTO inference_serving_groups_v2 (
    group_id, session_id, job_id, network_id, model_id, phase, support_role, device_id,
    ring_position, shard_column_start, shard_column_end, assigned_capacity_units,
    execution_provider, status, last_error, updated_at, lease_owner_device_id,
    lease_expires_at, execution_island_id, compatibility_class, backend_contract_hash,
    fast_path_eligible, protocol_class, backend_contract_json
)
SELECT
    group_id, session_id, job_id, network_id, model_id, phase, NULL, device_id,
    ring_position, shard_column_start, shard_column_end, assigned_capacity_units,
    execution_provider, status, last_error, updated_at, lease_owner_device_id,
    lease_expires_at, execution_island_id, compatibility_class, backend_contract_hash,
    fast_path_eligible, protocol_class, backend_contract_json
FROM inference_serving_groups;

DROP TABLE inference_serving_groups;
ALTER TABLE inference_serving_groups_v2 RENAME TO inference_serving_groups;

CREATE INDEX IF NOT EXISTS idx_inference_serving_groups_session_phase
    ON inference_serving_groups(session_id, phase, support_role, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_serving_groups_job
    ON inference_serving_groups(job_id, support_role, status, updated_at DESC);

CREATE TRIGGER IF NOT EXISTS trg_decode_queue_scheduler_metadata_insert
AFTER INSERT ON inference_decode_queue
BEGIN
    UPDATE inference_decode_queue
    SET blocked_reason = CASE
            WHEN NEW.status = 'blocked_on_prefill' THEN 'prefill_incomplete'
            WHEN NEW.status = 'blocked_on_kv_transfer' THEN 'kv_transfer_pending'
            WHEN NEW.status = 'blocked_on_regroup' THEN 'regroup_pending'
            WHEN NEW.status LIKE 'blocked_on_%' THEN substr(NEW.status, 12)
            ELSE NULL
        END,
        blocked_since = CASE
            WHEN NEW.status LIKE 'blocked_on_%' THEN COALESCE(NEW.blocked_since, NEW.updated_at)
            ELSE NULL
        END
    WHERE session_id = NEW.session_id;

    UPDATE inference_serving_groups
    SET lease_owner_device_id = NEW.lease_owner_device_id,
        lease_expires_at = NEW.lease_expires_at
    WHERE session_id = NEW.session_id
      AND group_id = NEW.group_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_decode_queue_scheduler_metadata_update
AFTER UPDATE OF status, ready_at, lease_owner_device_id, lease_expires_at, last_error, updated_at
ON inference_decode_queue
BEGIN
    UPDATE inference_decode_queue
    SET blocked_reason = CASE
            WHEN NEW.status = 'blocked_on_prefill' THEN 'prefill_incomplete'
            WHEN NEW.status = 'blocked_on_kv_transfer' THEN 'kv_transfer_pending'
            WHEN NEW.status = 'blocked_on_regroup' THEN 'regroup_pending'
            WHEN NEW.status LIKE 'blocked_on_%' THEN substr(NEW.status, 12)
            ELSE NULL
        END,
        blocked_since = CASE
            WHEN NEW.status LIKE 'blocked_on_%'
                THEN CASE
                    WHEN OLD.status LIKE 'blocked_on_%' THEN COALESCE(OLD.blocked_since, OLD.updated_at)
                    ELSE COALESCE(NEW.blocked_since, NEW.updated_at)
                END
            ELSE NULL
        END
    WHERE session_id = NEW.session_id;

    UPDATE inference_serving_groups
    SET lease_owner_device_id = NEW.lease_owner_device_id,
        lease_expires_at = NEW.lease_expires_at
    WHERE session_id = NEW.session_id
      AND group_id = NEW.group_id;
END;
