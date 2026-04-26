CREATE TABLE IF NOT EXISTS inference_regroup_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES inference_sessions(session_id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES inference_jobs(job_id) ON DELETE CASCADE,
    network_id TEXT NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    group_id TEXT NOT NULL,
    device_id TEXT,
    event_kind TEXT NOT NULL,
    reason TEXT,
    previous_status TEXT,
    new_status TEXT,
    segment_id TEXT,
    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inference_regroup_events_network_observed
    ON inference_regroup_events(network_id, observed_at DESC, event_id DESC);

CREATE INDEX IF NOT EXISTS idx_inference_regroup_events_job_observed
    ON inference_regroup_events(job_id, observed_at DESC, event_id DESC);

CREATE INDEX IF NOT EXISTS idx_inference_regroup_events_session_observed
    ON inference_regroup_events(session_id, observed_at DESC, event_id DESC);

CREATE TRIGGER IF NOT EXISTS trg_serving_groups_regroup_insert
AFTER INSERT ON inference_serving_groups
BEGIN
    INSERT INTO inference_regroup_events (
        session_id, job_id, network_id, model_id, phase, group_id, device_id,
        event_kind, reason, previous_status, new_status, segment_id, observed_at
    ) VALUES (
        NEW.session_id, NEW.job_id, NEW.network_id, NEW.model_id, NEW.phase, NEW.group_id, NEW.device_id,
        'member_added',
        CASE
            WHEN NEW.status = 'standby' THEN 'awaiting_decode_membership'
            WHEN NEW.status = 'superseded' THEN 'plan_superseded'
            ELSE NULL
        END,
        NULL,
        NEW.status,
        NULL,
        NEW.updated_at
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_serving_groups_regroup_update
AFTER UPDATE OF status, ring_position, shard_column_start, shard_column_end, assigned_capacity_units,
               execution_provider, lease_owner_device_id, lease_expires_at, updated_at
ON inference_serving_groups
WHEN OLD.status IS NOT NEW.status
   OR OLD.ring_position IS NOT NEW.ring_position
   OR OLD.shard_column_start IS NOT NEW.shard_column_start
   OR OLD.shard_column_end IS NOT NEW.shard_column_end
   OR OLD.assigned_capacity_units IS NOT NEW.assigned_capacity_units
   OR OLD.execution_provider IS NOT NEW.execution_provider
   OR OLD.lease_owner_device_id IS NOT NEW.lease_owner_device_id
   OR OLD.lease_expires_at IS NOT NEW.lease_expires_at
BEGIN
    INSERT INTO inference_regroup_events (
        session_id, job_id, network_id, model_id, phase, group_id, device_id,
        event_kind, reason, previous_status, new_status, segment_id, observed_at
    ) VALUES (
        NEW.session_id, NEW.job_id, NEW.network_id, NEW.model_id, NEW.phase, NEW.group_id, NEW.device_id,
        CASE
            WHEN OLD.lease_owner_device_id IS NOT NEW.lease_owner_device_id
              OR OLD.lease_expires_at IS NOT NEW.lease_expires_at THEN 'lease_changed'
            WHEN OLD.status IS NOT NEW.status THEN 'member_status_changed'
            ELSE 'membership_rebalanced'
        END,
        CASE
            WHEN NEW.status = 'superseded' THEN 'plan_superseded'
            WHEN NEW.status LIKE 'decode_pending_%' THEN 'kv_transfer_pending'
            WHEN NEW.status = 'standby' THEN 'awaiting_decode_membership'
            ELSE NULL
        END,
        OLD.status,
        NEW.status,
        NULL,
        NEW.updated_at
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_serving_groups_regroup_delete
AFTER DELETE ON inference_serving_groups
BEGIN
    INSERT INTO inference_regroup_events (
        session_id, job_id, network_id, model_id, phase, group_id, device_id,
        event_kind, reason, previous_status, new_status, segment_id, observed_at
    ) VALUES (
        OLD.session_id, OLD.job_id, OLD.network_id, OLD.model_id, OLD.phase, OLD.group_id, OLD.device_id,
        'member_removed',
        'group_row_removed',
        OLD.status,
        NULL,
        NULL,
        COALESCE(OLD.updated_at, CURRENT_TIMESTAMP)
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_decode_queue_regroup_update
AFTER UPDATE OF group_id, status, blocked_reason, lease_owner_device_id, lease_expires_at, updated_at
ON inference_decode_queue
WHEN OLD.group_id IS NOT NEW.group_id
   OR OLD.status IS NOT NEW.status
   OR OLD.lease_owner_device_id IS NOT NEW.lease_owner_device_id
   OR OLD.lease_expires_at IS NOT NEW.lease_expires_at
BEGIN
    INSERT INTO inference_regroup_events (
        session_id, job_id, network_id, model_id, phase, group_id, device_id,
        event_kind, reason, previous_status, new_status, segment_id, observed_at
    )
    SELECT
        NEW.session_id,
        NEW.job_id,
        NEW.network_id,
        s.model_id,
        'decode',
        NEW.group_id,
        NEW.lease_owner_device_id,
        CASE
            WHEN OLD.group_id IS NOT NEW.group_id THEN 'queue_regrouped'
            WHEN OLD.lease_owner_device_id IS NOT NEW.lease_owner_device_id
              OR OLD.lease_expires_at IS NOT NEW.lease_expires_at THEN 'queue_lease_changed'
            ELSE 'queue_status_changed'
        END,
        CASE
            WHEN NEW.status LIKE 'blocked_on_%' THEN COALESCE(NEW.blocked_reason, substr(NEW.status, 12))
            ELSE NULL
        END,
        OLD.status,
        NEW.status,
        NEW.segment_id,
        NEW.updated_at
    FROM inference_sessions s
    WHERE s.session_id = NEW.session_id;
END;
