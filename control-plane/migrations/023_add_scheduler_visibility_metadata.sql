ALTER TABLE inference_decode_queue
    ADD COLUMN blocked_reason TEXT;

ALTER TABLE inference_decode_queue
    ADD COLUMN blocked_since TEXT;

ALTER TABLE inference_decode_queue
    ADD COLUMN block_detail TEXT;

ALTER TABLE inference_serving_groups
    ADD COLUMN lease_owner_device_id TEXT;

ALTER TABLE inference_serving_groups
    ADD COLUMN lease_expires_at TEXT;

CREATE INDEX IF NOT EXISTS idx_inference_decode_queue_network_blocked_reason
    ON inference_decode_queue(network_id, blocked_reason, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_inference_serving_groups_session_lease
    ON inference_serving_groups(session_id, group_id, lease_expires_at DESC);

UPDATE inference_decode_queue
SET blocked_reason = CASE
        WHEN status = 'blocked_on_prefill' THEN 'prefill_incomplete'
        WHEN status = 'blocked_on_kv_transfer' THEN 'kv_transfer_pending'
        WHEN status = 'blocked_on_regroup' THEN 'regroup_pending'
        WHEN status LIKE 'blocked_on_%' THEN substr(status, 12)
        ELSE NULL
    END,
    blocked_since = CASE
        WHEN status LIKE 'blocked_on_%' THEN COALESCE(blocked_since, updated_at)
        ELSE NULL
    END
WHERE blocked_reason IS NULL OR blocked_since IS NULL;

UPDATE inference_serving_groups
SET lease_owner_device_id = (
        SELECT dq.lease_owner_device_id
        FROM inference_decode_queue dq
        WHERE dq.session_id = inference_serving_groups.session_id
          AND dq.group_id = inference_serving_groups.group_id
    ),
    lease_expires_at = (
        SELECT dq.lease_expires_at
        FROM inference_decode_queue dq
        WHERE dq.session_id = inference_serving_groups.session_id
          AND dq.group_id = inference_serving_groups.group_id
    )
WHERE phase = 'decode';

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
