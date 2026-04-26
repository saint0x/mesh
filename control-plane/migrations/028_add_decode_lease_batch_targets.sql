ALTER TABLE inference_decode_queue
    ADD COLUMN lease_target_session_count INTEGER;

ALTER TABLE inference_decode_queue
    ADD COLUMN lease_target_batch_size INTEGER;

CREATE INDEX IF NOT EXISTS idx_inference_decode_queue_group_status
    ON inference_decode_queue(network_id, group_id, status, updated_at ASC);
