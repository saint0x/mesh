ALTER TABLE inference_decode_queue
    ADD COLUMN batch_group_key TEXT;

CREATE INDEX IF NOT EXISTS idx_inference_decode_queue_batch_group
    ON inference_decode_queue(network_id, batch_group_key, status, updated_at ASC);
