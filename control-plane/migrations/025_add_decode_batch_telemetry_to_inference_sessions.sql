ALTER TABLE inference_sessions
    ADD COLUMN latest_batch_size INTEGER;

ALTER TABLE inference_sessions
    ADD COLUMN latest_active_decode_sessions INTEGER;
