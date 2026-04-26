ALTER TABLE inference_decode_batch_events
    ADD COLUMN target_session_count INTEGER;

ALTER TABLE inference_decode_batch_events
    ADD COLUMN target_batch_size INTEGER;
