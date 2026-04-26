ALTER TABLE inference_sessions
    ADD COLUMN latest_batch_kv_tokens INTEGER;

ALTER TABLE inference_sessions
    ADD COLUMN latest_deferred_decode_sessions INTEGER;
