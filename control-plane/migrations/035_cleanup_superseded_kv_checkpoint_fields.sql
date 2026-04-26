ALTER TABLE inference_sessions
    DROP COLUMN kv_checkpoint_device_id;

ALTER TABLE inference_sessions
    DROP COLUMN kv_checkpoint_created_at;

ALTER TABLE inference_session_replicas
    DROP COLUMN checkpoint_created_at;
