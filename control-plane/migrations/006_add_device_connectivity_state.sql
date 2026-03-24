ALTER TABLE devices ADD COLUMN connectivity_state TEXT;

CREATE INDEX IF NOT EXISTS idx_devices_connectivity_state ON devices(connectivity_state);
