ALTER TABLE devices ADD COLUMN direct_candidates TEXT;

CREATE INDEX IF NOT EXISTS idx_devices_direct_candidates ON devices(direct_candidates);
