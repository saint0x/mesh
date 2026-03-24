ALTER TABLE devices ADD COLUMN peer_id TEXT;
ALTER TABLE devices ADD COLUMN listen_addrs TEXT;

CREATE INDEX IF NOT EXISTS idx_devices_peer_id ON devices(peer_id);
