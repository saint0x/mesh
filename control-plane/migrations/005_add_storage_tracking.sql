-- Add storage contribution tracking to devices
ALTER TABLE devices ADD COLUMN contributed_storage INTEGER;

-- Add storage to resource_locks table
ALTER TABLE resource_locks ADD COLUMN storage_bytes INTEGER DEFAULT 0;
