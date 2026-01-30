//! Storage Manager - Cross-platform disk space reservation with cooldown enforcement
//!
//! Mirrors the ResourceManager pattern for memory locking but for persistent storage.
//! Reserves disk space for model shards, KV caches, and checkpoints using:
//! - Pre-allocated sparse files (fallocate on Linux, ftruncate elsewhere)
//! - 24-hour cooldown period before deallocation is allowed
//! - Configuration persistence across restarts
//! - 7% safety buffer on reserved storage
//! - Automatic re-reservation on restart if storage was previously reserved

use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tracing::{info, warn};

/// Configuration for storage reservation, persisted to disk
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StorageLockConfig {
    /// User-requested storage in bytes
    pub user_allocated: u64,
    /// Actual reserved storage (with buffer) in bytes
    pub reserved_storage: u64,
    /// Timestamp when storage was reserved (as seconds since UNIX_EPOCH)
    pub lock_timestamp: Option<u64>,
    /// Cooldown period in hours
    pub cooldown_hours: u64,
    /// Path to the reservation directory
    pub reservation_dir: String,
}

impl Default for StorageLockConfig {
    fn default() -> Self {
        Self {
            user_allocated: 0,
            reserved_storage: 0,
            lock_timestamp: None,
            cooldown_hours: 24,
            reservation_dir: String::new(),
        }
    }
}

/// Manifest tracking individual reserved files within the storage allocation
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct StorageManifest {
    /// Files currently occupying reserved space
    pub entries: Vec<StorageEntry>,
}

/// A single file tracked within the storage reservation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StorageEntry {
    /// Model or checkpoint ID this entry belongs to
    pub owner_id: String,
    /// Type of stored data
    pub entry_type: StorageEntryType,
    /// Relative path within reservation directory
    pub relative_path: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// When this entry was created (Unix timestamp)
    pub created_at: u64,
}

/// Types of data that can be stored in reserved space
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum StorageEntryType {
    /// Model weight shard
    ModelShard,
    /// KV cache snapshot
    KvCache,
    /// Inference checkpoint
    Checkpoint,
    /// Pre-allocated space (placeholder)
    Reservation,
}

/// Storage Manager for disk space reservation and cooldown enforcement
pub struct StorageManager {
    /// Total available disk space in bytes
    available_space: u64,
    /// User-requested allocation in bytes
    user_allocated: u64,
    /// Actual reserved storage (with 7% buffer) in bytes
    reserved_storage: u64,
    /// Timestamp when storage was reserved
    lock_timestamp: Option<SystemTime>,
    /// Cooldown period before deallocation is allowed
    cooldown_period: Duration,
    /// Path to configuration file
    config_path: PathBuf,
    /// Directory where reserved storage files live
    reservation_dir: PathBuf,
    /// Manifest tracking reserved files
    manifest: StorageManifest,
    /// Whether the reservation file exists on disk
    reservation_active: bool,
}

impl StorageManager {
    /// Create a new StorageManager with detected available disk space
    pub fn new() -> Result<Self> {
        let meshnet_dir = dirs::home_dir()
            .ok_or_else(|| AgentError::Config("Home directory not found".into()))?
            .join(".meshnet");

        let config_path = meshnet_dir.join("storage-lock.toml");
        let reservation_dir = meshnet_dir.join("reserved");
        let available_space = Self::detect_available_space(&meshnet_dir)?;

        Ok(Self {
            available_space,
            user_allocated: 0,
            reserved_storage: 0,
            lock_timestamp: None,
            cooldown_period: Duration::from_secs(24 * 60 * 60),
            config_path,
            reservation_dir,
            manifest: StorageManifest::default(),
            reservation_active: false,
        })
    }

    /// Create a StorageManager with custom config path (for testing)
    #[cfg(test)]
    pub fn with_config_path(config_path: PathBuf, reservation_dir: PathBuf) -> Result<Self> {
        let available_space = Self::detect_available_space(
            config_path.parent().unwrap_or(std::path::Path::new("/")),
        )?;
        Ok(Self {
            available_space,
            user_allocated: 0,
            reserved_storage: 0,
            lock_timestamp: None,
            cooldown_period: Duration::from_secs(24 * 60 * 60),
            config_path,
            reservation_dir,
            manifest: StorageManifest::default(),
            reservation_active: false,
        })
    }

    /// Detect available disk space at the given path
    fn detect_available_space(path: &std::path::Path) -> Result<u64> {
        use sysinfo::Disks;
        let disks = Disks::new_with_refreshed_list();

        // Find the disk that contains our path
        let mut best_mount = None;
        let mut best_len = 0;

        for disk in disks.list() {
            let mount = disk.mount_point();
            let mount_str = mount.to_string_lossy();
            let path_str = path.to_string_lossy();

            if path_str.starts_with(mount_str.as_ref()) && mount_str.len() > best_len {
                best_mount = Some(disk.available_space());
                best_len = mount_str.len();
            }
        }

        best_mount.ok_or_else(|| {
            AgentError::Resource(format!(
                "Could not determine available disk space for path: {}",
                path.display()
            ))
        })
    }

    /// Get total available disk space
    pub fn available_space(&self) -> u64 {
        self.available_space
    }

    /// Get user-requested allocation
    pub fn user_allocated(&self) -> u64 {
        self.user_allocated
    }

    /// Get actual reserved storage (with buffer)
    pub fn reserved_storage(&self) -> u64 {
        self.reserved_storage
    }

    /// Check if storage is reserved
    pub fn is_locked(&self) -> bool {
        self.lock_timestamp.is_some()
    }

    /// Check if the reservation files exist on disk
    pub fn is_reservation_active(&self) -> bool {
        self.reservation_active
    }

    /// Get the lock timestamp
    pub fn lock_timestamp(&self) -> Option<SystemTime> {
        self.lock_timestamp
    }

    /// Get the reservation directory path
    pub fn reservation_dir(&self) -> &PathBuf {
        &self.reservation_dir
    }

    /// Get storage manifest (read-only)
    pub fn manifest(&self) -> &StorageManifest {
        &self.manifest
    }

    /// Calculate used space from manifest entries
    pub fn used_space(&self) -> u64 {
        self.manifest.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Calculate remaining space within the reservation
    pub fn remaining_space(&self) -> u64 {
        self.reserved_storage.saturating_sub(self.used_space())
    }

    /// Set allocation amount with 7% safety buffer
    pub fn set_allocation(&mut self, bytes: u64) -> Result<()> {
        if bytes > self.available_space {
            return Err(AgentError::Resource(format!(
                "Requested {} bytes exceeds available disk space {} bytes",
                bytes, self.available_space
            )));
        }

        let buffer = (bytes as f64 * 1.07) as u64;

        if buffer > self.available_space {
            return Err(AgentError::Resource(format!(
                "Allocation with 7% buffer ({} bytes) exceeds available disk space {} bytes",
                buffer, self.available_space
            )));
        }

        self.user_allocated = bytes;
        self.reserved_storage = buffer;

        info!(
            "Set storage allocation: user={} GB, reserved={} GB (with 7% buffer)",
            bytes / 1_000_000_000,
            buffer / 1_000_000_000
        );

        Ok(())
    }

    /// Calculate time remaining until unlock is allowed
    pub fn time_until_unlock(&self) -> Option<Duration> {
        self.lock_timestamp.and_then(|ts| {
            let elapsed = SystemTime::now().duration_since(ts).ok()?;
            self.cooldown_period.checked_sub(elapsed)
        })
    }

    /// Reserve storage on disk
    ///
    /// Creates the reservation directory and a pre-allocated space file.
    pub fn lock_storage(&mut self) -> Result<()> {
        if self.is_locked() {
            return Err(AgentError::Resource("Storage already reserved".into()));
        }

        if self.reserved_storage == 0 {
            return Err(AgentError::Resource(
                "No allocation set. Call set_allocation first".into(),
            ));
        }

        // Create reservation directory
        std::fs::create_dir_all(&self.reservation_dir)?;

        // Create pre-allocated space file
        let space_file = self.reservation_dir.join(".space_reservation");
        Self::preallocate_file(&space_file, self.reserved_storage)?;

        self.lock_timestamp = Some(SystemTime::now());
        self.reservation_active = true;

        // Add reservation entry to manifest
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.manifest.entries.push(StorageEntry {
            owner_id: "system".to_string(),
            entry_type: StorageEntryType::Reservation,
            relative_path: ".space_reservation".to_string(),
            size_bytes: self.reserved_storage,
            created_at: now,
        });

        self.save_config()?;
        self.save_manifest()?;

        info!(
            "Reserved {} GB storage at {}",
            self.reserved_storage / 1_000_000_000,
            self.reservation_dir.display()
        );

        Ok(())
    }

    /// Release reserved storage
    pub fn unlock_storage(&mut self) -> Result<()> {
        if !self.is_locked() {
            return Err(AgentError::Resource("Storage is not reserved".into()));
        }

        // Check cooldown
        if let Some(remaining) = self.time_until_unlock() {
            let remaining_hours = remaining.as_secs().div_ceil(3600);
            return Err(AgentError::CooldownActive { remaining_hours });
        }

        // Remove reservation file
        let space_file = self.reservation_dir.join(".space_reservation");
        if space_file.exists() {
            std::fs::remove_file(&space_file)?;
        }

        // Clean up empty reservation directory (only if no other files)
        if self.reservation_dir.exists() {
            let has_other_files = std::fs::read_dir(&self.reservation_dir)
                .map(|mut entries| entries.next().is_some())
                .unwrap_or(false);
            if !has_other_files {
                std::fs::remove_dir(&self.reservation_dir).ok();
            }
        }

        self.lock_timestamp = None;
        self.reservation_active = false;
        self.manifest = StorageManifest::default();
        self.save_config()?;
        self.save_manifest()?;

        info!("Released reserved storage");
        Ok(())
    }

    /// Register a file within the reserved storage space
    ///
    /// This tracks files that are stored within the reservation (model shards,
    /// checkpoints, etc.) against the total reserved budget.
    pub fn register_entry(
        &mut self,
        owner_id: String,
        entry_type: StorageEntryType,
        relative_path: String,
        size_bytes: u64,
    ) -> Result<()> {
        if !self.is_locked() {
            return Err(AgentError::Resource(
                "Storage not reserved. Lock storage first".into(),
            ));
        }

        // Check if adding this would exceed the reservation (excluding the reservation placeholder)
        let used_real: u64 = self
            .manifest
            .entries
            .iter()
            .filter(|e| e.entry_type != StorageEntryType::Reservation)
            .map(|e| e.size_bytes)
            .sum();

        if used_real + size_bytes > self.user_allocated {
            return Err(AgentError::Resource(format!(
                "Entry ({} bytes) would exceed storage reservation ({} bytes used of {} bytes allocated)",
                size_bytes, used_real, self.user_allocated
            )));
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.manifest.entries.push(StorageEntry {
            owner_id,
            entry_type,
            relative_path,
            size_bytes,
            created_at: now,
        });

        self.save_manifest()?;
        Ok(())
    }

    /// Remove a tracked entry from the manifest
    pub fn remove_entry(&mut self, relative_path: &str) -> Result<()> {
        let initial_len = self.manifest.entries.len();
        self.manifest
            .entries
            .retain(|e| e.relative_path != relative_path);

        if self.manifest.entries.len() == initial_len {
            return Err(AgentError::Resource(format!(
                "Entry not found in manifest: {}",
                relative_path
            )));
        }

        // Remove file from disk
        let full_path = self.reservation_dir.join(relative_path);
        if full_path.exists() {
            std::fs::remove_file(&full_path)?;
        }

        self.save_manifest()?;
        Ok(())
    }

    /// Get entries for a specific owner (model_id, job_id, etc.)
    pub fn entries_for_owner(&self, owner_id: &str) -> Vec<&StorageEntry> {
        self.manifest
            .entries
            .iter()
            .filter(|e| e.owner_id == owner_id)
            .collect()
    }

    /// Save configuration to disk using atomic write
    pub fn save_config(&self) -> Result<()> {
        let lock_timestamp = match self.lock_timestamp {
            Some(ts) => {
                let secs = ts
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map_err(|_| {
                        AgentError::Config("Lock timestamp is before UNIX epoch".into())
                    })?
                    .as_secs();
                Some(secs)
            }
            None => None,
        };

        let config = StorageLockConfig {
            user_allocated: self.user_allocated,
            reserved_storage: self.reserved_storage,
            lock_timestamp,
            cooldown_hours: self.cooldown_period.as_secs() / 3600,
            reservation_dir: self.reservation_dir.to_string_lossy().to_string(),
        };

        let toml_str = toml::to_string_pretty(&config)
            .map_err(|e| AgentError::Config(e.to_string()))?;

        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let temp_path = self.config_path.with_extension("tmp");
        std::fs::write(&temp_path, toml_str)?;

        #[cfg(target_os = "windows")]
        {
            if self.config_path.exists() {
                std::fs::remove_file(&self.config_path)?;
            }
        }

        std::fs::rename(&temp_path, &self.config_path)?;

        info!("Saved storage lock config to {:?}", self.config_path);
        Ok(())
    }

    /// Load configuration from disk and re-verify reservation if previously locked
    pub fn load_config(&mut self) -> Result<()> {
        if !self.config_path.exists() {
            info!("No storage lock config found, using defaults");
            return Ok(());
        }

        let toml_str = std::fs::read_to_string(&self.config_path)?;
        let config: StorageLockConfig = toml::from_str(&toml_str)
            .map_err(|e| AgentError::Config(format!("Invalid storage config file: {}", e)))?;

        self.user_allocated = config.user_allocated;
        self.reserved_storage = config.reserved_storage;
        self.lock_timestamp = config.lock_timestamp.map(|secs| {
            SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
        });
        self.cooldown_period = Duration::from_secs(config.cooldown_hours * 3600);

        if !config.reservation_dir.is_empty() {
            self.reservation_dir = PathBuf::from(&config.reservation_dir);
        }

        info!(
            "Loaded storage lock config: allocated={} GB, locked={}",
            self.user_allocated / 1_000_000_000,
            self.is_locked()
        );

        // Verify reservation file still exists
        if self.is_locked() {
            let space_file = self.reservation_dir.join(".space_reservation");
            if space_file.exists() {
                self.reservation_active = true;
                info!("Storage reservation verified on disk");
            } else {
                warn!(
                    "Storage reservation config exists but reservation file is missing. \
                     Attempting to re-create..."
                );
                std::fs::create_dir_all(&self.reservation_dir)?;
                match Self::preallocate_file(&space_file, self.reserved_storage) {
                    Ok(()) => {
                        self.reservation_active = true;
                        info!("Storage reservation re-created successfully");
                    }
                    Err(e) => {
                        warn!(
                            "Failed to re-create storage reservation: {}. \
                             Commitment remains but space is not reserved.",
                            e
                        );
                    }
                }
            }
        }

        // Load manifest
        self.load_manifest()?;

        Ok(())
    }

    /// Save manifest to disk
    fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.reservation_dir.join("manifest.json");

        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.manifest)?;
        std::fs::write(&manifest_path, json)?;

        Ok(())
    }

    /// Load manifest from disk
    fn load_manifest(&mut self) -> Result<()> {
        let manifest_path = self.reservation_dir.join("manifest.json");

        if !manifest_path.exists() {
            return Ok(());
        }

        let json = std::fs::read_to_string(&manifest_path)?;
        self.manifest = serde_json::from_str(&json)?;

        Ok(())
    }

    /// Pre-allocate a file to the given size using platform-optimal methods
    fn preallocate_file(path: &std::path::Path, size: u64) -> Result<()> {
        use std::fs::OpenOptions;

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            let result = unsafe { libc::fallocate(fd, 0, 0, size as libc::off_t) };
            if result != 0 {
                let err = std::io::Error::last_os_error();
                // Fall back to ftruncate if fallocate not supported (e.g., some filesystems)
                warn!(
                    "fallocate failed ({}), falling back to ftruncate",
                    err
                );
                file.set_len(size)?;
            }
        }

        #[cfg(target_os = "macos")]
        {
            // macOS doesn't have fallocate, use ftruncate (creates sparse file)
            file.set_len(size)?;
        }

        #[cfg(target_os = "windows")]
        {
            // Windows: set_len pre-allocates on NTFS
            file.set_len(size)?;
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            file.set_len(size)?;
        }

        info!(
            "Pre-allocated {} GB at {}",
            size / 1_000_000_000,
            path.display()
        );

        Ok(())
    }
}

/// Parse a storage size string like "50GB" or raw bytes
pub fn parse_storage_string(s: &str) -> Result<u64> {
    // Delegate to the memory parser since the format is identical
    crate::resource_manager::parse_memory_string(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_storage_detection() {
        let manager = StorageManager::new().unwrap();
        // Should detect some available space
        assert!(manager.available_space() > 0);
    }

    #[test]
    fn test_allocation_with_buffer() {
        let mut manager = StorageManager::new().unwrap();
        let alloc = 1_000_000_000; // 1GB

        manager.set_allocation(alloc).unwrap();

        assert_eq!(manager.user_allocated(), alloc);
        let expected = (alloc as f64 * 1.07) as u64;
        assert_eq!(manager.reserved_storage(), expected);
    }

    #[test]
    fn test_allocation_exceeds_available() {
        let mut manager = StorageManager::new().unwrap();
        let result = manager.set_allocation(u64::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_time_until_unlock() {
        let mut manager = StorageManager::new().unwrap();

        assert!(manager.time_until_unlock().is_none());

        manager.lock_timestamp = Some(SystemTime::now());
        manager.cooldown_period = Duration::from_secs(24 * 60 * 60);

        let remaining = manager.time_until_unlock().unwrap();
        assert!(remaining.as_secs() > 23 * 60 * 60);
        assert!(remaining.as_secs() <= 24 * 60 * 60);
    }

    #[test]
    fn test_time_until_unlock_expired() {
        let mut manager = StorageManager::new().unwrap();

        manager.lock_timestamp = Some(SystemTime::now() - Duration::from_secs(25 * 60 * 60));
        manager.cooldown_period = Duration::from_secs(24 * 60 * 60);

        assert!(manager.time_until_unlock().is_none());
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("storage-lock.toml");
        let reservation_dir = temp_dir.path().join("reserved");

        let mut manager =
            StorageManager::with_config_path(config_path.clone(), reservation_dir.clone()).unwrap();
        manager.set_allocation(2_000_000_000).unwrap();
        manager.lock_timestamp = Some(SystemTime::now());
        manager.save_config().unwrap();

        let mut loaded =
            StorageManager::with_config_path(config_path, reservation_dir).unwrap();
        loaded.load_config().unwrap();

        assert_eq!(loaded.user_allocated(), manager.user_allocated());
        assert_eq!(loaded.reserved_storage(), manager.reserved_storage());
        assert!(loaded.is_locked());
    }

    #[test]
    fn test_lock_unlock_storage() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("storage-lock.toml");
        let reservation_dir = temp_dir.path().join("reserved");

        let mut manager =
            StorageManager::with_config_path(config_path, reservation_dir.clone()).unwrap();
        // Use a small allocation for testing
        manager.user_allocated = 1_000;
        manager.reserved_storage = 1_070;

        manager.lock_storage().unwrap();
        assert!(manager.is_locked());
        assert!(manager.is_reservation_active());

        let space_file = reservation_dir.join(".space_reservation");
        assert!(space_file.exists());

        // Should fail with cooldown
        let result = manager.unlock_storage();
        assert!(result.is_err());

        // Override cooldown for test
        manager.lock_timestamp = Some(SystemTime::now() - Duration::from_secs(25 * 60 * 60));
        manager.unlock_storage().unwrap();
        assert!(!manager.is_locked());
        assert!(!space_file.exists());
    }

    #[test]
    fn test_double_lock_prevented() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("storage-lock.toml");
        let reservation_dir = temp_dir.path().join("reserved");

        let mut manager =
            StorageManager::with_config_path(config_path, reservation_dir).unwrap();
        manager.user_allocated = 1_000;
        manager.reserved_storage = 1_070;

        manager.lock_storage().unwrap();
        let result = manager.lock_storage();
        assert!(result.is_err());
    }

    #[test]
    fn test_register_entry() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("storage-lock.toml");
        let reservation_dir = temp_dir.path().join("reserved");

        let mut manager =
            StorageManager::with_config_path(config_path, reservation_dir).unwrap();
        manager.user_allocated = 10_000;
        manager.reserved_storage = 10_700;

        manager.lock_storage().unwrap();

        manager
            .register_entry(
                "llama-70b".to_string(),
                StorageEntryType::ModelShard,
                "llama-70b/shard_0.bin".to_string(),
                5_000,
            )
            .unwrap();

        assert_eq!(manager.manifest().entries.len(), 2); // reservation + shard
        let entries = manager.entries_for_owner("llama-70b");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].size_bytes, 5_000);
    }

    #[test]
    fn test_register_entry_exceeds_budget() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("storage-lock.toml");
        let reservation_dir = temp_dir.path().join("reserved");

        let mut manager =
            StorageManager::with_config_path(config_path, reservation_dir).unwrap();
        manager.user_allocated = 1_000;
        manager.reserved_storage = 1_070;

        manager.lock_storage().unwrap();

        let result = manager.register_entry(
            "llama-70b".to_string(),
            StorageEntryType::ModelShard,
            "shard.bin".to_string(),
            2_000, // exceeds user_allocated
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_entry() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("storage-lock.toml");
        let reservation_dir = temp_dir.path().join("reserved");

        let mut manager =
            StorageManager::with_config_path(config_path, reservation_dir.clone()).unwrap();
        manager.user_allocated = 10_000;
        manager.reserved_storage = 10_700;

        manager.lock_storage().unwrap();

        // Create a file on disk
        let shard_dir = reservation_dir.join("model");
        std::fs::create_dir_all(&shard_dir).unwrap();
        std::fs::write(shard_dir.join("shard.bin"), b"test").unwrap();

        manager
            .register_entry(
                "model".to_string(),
                StorageEntryType::ModelShard,
                "model/shard.bin".to_string(),
                4,
            )
            .unwrap();

        manager.remove_entry("model/shard.bin").unwrap();

        let entries = manager.entries_for_owner("model");
        assert!(entries.is_empty());
        assert!(!shard_dir.join("shard.bin").exists());
    }

    #[test]
    fn test_is_locked() {
        let mut manager = StorageManager::new().unwrap();
        assert!(!manager.is_locked());

        manager.lock_timestamp = Some(SystemTime::now());
        assert!(manager.is_locked());
    }

    #[test]
    fn test_parse_storage_string() {
        assert_eq!(parse_storage_string("50GB").unwrap(), 50_000_000_000);
        assert_eq!(parse_storage_string("100MB").unwrap(), 100_000_000);
    }
}
