//! Resource Manager - Cross-platform memory locking with cooldown enforcement
//!
//! This module provides memory locking functionality for pool contribution with:
//! - Cross-platform support (Linux, macOS, Windows)
//! - 24-hour cooldown period before unlock is allowed
//! - Configuration persistence across restarts
//! - 7% safety buffer on locked memory
//! - Automatic re-lock on restart if memory was previously locked

use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tracing::{info, warn};

/// Configuration for resource locking, persisted to disk
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResourceLockConfig {
    /// User-requested allocation in bytes
    pub user_allocated: u64,
    /// Actual locked memory (with buffer) in bytes
    pub locked_memory: u64,
    /// Timestamp when memory was locked (as seconds since UNIX_EPOCH)
    pub lock_timestamp: Option<u64>,
    /// Cooldown period in hours
    pub cooldown_hours: u64,
}

impl Default for ResourceLockConfig {
    fn default() -> Self {
        Self {
            user_allocated: 0,
            locked_memory: 0,
            lock_timestamp: None,
            cooldown_hours: 24,
        }
    }
}

/// Resource Manager for memory locking and cooldown enforcement
pub struct ResourceManager {
    /// Total system memory in bytes
    total_memory: u64,
    /// User-requested allocation in bytes
    user_allocated: u64,
    /// Actual locked memory (with 7% buffer) in bytes
    locked_memory: u64,
    /// Timestamp when memory was locked
    lock_timestamp: Option<SystemTime>,
    /// Cooldown period before unlock is allowed
    cooldown_period: Duration,
    /// Path to configuration file
    config_path: PathBuf,
    /// Handle to locked memory (kept alive to maintain lock)
    #[allow(dead_code)]
    locked_buffer: Option<Vec<u8>>,
}

impl ResourceManager {
    /// Create a new ResourceManager with detected system memory
    pub fn new() -> Result<Self> {
        let total_memory = Self::detect_total_memory()?;
        let config_path = dirs::home_dir()
            .ok_or_else(|| AgentError::Config("Home directory not found".into()))?
            .join(".meshnet")
            .join("resource-lock.toml");

        Ok(Self {
            total_memory,
            user_allocated: 0,
            locked_memory: 0,
            lock_timestamp: None,
            cooldown_period: Duration::from_secs(24 * 60 * 60), // 24 hours
            config_path,
            locked_buffer: None,
        })
    }

    /// Create a ResourceManager with custom config path (for testing)
    #[cfg(test)]
    pub fn with_config_path(config_path: PathBuf) -> Result<Self> {
        let total_memory = Self::detect_total_memory()?;
        Ok(Self {
            total_memory,
            user_allocated: 0,
            locked_memory: 0,
            lock_timestamp: None,
            cooldown_period: Duration::from_secs(24 * 60 * 60),
            config_path,
            locked_buffer: None,
        })
    }

    /// Detect total system memory using sysinfo
    /// Uses refresh_memory() for better performance instead of new_all()
    fn detect_total_memory() -> Result<u64> {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        Ok(sys.total_memory())
    }

    /// Get total system memory
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    /// Get user-requested allocation
    pub fn user_allocated(&self) -> u64 {
        self.user_allocated
    }

    /// Get actual locked memory (with buffer)
    pub fn locked_memory(&self) -> u64 {
        self.locked_memory
    }

    /// Check if memory is physically pinned in RAM
    pub fn is_memory_pinned(&self) -> bool {
        self.locked_buffer.is_some()
    }

    /// Set allocation amount with 7% safety buffer
    pub fn set_allocation(&mut self, bytes: u64) -> Result<()> {
        if bytes > self.total_memory {
            return Err(AgentError::Resource(format!(
                "Requested {} bytes exceeds total memory {} bytes",
                bytes, self.total_memory
            )));
        }

        // Add 7% safety buffer
        let buffer = (bytes as f64 * 1.07) as u64;

        // Ensure buffered amount doesn't exceed total memory
        if buffer > self.total_memory {
            return Err(AgentError::Resource(format!(
                "Allocation with 7% buffer ({} bytes) exceeds total memory {} bytes",
                buffer, self.total_memory
            )));
        }

        self.user_allocated = bytes;
        self.locked_memory = buffer;

        info!(
            "Set allocation: user={} GB, locked={} GB (with 7% buffer)",
            bytes / 1_000_000_000,
            buffer / 1_000_000_000
        );

        Ok(())
    }

    /// Check if there is a lock commitment (may or may not be physically pinned)
    pub fn is_locked(&self) -> bool {
        self.lock_timestamp.is_some()
    }

    /// Get the lock timestamp
    pub fn lock_timestamp(&self) -> Option<SystemTime> {
        self.lock_timestamp
    }

    /// Calculate time remaining until unlock is allowed
    pub fn time_until_unlock(&self) -> Option<Duration> {
        self.lock_timestamp.and_then(|ts| {
            let elapsed = SystemTime::now().duration_since(ts).ok()?;
            self.cooldown_period.checked_sub(elapsed)
        })
    }

    /// Save configuration to disk using atomic write
    pub fn save_config(&self) -> Result<()> {
        // Convert timestamp to seconds, error if before epoch
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

        let config = ResourceLockConfig {
            user_allocated: self.user_allocated,
            locked_memory: self.locked_memory,
            lock_timestamp,
            cooldown_hours: self.cooldown_period.as_secs() / 3600,
        };

        let toml_str = toml::to_string_pretty(&config)
            .map_err(|e| AgentError::Config(e.to_string()))?;

        // Ensure parent directory exists
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Atomic write: write to temp file, then rename
        // On Windows, we need to remove the destination first if it exists
        let temp_path = self.config_path.with_extension("tmp");
        std::fs::write(&temp_path, toml_str)?;

        #[cfg(target_os = "windows")]
        {
            // Windows doesn't support atomic rename over existing file
            if self.config_path.exists() {
                std::fs::remove_file(&self.config_path)?;
            }
        }

        std::fs::rename(&temp_path, &self.config_path)?;

        info!("Saved resource lock config to {:?}", self.config_path);

        Ok(())
    }

    /// Load configuration from disk and re-lock memory if previously locked
    pub fn load_config(&mut self) -> Result<()> {
        if !self.config_path.exists() {
            info!("No resource lock config found, using defaults");
            return Ok(()); // No config yet, use defaults
        }

        let toml_str = std::fs::read_to_string(&self.config_path)?;
        let config: ResourceLockConfig = toml::from_str(&toml_str)
            .map_err(|e| AgentError::Config(format!("Invalid config file: {}", e)))?;

        self.user_allocated = config.user_allocated;
        self.locked_memory = config.locked_memory;
        self.lock_timestamp = config.lock_timestamp.map(|secs| {
            SystemTime::UNIX_EPOCH + Duration::from_secs(secs)
        });
        self.cooldown_period = Duration::from_secs(config.cooldown_hours * 3600);

        info!(
            "Loaded resource lock config: allocated={} GB, locked={}",
            self.user_allocated / 1_000_000_000,
            self.is_locked()
        );

        // If config indicates locked state but memory isn't pinned, re-lock it
        if self.is_locked() && !self.is_memory_pinned() && self.locked_memory > 0 {
            info!("Re-locking memory after restart...");
            if let Err(e) = self.relock_memory() {
                warn!(
                    "Failed to re-lock memory on startup: {}. Lock commitment remains but memory is not pinned.",
                    e
                );
                // Don't fail - the commitment still exists, just warn
            }
        }

        Ok(())
    }
}

// Platform-specific memory locking for Linux and macOS
#[cfg(any(target_os = "linux", target_os = "macos"))]
impl ResourceManager {
    /// Internal function to lock memory without modifying timestamp
    fn relock_memory(&mut self) -> Result<()> {
        // Check for 32-bit overflow
        if self.locked_memory > usize::MAX as u64 {
            return Err(AgentError::Resource(format!(
                "Allocation {} bytes exceeds maximum addressable memory on this platform ({} bytes)",
                self.locked_memory,
                usize::MAX
            )));
        }

        let size = self.locked_memory as usize;
        let mut buffer = vec![0u8; size];

        unsafe {
            let ptr = buffer.as_mut_ptr() as *mut libc::c_void;
            let result = libc::mlock(ptr, size);

            if result != 0 {
                let err = std::io::Error::last_os_error();
                return Err(AgentError::Resource(format!("mlock failed: {}", err)));
            }
        }

        self.locked_buffer = Some(buffer);
        info!("Re-locked {} GB memory", self.locked_memory / 1_000_000_000);

        Ok(())
    }

    /// Lock memory pages using mlock
    ///
    /// # Safety
    /// This function uses unsafe code to call libc::mlock. The memory buffer
    /// is allocated by Rust and remains valid for the lifetime of the ResourceManager.
    /// The mlock call pins pages in physical memory, preventing them from being swapped.
    pub fn lock_memory(&mut self) -> Result<()> {
        if self.is_locked() {
            return Err(AgentError::Resource("Memory already locked".into()));
        }

        if self.locked_memory == 0 {
            return Err(AgentError::Resource(
                "No allocation set. Call set_allocation first".into(),
            ));
        }

        // Check for 32-bit overflow
        if self.locked_memory > usize::MAX as u64 {
            return Err(AgentError::Resource(format!(
                "Allocation {} bytes exceeds maximum addressable memory on this platform ({} bytes)",
                self.locked_memory,
                usize::MAX
            )));
        }

        let size = self.locked_memory as usize;

        // Allocate memory region
        let mut buffer = vec![0u8; size];

        // Pin pages with mlock
        // SAFETY: buffer is a valid allocation of `size` bytes, and we're passing
        // a valid pointer and size to mlock. The buffer is kept alive in self.locked_buffer
        // to maintain the lock.
        unsafe {
            let ptr = buffer.as_mut_ptr() as *mut libc::c_void;
            let result = libc::mlock(ptr, size);

            if result != 0 {
                let err = std::io::Error::last_os_error();
                return Err(AgentError::Resource(format!("mlock failed: {}", err)));
            }
        }

        self.lock_timestamp = Some(SystemTime::now());
        self.locked_buffer = Some(buffer);
        self.save_config()?;

        info!("Locked {} GB memory", self.locked_memory / 1_000_000_000);

        Ok(())
    }

    /// Unlock memory pages using munlock
    ///
    /// # Safety
    /// This function uses unsafe code to call libc::munlock. The memory buffer
    /// must have been previously locked with mlock.
    pub fn unlock_memory(&mut self) -> Result<()> {
        if !self.is_locked() {
            return Err(AgentError::Resource("Memory is not locked".into()));
        }

        // Check cooldown - round up to avoid confusion
        if let Some(remaining) = self.time_until_unlock() {
            let remaining_hours = remaining.as_secs().div_ceil(3600);
            return Err(AgentError::CooldownActive { remaining_hours });
        }

        // Unlock memory if we have a buffer
        if let Some(ref buffer) = self.locked_buffer {
            let size = buffer.len();
            // SAFETY: buffer was previously locked with mlock, and we're passing
            // the same pointer and size to munlock.
            unsafe {
                let ptr = buffer.as_ptr() as *const libc::c_void;
                let result = libc::munlock(ptr, size);

                if result != 0 {
                    let err = std::io::Error::last_os_error();
                    return Err(AgentError::Resource(format!("munlock failed: {}", err)));
                }
            }
        } else {
            // Memory wasn't physically pinned (e.g., after restart without successful re-lock)
            // This is fine - we still clear the commitment
            info!("Memory was not physically pinned, clearing lock commitment only");
        }

        self.lock_timestamp = None;
        self.locked_buffer = None;
        self.save_config()?;

        info!("Unlocked memory");
        Ok(())
    }
}

// Platform-specific memory locking for Windows
#[cfg(target_os = "windows")]
impl ResourceManager {
    /// Internal function to lock memory without modifying timestamp
    fn relock_memory(&mut self) -> Result<()> {
        // Check for 32-bit overflow
        if self.locked_memory > usize::MAX as u64 {
            return Err(AgentError::Resource(format!(
                "Allocation {} bytes exceeds maximum addressable memory on this platform ({} bytes)",
                self.locked_memory,
                usize::MAX
            )));
        }

        let size = self.locked_memory as usize;
        let mut buffer = vec![0u8; size];

        unsafe {
            use windows_sys::Win32::System::Memory::VirtualLock;

            let ptr = buffer.as_mut_ptr() as *mut std::ffi::c_void;
            let result = VirtualLock(ptr, size);

            if result == 0 {
                let err = std::io::Error::last_os_error();
                return Err(AgentError::Resource(format!("VirtualLock failed: {}", err)));
            }
        }

        self.locked_buffer = Some(buffer);
        info!("Re-locked {} GB memory", self.locked_memory / 1_000_000_000);

        Ok(())
    }

    /// Lock memory pages using VirtualLock
    ///
    /// # Safety
    /// This function uses unsafe code to call VirtualLock from windows-sys.
    /// The memory buffer is allocated by Rust and remains valid for the lifetime
    /// of the ResourceManager.
    pub fn lock_memory(&mut self) -> Result<()> {
        if self.is_locked() {
            return Err(AgentError::Resource("Memory already locked".into()));
        }

        if self.locked_memory == 0 {
            return Err(AgentError::Resource(
                "No allocation set. Call set_allocation first".into(),
            ));
        }

        // Check for 32-bit overflow
        if self.locked_memory > usize::MAX as u64 {
            return Err(AgentError::Resource(format!(
                "Allocation {} bytes exceeds maximum addressable memory on this platform ({} bytes)",
                self.locked_memory,
                usize::MAX
            )));
        }

        let size = self.locked_memory as usize;

        // Allocate memory region
        let mut buffer = vec![0u8; size];

        // Lock pages with VirtualLock
        // SAFETY: buffer is a valid allocation of `size` bytes, and we're passing
        // a valid pointer and size to VirtualLock.
        unsafe {
            use windows_sys::Win32::System::Memory::VirtualLock;

            let ptr = buffer.as_mut_ptr() as *mut std::ffi::c_void;
            let result = VirtualLock(ptr, size);

            if result == 0 {
                let err = std::io::Error::last_os_error();
                return Err(AgentError::Resource(format!("VirtualLock failed: {}", err)));
            }
        }

        self.lock_timestamp = Some(SystemTime::now());
        self.locked_buffer = Some(buffer);
        self.save_config()?;

        info!("Locked {} GB memory", self.locked_memory / 1_000_000_000);

        Ok(())
    }

    /// Unlock memory pages using VirtualUnlock
    ///
    /// # Safety
    /// This function uses unsafe code to call VirtualUnlock from windows-sys.
    pub fn unlock_memory(&mut self) -> Result<()> {
        if !self.is_locked() {
            return Err(AgentError::Resource("Memory is not locked".into()));
        }

        // Check cooldown - round up to avoid confusion
        if let Some(remaining) = self.time_until_unlock() {
            let remaining_hours = remaining.as_secs().div_ceil(3600);
            return Err(AgentError::CooldownActive { remaining_hours });
        }

        // Unlock memory if we have a buffer
        if let Some(ref buffer) = self.locked_buffer {
            let size = buffer.len();
            // SAFETY: buffer was previously locked with VirtualLock
            unsafe {
                use windows_sys::Win32::System::Memory::VirtualUnlock;

                let ptr = buffer.as_ptr() as *const std::ffi::c_void;
                let result = VirtualUnlock(ptr as *mut _, size);

                if result == 0 {
                    let err = std::io::Error::last_os_error();
                    return Err(AgentError::Resource(format!("VirtualUnlock failed: {}", err)));
                }
            }
        } else {
            // Memory wasn't physically pinned (e.g., after restart without successful re-lock)
            // This is fine - we still clear the commitment
            info!("Memory was not physically pinned, clearing lock commitment only");
        }

        self.lock_timestamp = None;
        self.locked_buffer = None;
        self.save_config()?;

        info!("Unlocked memory");
        Ok(())
    }
}

/// Parse a memory string like "7GB" or raw bytes
///
/// # Units
/// This function uses **decimal (SI) units**:
/// - 1 GB = 1,000,000,000 bytes (10^9)
/// - 1 MB = 1,000,000 bytes (10^6)
/// - 1 KB = 1,000 bytes (10^3)
///
/// For binary units (GiB, MiB, KiB), use the explicit suffixes.
pub fn parse_memory_string(s: &str) -> Result<u64> {
    let s = s.trim().to_uppercase();

    // Binary units (IEC standard)
    if let Some(gib_str) = s.strip_suffix("GIB") {
        let gib: f64 = gib_str.trim().parse().map_err(|_| {
            AgentError::Config(format!("Invalid memory format: {}", s))
        })?;
        return Ok((gib * 1_073_741_824.0) as u64); // 2^30
    } else if let Some(mib_str) = s.strip_suffix("MIB") {
        let mib: f64 = mib_str.trim().parse().map_err(|_| {
            AgentError::Config(format!("Invalid memory format: {}", s))
        })?;
        return Ok((mib * 1_048_576.0) as u64); // 2^20
    } else if let Some(kib_str) = s.strip_suffix("KIB") {
        let kib: f64 = kib_str.trim().parse().map_err(|_| {
            AgentError::Config(format!("Invalid memory format: {}", s))
        })?;
        return Ok((kib * 1_024.0) as u64); // 2^10
    }

    // Decimal units (SI standard)
    if let Some(gb_str) = s.strip_suffix("GB") {
        let gb: f64 = gb_str.trim().parse().map_err(|_| {
            AgentError::Config(format!("Invalid memory format: {}", s))
        })?;
        Ok((gb * 1_000_000_000.0) as u64)
    } else if let Some(mb_str) = s.strip_suffix("MB") {
        let mb: f64 = mb_str.trim().parse().map_err(|_| {
            AgentError::Config(format!("Invalid memory format: {}", s))
        })?;
        Ok((mb * 1_000_000.0) as u64)
    } else if let Some(kb_str) = s.strip_suffix("KB") {
        let kb: f64 = kb_str.trim().parse().map_err(|_| {
            AgentError::Config(format!("Invalid memory format: {}", s))
        })?;
        Ok((kb * 1_000.0) as u64)
    } else {
        // Assume raw bytes
        s.parse::<u64>().map_err(|_| {
            AgentError::Config(format!(
                "Invalid memory format: {}. Use format like '7GB', '7GiB', '512MB', or bytes",
                s
            ))
        })
    }
}

/// Format bytes as human-readable string (using SI decimal units)
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_memory_detection() {
        let manager = ResourceManager::new().unwrap();
        // Should detect some memory (at least 1GB on any modern system)
        assert!(manager.total_memory() >= 1_000_000_000);
    }

    #[test]
    fn test_allocation_with_buffer() {
        let mut manager = ResourceManager::new().unwrap();
        let alloc = 1_000_000_000; // 1GB

        manager.set_allocation(alloc).unwrap();

        assert_eq!(manager.user_allocated(), alloc);
        // Should have 7% buffer
        let expected_locked = (alloc as f64 * 1.07) as u64;
        assert_eq!(manager.locked_memory(), expected_locked);
    }

    #[test]
    fn test_allocation_exceeds_total() {
        let mut manager = ResourceManager::new().unwrap();
        let huge_alloc = u64::MAX;

        let result = manager.set_allocation(huge_alloc);
        assert!(result.is_err());

        if let Err(AgentError::Resource(msg)) = result {
            assert!(msg.contains("exceeds total memory"));
        } else {
            panic!("Expected Resource error");
        }
    }

    #[test]
    fn test_time_until_unlock() {
        let mut manager = ResourceManager::new().unwrap();

        // Not locked, should return None
        assert!(manager.time_until_unlock().is_none());

        // Manually set a recent lock timestamp
        manager.lock_timestamp = Some(SystemTime::now());
        manager.cooldown_period = Duration::from_secs(24 * 60 * 60);

        // Should have ~24 hours remaining
        let remaining = manager.time_until_unlock().unwrap();
        assert!(remaining.as_secs() > 23 * 60 * 60);
        assert!(remaining.as_secs() <= 24 * 60 * 60);
    }

    #[test]
    fn test_time_until_unlock_expired() {
        let mut manager = ResourceManager::new().unwrap();

        // Set lock timestamp to 25 hours ago
        manager.lock_timestamp = Some(SystemTime::now() - Duration::from_secs(25 * 60 * 60));
        manager.cooldown_period = Duration::from_secs(24 * 60 * 60);

        // Cooldown should be expired
        assert!(manager.time_until_unlock().is_none());
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("resource-lock.toml");

        // Create and configure manager
        let mut manager = ResourceManager::with_config_path(config_path.clone()).unwrap();
        manager.set_allocation(2_000_000_000).unwrap(); // 2GB
        manager.lock_timestamp = Some(SystemTime::now());
        manager.save_config().unwrap();

        // Load in new manager
        let mut loaded = ResourceManager::with_config_path(config_path).unwrap();
        loaded.load_config().unwrap();

        assert_eq!(loaded.user_allocated(), manager.user_allocated());
        assert_eq!(loaded.locked_memory(), manager.locked_memory());
        assert!(loaded.is_locked());
    }

    #[test]
    fn test_config_missing() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("nonexistent.toml");

        let mut manager = ResourceManager::with_config_path(config_path).unwrap();
        // Should not error on missing config
        assert!(manager.load_config().is_ok());
        assert!(!manager.is_locked());
    }

    #[test]
    fn test_config_invalid() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("invalid.toml");

        // Write invalid TOML
        std::fs::write(&config_path, "this is not valid toml {{{").unwrap();

        let mut manager = ResourceManager::with_config_path(config_path).unwrap();
        let result = manager.load_config();

        assert!(result.is_err());
        if let Err(AgentError::Config(msg)) = result {
            assert!(msg.contains("Invalid config file"));
        } else {
            panic!("Expected Config error");
        }
    }

    #[test]
    fn test_parse_memory_string_gb() {
        assert_eq!(parse_memory_string("7GB").unwrap(), 7_000_000_000);
        assert_eq!(parse_memory_string("7.5GB").unwrap(), 7_500_000_000);
        assert_eq!(parse_memory_string("7 GB").unwrap(), 7_000_000_000);
        assert_eq!(parse_memory_string("7gb").unwrap(), 7_000_000_000);
    }

    #[test]
    fn test_parse_memory_string_gib() {
        assert_eq!(parse_memory_string("1GiB").unwrap(), 1_073_741_824);
        assert_eq!(parse_memory_string("7GiB").unwrap(), 7_516_192_768);
    }

    #[test]
    fn test_parse_memory_string_mb() {
        assert_eq!(parse_memory_string("512MB").unwrap(), 512_000_000);
        assert_eq!(parse_memory_string("1024MB").unwrap(), 1_024_000_000);
    }

    #[test]
    fn test_parse_memory_string_mib() {
        assert_eq!(parse_memory_string("1MiB").unwrap(), 1_048_576);
        assert_eq!(parse_memory_string("512MiB").unwrap(), 536_870_912);
    }

    #[test]
    fn test_parse_memory_string_kb() {
        assert_eq!(parse_memory_string("1024KB").unwrap(), 1_024_000);
    }

    #[test]
    fn test_parse_memory_string_kib() {
        assert_eq!(parse_memory_string("1024KiB").unwrap(), 1_048_576);
    }

    #[test]
    fn test_parse_memory_string_bytes() {
        assert_eq!(parse_memory_string("1000000000").unwrap(), 1_000_000_000);
    }

    #[test]
    fn test_parse_memory_string_invalid() {
        assert!(parse_memory_string("invalid").is_err());
        assert!(parse_memory_string("7XB").is_err());
        assert!(parse_memory_string("").is_err());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(7_000_000_000), "7.0 GB");
        assert_eq!(format_bytes(512_000_000), "512.0 MB");
        assert_eq!(format_bytes(1_024_000), "1.0 MB");
        assert_eq!(format_bytes(1_000), "1.0 KB");
        assert_eq!(format_bytes(500), "500 bytes");
    }

    #[test]
    fn test_is_locked() {
        let mut manager = ResourceManager::new().unwrap();

        assert!(!manager.is_locked());

        manager.lock_timestamp = Some(SystemTime::now());
        assert!(manager.is_locked());
    }

    #[test]
    fn test_is_memory_pinned() {
        let manager = ResourceManager::new().unwrap();
        assert!(!manager.is_memory_pinned());
    }

    // Note: Actual lock/unlock tests are platform-specific and may require
    // elevated privileges. These are tested manually or in integration tests.
    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_lock_without_allocation() {
        let mut manager = ResourceManager::new().unwrap();

        let result = manager.lock_memory();
        assert!(result.is_err());

        if let Err(AgentError::Resource(msg)) = result {
            assert!(msg.contains("No allocation set"));
        } else {
            panic!("Expected Resource error");
        }
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn test_double_lock_prevented() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("resource-lock.toml");

        let mut manager = ResourceManager::with_config_path(config_path).unwrap();
        manager.set_allocation(1_000_000).unwrap(); // Small allocation for test

        // First lock should succeed (may fail due to permissions, that's ok)
        if manager.lock_memory().is_ok() {
            // Second lock should fail
            let result = manager.lock_memory();
            assert!(result.is_err());

            if let Err(AgentError::Resource(msg)) = result {
                assert!(msg.contains("already locked"));
            } else {
                panic!("Expected Resource error");
            }
        }
    }
}
