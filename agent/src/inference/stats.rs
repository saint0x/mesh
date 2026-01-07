//! Inference statistics tracking
//!
//! This module provides metrics tracking for distributed inference operations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tracing::info;

/// Statistics for inference operations
#[derive(Debug)]
pub struct InferenceStats {
    /// Total number of inference jobs completed
    pub jobs_completed: AtomicU64,

    /// Total number of inference jobs failed
    pub jobs_failed: AtomicU64,

    /// Total tokens generated across all jobs
    pub total_tokens_generated: AtomicU64,

    /// Total prompt tokens processed
    pub total_prompt_tokens: AtomicU64,

    /// Total inference time in milliseconds
    pub total_inference_time_ms: AtomicU64,

    /// Total all-reduce time in milliseconds
    pub total_allreduce_time_ms: AtomicU64,

    /// Number of checkpoints created
    pub checkpoints_created: AtomicU64,

    /// Number of recoveries from checkpoint
    pub checkpoint_recoveries: AtomicU64,

    /// Start time for uptime tracking
    pub start_time: Instant,

    /// Number of ring all-reduce operations performed
    pub allreduce_operations: AtomicU64,

    /// Total layers processed (for averaging)
    pub total_layers_processed: AtomicU64,
}

impl Default for InferenceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceStats {
    /// Create new inference statistics tracker
    pub fn new() -> Self {
        Self {
            jobs_completed: AtomicU64::new(0),
            jobs_failed: AtomicU64::new(0),
            total_tokens_generated: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_inference_time_ms: AtomicU64::new(0),
            total_allreduce_time_ms: AtomicU64::new(0),
            checkpoints_created: AtomicU64::new(0),
            checkpoint_recoveries: AtomicU64::new(0),
            start_time: Instant::now(),
            allreduce_operations: AtomicU64::new(0),
            total_layers_processed: AtomicU64::new(0),
        }
    }

    /// Record a successful inference job
    pub fn record_success(
        &self,
        prompt_tokens: u32,
        completion_tokens: u32,
        inference_time_ms: u64,
    ) {
        self.jobs_completed.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.total_inference_time_ms
            .fetch_add(inference_time_ms, Ordering::Relaxed);
    }

    /// Record a failed inference job
    pub fn record_failure(&self) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an all-reduce operation
    pub fn record_allreduce(&self, duration_ms: u64) {
        self.allreduce_operations.fetch_add(1, Ordering::Relaxed);
        self.total_allreduce_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Record a layer processed
    pub fn record_layer(&self) {
        self.total_layers_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a checkpoint creation
    pub fn record_checkpoint(&self) {
        self.checkpoints_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a recovery from checkpoint
    pub fn record_recovery(&self) {
        self.checkpoint_recoveries.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total jobs (completed + failed)
    pub fn total_jobs(&self) -> u64 {
        self.jobs_completed.load(Ordering::Relaxed) + self.jobs_failed.load(Ordering::Relaxed)
    }

    /// Get success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f64 {
        let total = self.total_jobs();
        if total == 0 {
            return 0.0;
        }
        self.jobs_completed.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get average tokens per second
    pub fn avg_tokens_per_second(&self) -> f64 {
        let total_time_s = self.total_inference_time_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        if total_time_s == 0.0 {
            return 0.0;
        }
        self.total_tokens_generated.load(Ordering::Relaxed) as f64 / total_time_s
    }

    /// Get average all-reduce latency in milliseconds
    pub fn avg_allreduce_latency_ms(&self) -> f64 {
        let ops = self.allreduce_operations.load(Ordering::Relaxed);
        if ops == 0 {
            return 0.0;
        }
        self.total_allreduce_time_ms.load(Ordering::Relaxed) as f64 / ops as f64
    }

    /// Get average inference time per job in milliseconds
    pub fn avg_inference_time_ms(&self) -> f64 {
        let jobs = self.jobs_completed.load(Ordering::Relaxed);
        if jobs == 0 {
            return 0.0;
        }
        self.total_inference_time_ms.load(Ordering::Relaxed) as f64 / jobs as f64
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Format uptime as human-readable string
    pub fn uptime_string(&self) -> String {
        let seconds = self.uptime_seconds();
        let days = seconds / 86400;
        let hours = (seconds % 86400) / 3600;
        let minutes = (seconds % 3600) / 60;
        let secs = seconds % 60;

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, secs)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, secs)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, secs)
        } else {
            format!("{}s", secs)
        }
    }

    /// Print statistics summary
    pub fn print_summary(&self) {
        let jobs_completed = self.jobs_completed.load(Ordering::Relaxed);
        let jobs_failed = self.jobs_failed.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens_generated.load(Ordering::Relaxed);
        let checkpoints = self.checkpoints_created.load(Ordering::Relaxed);
        let recoveries = self.checkpoint_recoveries.load(Ordering::Relaxed);

        info!(
            jobs_completed = jobs_completed,
            jobs_failed = jobs_failed,
            total_tokens = total_tokens,
            success_rate = format!("{:.1}%", self.success_rate() * 100.0),
            avg_tokens_per_second = format!("{:.2}", self.avg_tokens_per_second()),
            avg_allreduce_latency_ms = format!("{:.2}", self.avg_allreduce_latency_ms()),
            checkpoints_created = checkpoints,
            checkpoint_recoveries = recoveries,
            uptime = %self.uptime_string(),
            "Inference statistics"
        );
    }

    /// Display formatted metrics (for CLI)
    pub fn display(&self) {
        use colored::Colorize;

        println!("\n{}", "Inference Metrics".bold().cyan());
        println!("{}", "=================".cyan());

        println!("\n{}", "Job Statistics:".bold());
        println!(
            "  Completed:           {}",
            self.jobs_completed
                .load(Ordering::Relaxed)
                .to_string()
                .green()
        );
        println!(
            "  Failed:              {}",
            self.jobs_failed.load(Ordering::Relaxed).to_string().red()
        );
        println!("  Success Rate:        {:.1}%", self.success_rate() * 100.0);

        println!("\n{}", "Token Statistics:".bold());
        println!(
            "  Tokens Generated:    {}",
            self.total_tokens_generated.load(Ordering::Relaxed)
        );
        println!(
            "  Prompt Tokens:       {}",
            self.total_prompt_tokens.load(Ordering::Relaxed)
        );
        println!(
            "  Avg Tokens/sec:      {:.2}",
            self.avg_tokens_per_second()
        );

        println!("\n{}", "Ring Performance:".bold());
        println!(
            "  All-Reduce Ops:      {}",
            self.allreduce_operations.load(Ordering::Relaxed)
        );
        println!(
            "  Avg Latency:         {:.2}ms",
            self.avg_allreduce_latency_ms()
        );
        println!(
            "  Layers Processed:    {}",
            self.total_layers_processed.load(Ordering::Relaxed)
        );

        println!("\n{}", "Fault Tolerance:".bold());
        println!(
            "  Checkpoints:         {}",
            self.checkpoints_created.load(Ordering::Relaxed)
        );
        println!(
            "  Recoveries:          {}",
            self.checkpoint_recoveries.load(Ordering::Relaxed)
        );

        println!("\n{}", "System:".bold());
        println!("  Uptime:              {}", self.uptime_string());
        println!();
    }

    /// Serialize to JSON for persistence
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "jobs_completed": self.jobs_completed.load(Ordering::Relaxed),
            "jobs_failed": self.jobs_failed.load(Ordering::Relaxed),
            "total_tokens_generated": self.total_tokens_generated.load(Ordering::Relaxed),
            "total_prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
            "total_inference_time_ms": self.total_inference_time_ms.load(Ordering::Relaxed),
            "total_allreduce_time_ms": self.total_allreduce_time_ms.load(Ordering::Relaxed),
            "allreduce_operations": self.allreduce_operations.load(Ordering::Relaxed),
            "total_layers_processed": self.total_layers_processed.load(Ordering::Relaxed),
            "checkpoints_created": self.checkpoints_created.load(Ordering::Relaxed),
            "checkpoint_recoveries": self.checkpoint_recoveries.load(Ordering::Relaxed),
            "success_rate": self.success_rate(),
            "avg_tokens_per_second": self.avg_tokens_per_second(),
            "avg_allreduce_latency_ms": self.avg_allreduce_latency_ms(),
            "uptime": self.uptime_string(),
            "last_updated": chrono::Local::now().to_rfc3339(),
        })
    }

    /// Save statistics to file
    pub fn save_to_file(&self) -> std::io::Result<()> {
        let stats_path = dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".meshnet")
            .join("inference_stats.json");

        if let Some(parent) = stats_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.to_json())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        std::fs::write(&stats_path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_new() {
        let stats = InferenceStats::new();
        assert_eq!(stats.total_jobs(), 0);
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.avg_tokens_per_second(), 0.0);
    }

    #[test]
    fn test_stats_record_success() {
        let stats = InferenceStats::new();

        stats.record_success(10, 50, 1000);
        stats.record_success(20, 100, 2000);

        assert_eq!(stats.jobs_completed.load(Ordering::Relaxed), 2);
        assert_eq!(stats.total_tokens_generated.load(Ordering::Relaxed), 150);
        assert_eq!(stats.total_prompt_tokens.load(Ordering::Relaxed), 30);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.avg_tokens_per_second(), 50.0); // 150 tokens / 3 seconds
    }

    #[test]
    fn test_stats_record_failure() {
        let stats = InferenceStats::new();

        stats.record_success(10, 50, 1000);
        stats.record_failure();

        assert_eq!(stats.total_jobs(), 2);
        assert_eq!(stats.success_rate(), 0.5);
    }

    #[test]
    fn test_stats_allreduce() {
        let stats = InferenceStats::new();

        stats.record_allreduce(10);
        stats.record_allreduce(20);
        stats.record_allreduce(30);

        assert_eq!(stats.allreduce_operations.load(Ordering::Relaxed), 3);
        assert_eq!(stats.avg_allreduce_latency_ms(), 20.0);
    }

    #[test]
    fn test_stats_checkpoints() {
        let stats = InferenceStats::new();

        stats.record_checkpoint();
        stats.record_checkpoint();
        stats.record_recovery();

        assert_eq!(stats.checkpoints_created.load(Ordering::Relaxed), 2);
        assert_eq!(stats.checkpoint_recoveries.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stats_to_json() {
        let stats = InferenceStats::new();
        stats.record_success(10, 50, 1000);

        let json = stats.to_json();

        assert_eq!(json["jobs_completed"], 1);
        assert_eq!(json["total_tokens_generated"], 50);
    }
}
