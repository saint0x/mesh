//! Inference job types for distributed inference
//!
//! This module defines the types for inference requests, results, and configuration.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: u32,

    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative)
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,

    /// Stop sequences (generation stops when any of these are produced)
    pub stop_sequences: Vec<String>,

    /// Whether to stream tokens as they're generated
    pub stream: bool,

    /// Checkpoint interval (save state every N tokens)
    pub checkpoint_interval: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: vec![],
            stream: false,
            checkpoint_interval: 50,
        }
    }
}

/// An inference request from an executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique job identifier
    pub job_id: Uuid,

    /// Network/pool this job belongs to
    pub network_id: String,

    /// Model to use for inference
    pub model_id: String,

    /// Input prompt tokens (already tokenized)
    pub prompt_tokens: Vec<u32>,

    /// Generation configuration
    pub config: GenerationConfig,

    /// Executor ID (for credit tracking)
    pub executor_id: String,

    /// Unix timestamp when request was created
    pub created_at: u64,
}

impl InferenceRequest {
    /// Create a new inference request
    pub fn new(
        network_id: String,
        model_id: String,
        prompt_tokens: Vec<u32>,
        executor_id: String,
    ) -> Self {
        Self {
            job_id: Uuid::new_v4(),
            network_id,
            model_id,
            prompt_tokens,
            config: GenerationConfig::default(),
            executor_id,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Set the generation configuration
    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }
}

/// Result of an inference job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Job ID this result corresponds to
    pub job_id: Uuid,

    /// Whether inference completed successfully
    pub success: bool,

    /// Generated tokens (if successful)
    pub generated_tokens: Option<Vec<u32>>,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Number of prompt tokens processed
    pub prompt_tokens: u32,

    /// Number of tokens generated
    pub completion_tokens: u32,

    /// Total execution time in milliseconds
    pub execution_time_ms: u64,

    /// Time to first token in milliseconds (TTFT)
    pub time_to_first_token_ms: Option<u64>,

    /// Average tokens per second
    pub tokens_per_second: Option<f32>,

    /// Whether the result was recovered from a checkpoint
    pub recovered_from_checkpoint: bool,
}

impl InferenceResult {
    /// Create a successful result
    pub fn success(
        job_id: Uuid,
        generated_tokens: Vec<u32>,
        prompt_tokens: u32,
        execution_time_ms: u64,
    ) -> Self {
        let completion_tokens = generated_tokens.len() as u32;
        let tokens_per_second = if execution_time_ms > 0 {
            Some((completion_tokens as f32) / (execution_time_ms as f32 / 1000.0))
        } else {
            None
        };

        Self {
            job_id,
            success: true,
            generated_tokens: Some(generated_tokens),
            error: None,
            prompt_tokens,
            completion_tokens,
            execution_time_ms,
            time_to_first_token_ms: None,
            tokens_per_second,
            recovered_from_checkpoint: false,
        }
    }

    /// Create a failed result
    pub fn failure(job_id: Uuid, error: String, execution_time_ms: u64) -> Self {
        Self {
            job_id,
            success: false,
            generated_tokens: None,
            error: Some(error),
            prompt_tokens: 0,
            completion_tokens: 0,
            execution_time_ms,
            time_to_first_token_ms: None,
            tokens_per_second: None,
            recovered_from_checkpoint: false,
        }
    }

    /// Mark as recovered from checkpoint
    pub fn with_recovery(mut self) -> Self {
        self.recovered_from_checkpoint = true;
        self
    }

    /// Set time to first token
    pub fn with_ttft(mut self, ttft_ms: u64) -> Self {
        self.time_to_first_token_ms = Some(ttft_ms);
        self
    }
}

/// State of an active inference job
#[derive(Debug, Clone)]
pub struct InferenceJob {
    /// The original request
    pub request: InferenceRequest,

    /// Current position in token generation
    pub current_token_idx: u32,

    /// Tokens generated so far
    pub generated_tokens: Vec<u32>,

    /// Start time of inference
    pub start_time: std::time::Instant,

    /// Time when first token was generated
    pub first_token_time: Option<std::time::Instant>,

    /// Last checkpoint token index
    pub last_checkpoint_idx: u32,

    /// Current layer being processed (for forward pass tracking)
    pub current_layer: u32,

    /// Total number of layers in the model
    pub total_layers: u32,
}

impl InferenceJob {
    /// Create a new inference job from a request
    pub fn new(request: InferenceRequest, total_layers: u32) -> Self {
        Self {
            request,
            current_token_idx: 0,
            generated_tokens: Vec::new(),
            start_time: std::time::Instant::now(),
            first_token_time: None,
            last_checkpoint_idx: 0,
            current_layer: 0,
            total_layers,
        }
    }

    /// Record a generated token
    pub fn add_token(&mut self, token: u32) {
        if self.generated_tokens.is_empty() {
            self.first_token_time = Some(std::time::Instant::now());
        }
        self.generated_tokens.push(token);
        self.current_token_idx += 1;
    }

    /// Check if generation is complete
    pub fn is_complete(&self) -> bool {
        self.current_token_idx >= self.request.config.max_tokens
    }

    /// Check if a checkpoint should be created
    pub fn should_checkpoint(&self) -> bool {
        let interval = self.request.config.checkpoint_interval;
        interval > 0 && self.current_token_idx > 0
            && self.current_token_idx % interval == 0
            && self.current_token_idx > self.last_checkpoint_idx
    }

    /// Mark checkpoint as taken
    pub fn mark_checkpointed(&mut self) {
        self.last_checkpoint_idx = self.current_token_idx;
    }

    /// Get time to first token
    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        self.first_token_time.map(|t| t.duration_since(self.start_time))
    }

    /// Get total elapsed time
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Get progress as a percentage
    pub fn progress(&self) -> f32 {
        (self.current_token_idx as f32 / self.request.config.max_tokens as f32) * 100.0
    }

    /// Convert to result
    pub fn into_result(self) -> InferenceResult {
        let execution_time_ms = self.elapsed().as_millis() as u64;
        let ttft = self.time_to_first_token();
        let prompt_len = self.request.prompt_tokens.len() as u32;

        let mut result = InferenceResult::success(
            self.request.job_id,
            self.generated_tokens,
            prompt_len,
            execution_time_ms,
        );

        if let Some(ttft) = ttft {
            result = result.with_ttft(ttft.as_millis() as u64);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.checkpoint_interval, 50);
    }

    #[test]
    fn test_inference_request_new() {
        let request = InferenceRequest::new(
            "network-1".to_string(),
            "llama-70b".to_string(),
            vec![1, 2, 3, 4, 5],
            "executor-1".to_string(),
        );

        assert_eq!(request.network_id, "network-1");
        assert_eq!(request.model_id, "llama-70b");
        assert_eq!(request.prompt_tokens.len(), 5);
    }

    #[test]
    fn test_inference_result_success() {
        let job_id = Uuid::new_v4();
        let result = InferenceResult::success(
            job_id,
            vec![100, 101, 102],
            10,
            1000,
        );

        assert!(result.success);
        assert_eq!(result.completion_tokens, 3);
        assert_eq!(result.prompt_tokens, 10);
        assert!(result.tokens_per_second.is_some());
    }

    #[test]
    fn test_inference_result_failure() {
        let job_id = Uuid::new_v4();
        let result = InferenceResult::failure(job_id, "OOM error".to_string(), 500);

        assert!(!result.success);
        assert!(result.error.is_some());
        assert!(result.generated_tokens.is_none());
    }

    #[test]
    fn test_inference_job_lifecycle() {
        let request = InferenceRequest::new(
            "network-1".to_string(),
            "llama-70b".to_string(),
            vec![1, 2, 3],
            "executor-1".to_string(),
        ).with_config(GenerationConfig {
            max_tokens: 5,
            checkpoint_interval: 2,
            ..Default::default()
        });

        let mut job = InferenceJob::new(request, 70);

        // Add tokens
        assert!(!job.is_complete());
        assert!(!job.should_checkpoint());

        job.add_token(100);
        assert!(!job.should_checkpoint());

        job.add_token(101);
        assert!(job.should_checkpoint());
        job.mark_checkpointed();

        job.add_token(102);
        job.add_token(103);
        assert!(job.should_checkpoint());
        job.mark_checkpointed();

        job.add_token(104);
        assert!(job.is_complete());

        // Convert to result
        let result = job.into_result();
        assert!(result.success);
        assert_eq!(result.completion_tokens, 5);
    }

    #[test]
    fn test_inference_job_progress() {
        let request = InferenceRequest::new(
            "net".to_string(),
            "model".to_string(),
            vec![1],
            "exec".to_string(),
        ).with_config(GenerationConfig {
            max_tokens: 10,
            ..Default::default()
        });

        let mut job = InferenceJob::new(request, 70);
        assert_eq!(job.progress(), 0.0);

        job.add_token(100);
        assert_eq!(job.progress(), 10.0);

        for _ in 0..4 {
            job.add_token(100);
        }
        assert_eq!(job.progress(), 50.0);
    }
}
