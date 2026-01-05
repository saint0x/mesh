//! Embeddings workload executor
//!
//! This module provides text embedding generation. Currently uses a mock implementation
//! for testing the compute sharing infrastructure. The architecture is designed to easily
//! swap in real ONNX Runtime inference later.
//!
//! ## Mock vs. Real Implementation
//!
//! **Current (Mock)**:
//! - Returns deterministic fake embeddings based on text hash
//! - Simulates realistic latency (50-200ms)
//! - Validates input constraints (empty text, max length)
//! - Proper error handling and timeout support
//!
//! **Future (Real)**:
//! - ONNX Runtime with all-MiniLM-L6-v2 model
//! - Model downloading and caching
//! - Tokenization with HuggingFace tokenizers
//! - Mean pooling and L2 normalization
//!
//! The mock implementation allows us to test the distributed compute infrastructure
//! without the complexity of ML model management.

use crate::executor::types::{EmbeddingsInput, EmbeddingsOutput, ExecutorError, ExecutorResult};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;
use tracing::{debug, info, instrument, warn};

/// Default embedding dimensions (matches all-MiniLM-L6-v2)
const EMBEDDING_DIMENSIONS: usize = 384;

/// Maximum input text length (characters)
const MAX_INPUT_LENGTH: usize = 10_000;

/// Minimum simulated latency (milliseconds)
const MIN_LATENCY_MS: u64 = 50;

/// Maximum simulated latency (milliseconds)
const MAX_LATENCY_MS: u64 = 200;

/// Embeddings executor
///
/// **Current Implementation**: Mock executor that generates deterministic fake embeddings.
/// This allows testing the compute sharing network without actual ML inference.
///
/// **Design**: The API is designed to match what a real ONNX Runtime executor would provide,
/// making it easy to swap implementations later.
#[derive(Clone)]
pub struct EmbeddingsExecutor {
    model_name: String,
    dimensions: usize,
    simulate_latency: bool,
}

impl EmbeddingsExecutor {
    /// Create new embeddings executor with default configuration
    ///
    /// # Example
    /// ```
    /// use agent::EmbeddingsExecutor;
    ///
    /// let executor = EmbeddingsExecutor::new().unwrap();
    /// ```
    pub fn new() -> ExecutorResult<Self> {
        Ok(Self {
            model_name: "mock-all-MiniLM-L6-v2".to_string(),
            dimensions: EMBEDDING_DIMENSIONS,
            simulate_latency: true,
        })
    }

    /// Create executor without latency simulation (for testing)
    #[cfg(test)]
    pub fn new_fast() -> ExecutorResult<Self> {
        Ok(Self {
            model_name: "mock-all-MiniLM-L6-v2".to_string(),
            dimensions: EMBEDDING_DIMENSIONS,
            simulate_latency: false,
        })
    }

    /// Execute embeddings inference
    ///
    /// Generates a 384-dimensional embedding vector for the input text.
    /// The embedding is deterministic based on the text content (uses hash).
    ///
    /// # Arguments
    /// * `input` - Text input to generate embeddings for
    ///
    /// # Returns
    /// * `Ok(EmbeddingsOutput)` - Generated embedding vector
    /// * `Err(ExecutorError)` - If input validation fails or execution errors
    ///
    /// # Example
    /// ```
    /// use agent::{EmbeddingsExecutor, EmbeddingsInput};
    ///
    /// # tokio_test::block_on(async {
    /// let executor = EmbeddingsExecutor::new().unwrap();
    /// let input = EmbeddingsInput::new("Hello world");
    /// let output = executor.execute(&input).await.unwrap();
    /// assert_eq!(output.dimensions, 384);
    /// # })
    /// ```
    #[instrument(skip(self, input), fields(text_len = input.text.len()))]
    pub async fn execute(&self, input: &EmbeddingsInput) -> ExecutorResult<EmbeddingsOutput> {
        // Validate input
        self.validate_input(input)?;

        let text = &input.text;
        debug!(text_len = text.len(), "Generating mock embedding");

        // Simulate realistic latency
        if self.simulate_latency {
            let latency = self.calculate_latency(text);
            debug!(latency_ms = latency, "Simulating inference latency");
            tokio::time::sleep(Duration::from_millis(latency)).await;
        }

        // Generate deterministic embedding based on text hash
        let embedding = self.generate_mock_embedding(text);

        info!(
            model = %self.model_name,
            dimensions = self.dimensions,
            "Embedding generated successfully"
        );

        Ok(EmbeddingsOutput::new(embedding, &self.model_name))
    }

    /// Execute with timeout
    ///
    /// Wrapper around `execute()` that enforces a maximum execution time.
    ///
    /// # Arguments
    /// * `input` - Text input to generate embeddings for
    /// * `timeout_ms` - Maximum execution time in milliseconds
    ///
    /// # Returns
    /// * `Ok(EmbeddingsOutput)` - Generated embedding vector
    /// * `Err(ExecutorError::Timeout)` - If execution exceeds timeout
    ///
    /// # Example
    /// ```
    /// use agent::{EmbeddingsExecutor, EmbeddingsInput};
    ///
    /// # tokio_test::block_on(async {
    /// let executor = EmbeddingsExecutor::new().unwrap();
    /// let input = EmbeddingsInput::new("Hello world");
    /// let output = executor.execute_with_timeout(&input, 5000).await.unwrap();
    /// # })
    /// ```
    #[instrument(skip(self, input), fields(timeout_ms = timeout_ms))]
    pub async fn execute_with_timeout(
        &self,
        input: &EmbeddingsInput,
        timeout_ms: u64,
    ) -> ExecutorResult<EmbeddingsOutput> {
        match tokio::time::timeout(Duration::from_millis(timeout_ms), self.execute(input)).await {
            Ok(result) => result,
            Err(_) => {
                warn!(timeout_ms = timeout_ms, "Execution timed out");
                Err(ExecutorError::Timeout(timeout_ms))
            }
        }
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Validate input constraints
    fn validate_input(&self, input: &EmbeddingsInput) -> ExecutorResult<()> {
        if input.text.is_empty() {
            return Err(ExecutorError::InvalidInput("Text cannot be empty".into()));
        }

        if input.text.len() > MAX_INPUT_LENGTH {
            return Err(ExecutorError::InvalidInput(format!(
                "Text too long: {} characters (max: {})",
                input.text.len(),
                MAX_INPUT_LENGTH
            )));
        }

        Ok(())
    }

    /// Calculate simulated latency based on text length
    ///
    /// Real ML inference latency typically scales with input length.
    /// This simulates that behavior: longer text = longer inference time.
    fn calculate_latency(&self, text: &str) -> u64 {
        let text_len = text.len();

        // Base latency + length-proportional component
        let base = MIN_LATENCY_MS;
        let length_factor = ((text_len as f64 / 100.0) * 10.0) as u64;
        let latency = base + length_factor;

        // Clamp to realistic range
        latency.min(MAX_LATENCY_MS)
    }

    /// Generate deterministic mock embedding based on text hash
    ///
    /// Uses a hash of the text to generate a reproducible embedding vector.
    /// This makes testing easier (same input = same output) while still
    /// producing different embeddings for different inputs.
    ///
    /// The embedding values are normalized to roughly [-1.0, 1.0] range,
    /// similar to real embeddings.
    fn generate_mock_embedding(&self, text: &str) -> Vec<f32> {
        // Hash the text to get a seed
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate deterministic pseudo-random values
        let mut embedding = Vec::with_capacity(self.dimensions);
        let mut state = seed;

        for i in 0..self.dimensions {
            // Simple LCG (Linear Congruential Generator) for deterministic randomness
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(i as u64);

            // Map to [-1.0, 1.0] range
            let value = ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            embedding.push(value);
        }

        // L2 normalize (like real sentence embeddings)
        self.l2_normalize(&embedding)
    }

    /// L2 normalization
    ///
    /// Normalizes the embedding vector to unit length.
    /// This matches the behavior of real sentence transformers.
    fn l2_normalize(&self, embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm == 0.0 || !norm.is_finite() {
            // Fallback: return as-is if norm is invalid
            warn!("Invalid norm during L2 normalization, returning unnormalized");
            return embedding.to_vec();
        }

        embedding.iter().map(|x| x / norm).collect()
    }
}

impl Default for EmbeddingsExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default executor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = EmbeddingsExecutor::new().unwrap();
        assert_eq!(executor.model_name(), "mock-all-MiniLM-L6-v2");
        assert_eq!(executor.dimensions(), 384);
    }

    #[tokio::test]
    async fn test_basic_execution() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let input = EmbeddingsInput::new("Hello world");

        let output = executor.execute(&input).await.unwrap();

        assert_eq!(output.dimensions, 384);
        assert_eq!(output.embedding.len(), 384);
        assert_eq!(output.model, "mock-all-MiniLM-L6-v2");
    }

    #[tokio::test]
    async fn test_deterministic_embeddings() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let input = EmbeddingsInput::new("Test text");

        let output1 = executor.execute(&input).await.unwrap();
        let output2 = executor.execute(&input).await.unwrap();

        // Same input should produce same embedding
        assert_eq!(output1.embedding, output2.embedding);
    }

    #[tokio::test]
    async fn test_different_texts_produce_different_embeddings() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();

        let output1 = executor
            .execute(&EmbeddingsInput::new("Hello"))
            .await
            .unwrap();
        let output2 = executor
            .execute(&EmbeddingsInput::new("World"))
            .await
            .unwrap();

        // Different inputs should produce different embeddings
        assert_ne!(output1.embedding, output2.embedding);
    }

    #[tokio::test]
    async fn test_empty_text_validation() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let input = EmbeddingsInput::new("");

        let result = executor.execute(&input).await;

        assert!(result.is_err());
        match result {
            Err(ExecutorError::InvalidInput(msg)) => {
                assert!(msg.contains("empty"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_text_too_long_validation() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let long_text = "a".repeat(MAX_INPUT_LENGTH + 1);
        let input = EmbeddingsInput::new(long_text);

        let result = executor.execute(&input).await;

        assert!(result.is_err());
        match result {
            Err(ExecutorError::InvalidInput(msg)) => {
                assert!(msg.contains("too long"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[tokio::test]
    async fn test_timeout_enforcement() {
        let executor = EmbeddingsExecutor::new().unwrap(); // With latency simulation
        let input = EmbeddingsInput::new("Test");

        // Set timeout lower than simulated latency
        let result = executor.execute_with_timeout(&input, 1).await;

        assert!(result.is_err());
        match result {
            Err(ExecutorError::Timeout(ms)) => {
                assert_eq!(ms, 1);
            }
            _ => panic!("Expected Timeout error"),
        }
    }

    #[tokio::test]
    async fn test_successful_execution_with_timeout() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let input = EmbeddingsInput::new("Test");

        let result = executor.execute_with_timeout(&input, 5000).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_l2_normalization() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let input = EmbeddingsInput::new("Test");

        let output = executor.execute(&input).await.unwrap();

        // Calculate norm
        let norm: f32 = output.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        // L2 normalized vector should have norm ≈ 1.0
        assert!(
            (norm - 1.0).abs() < 0.001,
            "Norm should be 1.0, got {}",
            norm
        );
    }

    #[tokio::test]
    async fn test_cbor_roundtrip() {
        let executor = EmbeddingsExecutor::new_fast().unwrap();
        let input = EmbeddingsInput::new("Test text for CBOR");

        let output = executor.execute(&input).await.unwrap();

        // Serialize to CBOR
        let bytes = output.to_cbor().unwrap();

        // Deserialize back
        let decoded = EmbeddingsOutput::from_cbor(&bytes).unwrap();

        assert_eq!(output.embedding, decoded.embedding);
        assert_eq!(output.model, decoded.model);
        assert_eq!(output.dimensions, decoded.dimensions);
    }

    #[test]
    fn test_latency_calculation() {
        let executor = EmbeddingsExecutor::new().unwrap();

        // Short text should have minimum latency
        let short_latency = executor.calculate_latency("Hi");
        assert!(short_latency >= MIN_LATENCY_MS);
        assert!(short_latency <= MAX_LATENCY_MS);

        // Long text should have higher latency
        let long_text = "a".repeat(1000);
        let long_latency = executor.calculate_latency(&long_text);
        assert!(long_latency > short_latency);
        assert!(long_latency <= MAX_LATENCY_MS);
    }
}
