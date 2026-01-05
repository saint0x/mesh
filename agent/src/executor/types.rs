//! Payload types for job execution
//!
//! These types are serialized with CBOR for network transmission.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for job execution
#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("Model download failed: {0}")]
    ModelDownload(String),

    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Result type for executor operations
pub type ExecutorResult<T> = Result<T, ExecutorError>;

/// Input for embeddings workload
///
/// CBOR-serialized format sent in JobEnvelope.payload
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingsInput {
    /// Text to generate embeddings for (max 512 tokens for all-MiniLM-L6-v2)
    pub text: String,

    /// Optional model identifier (default: "all-MiniLM-L6-v2")
    #[serde(default)]
    pub model: Option<String>,
}

impl EmbeddingsInput {
    /// Create new embeddings input
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: None,
        }
    }

    /// Create embeddings input with specific model
    pub fn with_model(text: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            model: Some(model.into()),
        }
    }

    /// Serialize to CBOR bytes
    pub fn to_cbor(&self) -> ExecutorResult<Vec<u8>> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf)
            .map_err(|e| ExecutorError::Serialization(e.to_string()))?;
        Ok(buf)
    }

    /// Deserialize from CBOR bytes
    pub fn from_cbor(bytes: &[u8]) -> ExecutorResult<Self> {
        ciborium::from_reader(bytes).map_err(|e| ExecutorError::Serialization(e.to_string()))
    }
}

/// Output for embeddings workload
///
/// CBOR-serialized format sent in JobResult.result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingsOutput {
    /// Embedding vector (384 dimensions for all-MiniLM-L6-v2)
    pub embedding: Vec<f32>,

    /// Model used for generation
    pub model: String,

    /// Dimension of the embedding
    pub dimensions: usize,
}

impl EmbeddingsOutput {
    /// Create new embeddings output
    pub fn new(embedding: Vec<f32>, model: impl Into<String>) -> Self {
        let dimensions = embedding.len();
        Self {
            embedding,
            model: model.into(),
            dimensions,
        }
    }

    /// Serialize to CBOR bytes
    pub fn to_cbor(&self) -> ExecutorResult<Vec<u8>> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf)
            .map_err(|e| ExecutorError::Serialization(e.to_string()))?;
        Ok(buf)
    }

    /// Deserialize from CBOR bytes
    pub fn from_cbor(bytes: &[u8]) -> ExecutorResult<Self> {
        ciborium::from_reader(bytes).map_err(|e| ExecutorError::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings_input_new() {
        let input = EmbeddingsInput::new("Hello world");
        assert_eq!(input.text, "Hello world");
        assert_eq!(input.model, None);
    }

    #[test]
    fn test_embeddings_input_with_model() {
        let input = EmbeddingsInput::with_model("Hello", "custom-model");
        assert_eq!(input.text, "Hello");
        assert_eq!(input.model, Some("custom-model".to_string()));
    }

    #[test]
    fn test_embeddings_input_cbor_roundtrip() {
        let input = EmbeddingsInput::new("Test text");
        let bytes = input.to_cbor().unwrap();
        let decoded = EmbeddingsInput::from_cbor(&bytes).unwrap();
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_embeddings_output_new() {
        let embedding = vec![0.1, 0.2, 0.3];
        let output = EmbeddingsOutput::new(embedding.clone(), "test-model");
        assert_eq!(output.embedding, embedding);
        assert_eq!(output.model, "test-model");
        assert_eq!(output.dimensions, 3);
    }

    #[test]
    fn test_embeddings_output_cbor_roundtrip() {
        let output = EmbeddingsOutput::new(vec![0.1, 0.2, 0.3], "model");
        let bytes = output.to_cbor().unwrap();
        let decoded = EmbeddingsOutput::from_cbor(&bytes).unwrap();
        assert_eq!(output, decoded);
    }

    #[test]
    fn test_cbor_with_large_embedding() {
        // Test with realistic 384-dim embedding
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.001).collect();
        let output = EmbeddingsOutput::new(embedding.clone(), "all-MiniLM-L6-v2");

        let bytes = output.to_cbor().unwrap();
        let decoded = EmbeddingsOutput::from_cbor(&bytes).unwrap();

        assert_eq!(decoded.embedding.len(), 384);
        assert_eq!(decoded.dimensions, 384);
        assert_eq!(decoded.model, "all-MiniLM-L6-v2");
    }
}
