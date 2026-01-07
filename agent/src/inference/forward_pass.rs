//! Tensor-Parallel Forward Pass Implementation
//!
//! This module implements the actual tensor-parallel forward pass for transformer inference.
//! Each worker holds a column shard of the model weights and participates in ring all-reduce
//! to produce identical activations across all workers.
//!
//! ## Forward Pass Flow (per layer)
//!
//! ```text
//! Input: hidden_states [seq_len, hidden_dim]
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 1. Partial Attention (my columns only)              │
//! │    Q_partial = hidden @ W_q[my_cols]                │
//! │    K_partial = hidden @ W_k[my_cols]                │
//! │    V_partial = hidden @ W_v[my_cols]                │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 2. Ring All-Reduce (combine partials)               │
//! │    Q_full = ring_allreduce(Q_partial)               │
//! │    K_full = ring_allreduce(K_partial)               │
//! │    V_full = ring_allreduce(V_partial)               │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 3. Attention Computation (identical on all workers) │
//! │    scores = softmax(Q @ K^T / sqrt(d))              │
//! │    attn_output = scores @ V                         │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 4. MLP (my columns only + all-reduce)               │
//! │    mlp_partial = attn_output @ W_mlp[my_cols]       │
//! │    mlp_full = ring_allreduce(mlp_partial)           │
//! │    output = activation(mlp_full) @ W_down          │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! Output: next_hidden_states [seq_len, hidden_dim]
//! ```

use crate::errors::Result;
use crate::executor::ring_allreduce::WorkerRing;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

use super::kv_cache::KVCache;
use super::tensor_ops::{
    embed_tokens, matmul, rms_norm, sample_greedy, sample_token, silu, softmax, Tensor1D, Tensor2D,
};

/// Weights for a single transformer layer (sharded)
///
/// Each worker holds only their column range of the weight matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Layer index
    pub layer_idx: usize,

    /// Query projection weights [hidden_dim, shard_cols]
    pub w_q: Tensor2D,
    /// Key projection weights [hidden_dim, shard_cols]
    pub w_k: Tensor2D,
    /// Value projection weights [hidden_dim, shard_cols]
    pub w_v: Tensor2D,
    /// Output projection weights [shard_cols, hidden_dim]
    pub w_o: Tensor2D,

    /// MLP up projection [hidden_dim, shard_cols]
    pub w_up: Tensor2D,
    /// MLP gate projection (for SwiGLU) [hidden_dim, shard_cols]
    pub w_gate: Tensor2D,
    /// MLP down projection [shard_cols, hidden_dim]
    pub w_down: Tensor2D,

    /// RMS norm weight for attention [hidden_dim]
    pub attn_norm: Tensor1D,
    /// RMS norm weight for MLP [hidden_dim]
    pub mlp_norm: Tensor1D,
}

impl LayerWeights {
    /// Create placeholder weights for testing
    pub fn placeholder(layer_idx: usize, hidden_dim: usize, shard_cols: usize) -> Self {
        Self {
            layer_idx,
            w_q: Tensor2D::filled(hidden_dim, shard_cols, 0.01),
            w_k: Tensor2D::filled(hidden_dim, shard_cols, 0.01),
            w_v: Tensor2D::filled(hidden_dim, shard_cols, 0.01),
            w_o: Tensor2D::filled(shard_cols, hidden_dim, 0.01),
            w_up: Tensor2D::filled(hidden_dim, shard_cols, 0.01),
            w_gate: Tensor2D::filled(hidden_dim, shard_cols, 0.01),
            w_down: Tensor2D::filled(shard_cols, hidden_dim, 0.01),
            attn_norm: Tensor1D::new(vec![1.0; hidden_dim]),
            mlp_norm: Tensor1D::new(vec![1.0; hidden_dim]),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        (self.w_q.len()
            + self.w_k.len()
            + self.w_v.len()
            + self.w_o.len()
            + self.w_up.len()
            + self.w_gate.len()
            + self.w_down.len()
            + self.attn_norm.len()
            + self.mlp_norm.len())
            * 4 // f32 = 4 bytes
    }
}

/// Full model weights (sharded across workers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    /// Model identifier
    pub model_id: String,

    /// Token embedding table [vocab_size, hidden_dim]
    pub embedding: Tensor2D,

    /// Per-layer weights (sharded)
    pub layers: Vec<LayerWeights>,

    /// Final RMS norm weights [hidden_dim]
    pub final_norm: Tensor1D,

    /// Output projection (lm_head) [hidden_dim, vocab_size]
    /// Note: Often tied to embedding.transpose()
    pub lm_head: Tensor2D,

    /// Model configuration
    pub config: ModelConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE base frequency
    pub rope_base: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        // LLaMA 70B configuration
        Self {
            hidden_dim: 8192,
            num_heads: 64,
            num_kv_heads: 8,
            num_layers: 80,
            vocab_size: 32000,
            intermediate_size: 28672,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }
}

impl ModelWeights {
    /// Create placeholder weights for testing
    pub fn placeholder(config: ModelConfig, shard_cols: usize) -> Self {
        let layers = (0..config.num_layers)
            .map(|i| LayerWeights::placeholder(i, config.hidden_dim, shard_cols))
            .collect();

        Self {
            model_id: "placeholder".to_string(),
            embedding: Tensor2D::filled(config.vocab_size, config.hidden_dim, 0.01),
            layers,
            final_norm: Tensor1D::new(vec![1.0; config.hidden_dim]),
            lm_head: Tensor2D::filled(config.hidden_dim, config.vocab_size, 0.01),
            config,
        }
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let layer_mem: usize = self.layers.iter().map(|l| l.memory_usage()).sum();
        let embed_mem = self.embedding.len() * 4;
        let head_mem = self.lm_head.len() * 4;
        let norm_mem = self.final_norm.len() * 4;
        layer_mem + embed_mem + head_mem + norm_mem
    }
}

/// Forward pass state for tensor-parallel inference
pub struct ForwardPass {
    /// Model weights (sharded for this worker)
    pub weights: ModelWeights,

    /// KV cache for attention
    pub kv_cache: KVCache,

    /// This worker's shard column range
    pub shard_start: usize,
    pub shard_end: usize,

    /// Total workers in ring
    pub total_workers: u32,

    /// Current position in sequence
    pub position: usize,
}

impl ForwardPass {
    /// Create a new forward pass state
    pub fn new(
        weights: ModelWeights,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
    ) -> Self {
        let kv_config = super::kv_cache::KVCacheConfig {
            num_layers: weights.config.num_layers,
            num_heads: weights.config.num_heads,
            head_dim: weights.config.hidden_dim / weights.config.num_heads,
            max_seq_len: 4096,
        };

        Self {
            weights,
            kv_cache: KVCache::new(kv_config),
            shard_start,
            shard_end,
            total_workers,
            position: 0,
        }
    }

    /// Embed input tokens
    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor2D> {
        embed_tokens(&self.weights.embedding, tokens)
    }

    /// Run forward pass for a single layer with ring all-reduce
    ///
    /// This is the core tensor-parallel operation:
    /// 1. Compute partial attention/MLP with local shard
    /// 2. Ring all-reduce to combine results
    /// 3. Apply activation and normalization
    pub async fn forward_layer(
        &mut self,
        hidden: &Tensor2D,
        layer_idx: usize,
        worker_ring: &mut WorkerRing,
        job_id: Uuid,
    ) -> Result<Tensor2D> {
        let layer = &self.weights.layers[layer_idx];
        let config = &self.weights.config;
        let start = Instant::now();

        // 1. RMS Norm before attention
        let normed = rms_norm(hidden, &layer.attn_norm, config.rms_norm_eps)?;

        // 2. Compute partial QKV projections (my columns only)
        let q_partial = matmul(&normed, &layer.w_q)?;
        let k_partial = matmul(&normed, &layer.w_k)?;
        let v_partial = matmul(&normed, &layer.w_v)?;

        debug!(
            "Layer {} QKV partial computed: {}x{} -> {}x{}",
            layer_idx, normed.rows, normed.cols, q_partial.rows, q_partial.cols
        );

        // 3. Ring all-reduce to get full QKV
        let q_full = self
            .ring_allreduce_tensor(&q_partial, worker_ring, job_id, layer_idx as u32)
            .await?;
        let k_full = self
            .ring_allreduce_tensor(&k_partial, worker_ring, job_id, layer_idx as u32)
            .await?;
        let v_full = self
            .ring_allreduce_tensor(&v_partial, worker_ring, job_id, layer_idx as u32)
            .await?;

        // 4. Update KV cache
        self.kv_cache.update_layer(layer_idx, k_full.clone(), v_full.clone())?;

        // 5. Compute attention (simplified - no heads split for clarity)
        // In production, this would properly handle multi-head attention
        let attn_scores = matmul(&q_full, &k_full.transpose())?;
        let scale = 1.0 / (config.hidden_dim as f32 / config.num_heads as f32).sqrt();
        let scaled_scores = attn_scores.scale(scale);
        let attn_probs = softmax(&scaled_scores);
        let attn_output = matmul(&attn_probs, &v_full)?;

        // 6. Output projection (partial)
        let o_partial = matmul(&attn_output, &layer.w_o)?;
        let o_full = self
            .ring_allreduce_tensor(&o_partial, worker_ring, job_id, layer_idx as u32)
            .await?;

        // 7. Residual connection
        let post_attn = hidden.add(&o_full)?;

        // 8. MLP with SwiGLU
        let mlp_normed = rms_norm(&post_attn, &layer.mlp_norm, config.rms_norm_eps)?;

        // Gate and up projections (partial)
        let gate_partial = matmul(&mlp_normed, &layer.w_gate)?;
        let up_partial = matmul(&mlp_normed, &layer.w_up)?;

        // All-reduce gate and up
        let gate_full = self
            .ring_allreduce_tensor(&gate_partial, worker_ring, job_id, layer_idx as u32)
            .await?;
        let up_full = self
            .ring_allreduce_tensor(&up_partial, worker_ring, job_id, layer_idx as u32)
            .await?;

        // SwiGLU: silu(gate) * up
        let gate_activated = silu(&gate_full);
        let mlp_hidden = gate_activated.mul(&up_full)?;

        // Down projection (partial)
        let down_partial = matmul(&mlp_hidden, &layer.w_down)?;
        let down_full = self
            .ring_allreduce_tensor(&down_partial, worker_ring, job_id, layer_idx as u32)
            .await?;

        // 9. Final residual
        let output = post_attn.add(&down_full)?;

        debug!(
            "Layer {} forward complete in {:?}",
            layer_idx,
            start.elapsed()
        );

        Ok(output)
    }

    /// Compute logits from final hidden states
    pub fn compute_logits(&self, hidden: &Tensor2D) -> Result<Tensor1D> {
        // Apply final RMS norm
        let normed = rms_norm(hidden, &self.weights.final_norm, self.weights.config.rms_norm_eps)?;

        // Take last token's hidden state
        let last_row = normed.row(normed.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, normed.cols)?;

        // Project to vocabulary
        let logits_2d = matmul(&last_hidden, &self.weights.lm_head)?;

        // Flatten to 1D
        Ok(Tensor1D::new(logits_2d.data))
    }

    /// Sample next token from logits
    pub fn sample(&self, logits: &Tensor1D, temperature: f32, top_p: f32, seed: u64) -> u32 {
        if temperature <= 0.0 || temperature == 1.0 && top_p >= 1.0 {
            sample_greedy(logits)
        } else {
            sample_token(logits, temperature, top_p, seed)
        }
    }

    /// Run full forward pass for all layers
    pub async fn forward(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing,
        job_id: Uuid,
    ) -> Result<Tensor2D> {
        let start = Instant::now();

        // 1. Embed tokens
        let mut hidden = self.embed(tokens)?;
        debug!("Embedded {} tokens -> {:?}", tokens.len(), (hidden.rows, hidden.cols));

        // 2. Run through all layers
        for layer_idx in 0..self.weights.config.num_layers {
            hidden = self.forward_layer(&hidden, layer_idx, worker_ring, job_id).await?;
        }

        // 3. Apply final norm
        hidden = rms_norm(&hidden, &self.weights.final_norm, self.weights.config.rms_norm_eps)?;

        info!(
            "Full forward pass complete: {} layers in {:?}",
            self.weights.config.num_layers,
            start.elapsed()
        );

        Ok(hidden)
    }

    /// Generate next token using full forward pass
    pub async fn generate_next_token(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing,
        job_id: Uuid,
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
        // Run forward pass
        let hidden = self.forward(tokens, worker_ring, job_id).await?;

        // Compute logits
        let logits = self.compute_logits(&hidden)?;

        // Sample
        let seed = job_id.as_u128() as u64 ^ self.position as u64;
        let next_token = self.sample(&logits, temperature, top_p, seed);

        self.position += 1;

        Ok(next_token)
    }

    /// Helper: Convert Tensor2D to ring all-reduce format and back
    async fn ring_allreduce_tensor(
        &self,
        tensor: &Tensor2D,
        worker_ring: &mut WorkerRing,
        job_id: Uuid,
        layer_idx: u32,
    ) -> Result<Tensor2D> {
        // Convert to flat tensor for all-reduce
        let flat = tensor.to_allreduce_tensor();

        // Perform ring all-reduce
        let reduced = worker_ring
            .ring_all_reduce(flat, job_id, layer_idx)
            .await?;

        // Convert back to 2D
        Tensor2D::from_allreduce_tensor(&reduced)
    }

    /// Clear KV cache (for new sequence)
    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
        self.position = 0;
    }

    /// Get current KV cache memory usage
    pub fn cache_memory_usage(&self) -> usize {
        self.kv_cache.memory_usage()
    }
}

/// Simplified forward pass for testing without ring all-reduce
pub struct LocalForwardPass {
    /// Model weights
    pub weights: ModelWeights,
    /// KV cache
    pub kv_cache: KVCache,
    /// Current position
    pub position: usize,
}

impl LocalForwardPass {
    /// Create a new local forward pass
    pub fn new(weights: ModelWeights) -> Self {
        let kv_config = super::kv_cache::KVCacheConfig {
            num_layers: weights.config.num_layers,
            num_heads: weights.config.num_heads,
            head_dim: weights.config.hidden_dim / weights.config.num_heads,
            max_seq_len: 4096,
        };

        Self {
            weights,
            kv_cache: KVCache::new(kv_config),
            position: 0,
        }
    }

    /// Run forward pass without distribution (for testing)
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Tensor2D> {
        // Embed tokens
        let mut hidden = embed_tokens(&self.weights.embedding, tokens)?;

        // Run through all layers (simplified - no attention for demo)
        for layer_idx in 0..self.weights.config.num_layers {
            let layer = &self.weights.layers[layer_idx];
            let config = &self.weights.config;

            // RMS Norm
            let normed = rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps)?;

            // Simplified: just MLP (no real attention for testing)
            let gate = matmul(&normed, &layer.w_gate)?;
            let up = matmul(&normed, &layer.w_up)?;
            let gate_activated = silu(&gate);
            let mlp_hidden = gate_activated.mul(&up)?;
            let mlp_out = matmul(&mlp_hidden, &layer.w_down)?;

            // Residual
            hidden = hidden.add(&mlp_out)?;
        }

        // Final norm
        hidden = rms_norm(&hidden, &self.weights.final_norm, self.weights.config.rms_norm_eps)?;

        Ok(hidden)
    }

    /// Generate next token locally
    pub fn generate_next_token(&mut self, tokens: &[u32], temperature: f32, top_p: f32) -> Result<u32> {
        let hidden = self.forward(tokens)?;

        // Get last hidden state
        let last_row = hidden.row(hidden.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, hidden.cols)?;

        // Project to logits
        let logits_2d = matmul(&last_hidden, &self.weights.lm_head)?;
        let logits = Tensor1D::new(logits_2d.data);

        // Sample
        let seed = self.position as u64;
        let next_token = if temperature <= 0.0 {
            sample_greedy(&logits)
        } else {
            sample_token(&logits, temperature, top_p, seed)
        };

        self.position += 1;
        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 4,
            num_layers: 2,
            vocab_size: 100,
            intermediate_size: 128,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    #[test]
    fn test_placeholder_weights() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config.clone(), 32);

        assert_eq!(weights.layers.len(), config.num_layers);
        assert!(weights.memory_usage() > 0);
    }

    #[test]
    fn test_layer_weights_memory() {
        let layer = LayerWeights::placeholder(0, 64, 32);
        let mem = layer.memory_usage();

        // Should be roughly: 7 matrices * 64 * 32 * 4 + 2 norms * 64 * 4
        assert!(mem > 0);
    }

    #[test]
    fn test_local_forward_pass() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64); // Full width for local
        let mut forward = LocalForwardPass::new(weights);

        let tokens = vec![1, 2, 3];
        let hidden = forward.forward(&tokens).unwrap();

        assert_eq!(hidden.rows, 3); // seq_len
        assert_eq!(hidden.cols, 64); // hidden_dim
    }

    #[test]
    fn test_local_generate() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let mut forward = LocalForwardPass::new(weights);

        let tokens = vec![1, 2, 3];
        let next_token = forward.generate_next_token(&tokens, 1.0, 0.9).unwrap();

        // Token should be in vocab range
        assert!(next_token < 100);
    }

    #[test]
    fn test_forward_pass_creation() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 16);
        let forward = ForwardPass::new(weights, 0, 16, 4);

        assert_eq!(forward.shard_start, 0);
        assert_eq!(forward.shard_end, 16);
        assert_eq!(forward.total_workers, 4);
    }
}
