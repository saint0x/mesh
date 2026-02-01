//! Pipeline-Parallel Forward Pass Implementation
//!
//! Each worker runs a contiguous range of complete transformer layers.
//! Activations flow through the pipeline:
//!
//! ```text
//! [Stage 0: embed + layers 0..19] → activations → [Stage 1: layers 20..39]
//!     → activations → [Stage 2: layers 40..59] → activations →
//!     [Stage 3: layers 60..79 + lm_head + sample] → token broadcast → all stages
//! ```
//!
//! Only N-1 activation transfers per token (vs 560+ all-reduce ops in TP).

use crate::errors::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};

use super::kv_cache::KVCache;
use super::tensor_ops::{
    embed_tokens, matmul, rms_norm, sample_greedy, sample_token, silu, softmax, Tensor1D, Tensor2D,
};

/// Weights for a single transformer layer (full — not sharded)
///
/// In pipeline parallelism, each worker holds complete layer weights
/// for the layers it owns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub layer_idx: usize,

    /// Query projection weights [hidden_dim, hidden_dim]
    pub w_q: Tensor2D,
    /// Key projection weights [hidden_dim, hidden_dim]
    pub w_k: Tensor2D,
    /// Value projection weights [hidden_dim, hidden_dim]
    pub w_v: Tensor2D,
    /// Output projection weights [hidden_dim, hidden_dim]
    pub w_o: Tensor2D,

    /// MLP up projection [hidden_dim, intermediate_size]
    pub w_up: Tensor2D,
    /// MLP gate projection (for SwiGLU) [hidden_dim, intermediate_size]
    pub w_gate: Tensor2D,
    /// MLP down projection [intermediate_size, hidden_dim]
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
            * 4
    }
}

/// Full model weights for this pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeights {
    pub model_id: String,

    /// Token embedding table [vocab_size, hidden_dim]
    pub embedding: Tensor2D,

    /// Layer weights for this stage's layer range
    pub layers: Vec<LayerWeights>,

    /// Final RMS norm weights [hidden_dim]
    pub final_norm: Tensor1D,

    /// Output projection (lm_head) [hidden_dim, vocab_size]
    pub lm_head: Tensor2D,

    pub config: ModelConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
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

    pub fn memory_usage(&self) -> usize {
        let layer_mem: usize = self.layers.iter().map(|l| l.memory_usage()).sum();
        let embed_mem = self.embedding.len() * 4;
        let head_mem = self.lm_head.len() * 4;
        let norm_mem = self.final_norm.len() * 4;
        layer_mem + embed_mem + head_mem + norm_mem
    }
}

/// Pipeline-parallel forward pass state
///
/// Each stage runs its assigned layers locally with no network communication
/// during the per-layer compute. The only network transfers are:
/// 1. Receiving activations from the previous stage (or embedding locally)
/// 2. Sending activations to the next stage (or computing logits locally)
pub struct ForwardPass {
    /// Model weights for this stage
    pub weights: ModelWeights,

    /// KV cache for attention (scoped to this stage's layers)
    pub kv_cache: KVCache,

    /// First layer this stage owns (inclusive)
    pub layer_start: usize,

    /// Last layer this stage owns (exclusive)
    pub layer_end: usize,

    /// Total workers in pipeline
    pub total_workers: u32,

    /// This stage's position in the pipeline
    pub stage_position: u32,

    /// Current position in sequence
    pub position: usize,

    // Legacy fields kept for API compat
    pub shard_start: usize,
    pub shard_end: usize,
}

impl ForwardPass {
    /// Create a new forward pass (legacy API — interprets shard range as layers)
    pub fn new(
        weights: ModelWeights,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
    ) -> Self {
        let num_stage_layers = shard_end - shard_start;
        let kv_config = super::kv_cache::KVCacheConfig {
            num_layers: num_stage_layers,
            num_heads: weights.config.num_heads,
            head_dim: weights.config.hidden_dim / weights.config.num_heads,
            max_seq_len: 4096,
        };

        Self {
            weights,
            kv_cache: KVCache::new(kv_config),
            layer_start: shard_start,
            layer_end: shard_end,
            total_workers,
            stage_position: 0,
            position: 0,
            shard_start,
            shard_end,
        }
    }

    /// Create with explicit pipeline stage info
    pub fn new_pipeline_stage(
        weights: ModelWeights,
        layer_start: usize,
        layer_end: usize,
        stage_position: u32,
        total_workers: u32,
    ) -> Self {
        let num_stage_layers = layer_end - layer_start;
        let kv_config = super::kv_cache::KVCacheConfig {
            num_layers: num_stage_layers,
            num_heads: weights.config.num_heads,
            head_dim: weights.config.hidden_dim / weights.config.num_heads,
            max_seq_len: 4096,
        };

        Self {
            weights,
            kv_cache: KVCache::new(kv_config),
            layer_start,
            layer_end,
            total_workers,
            stage_position,
            position: 0,
            shard_start: layer_start,
            shard_end: layer_end,
        }
    }

    pub fn is_first_stage(&self) -> bool {
        self.stage_position == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.stage_position == self.total_workers - 1
    }

    /// Embed input tokens (only called by first stage)
    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor2D> {
        embed_tokens(&self.weights.embedding, tokens)
    }

    /// Run a single layer forward pass locally (no network)
    ///
    /// `rel_layer_idx` is the index into `self.weights.layers` (0-based).
    pub fn forward_layer_local(
        &mut self,
        hidden: &Tensor2D,
        rel_layer_idx: usize,
    ) -> Result<Tensor2D> {
        let layer = &self.weights.layers[rel_layer_idx];
        let config = &self.weights.config;

        // 1. RMS Norm before attention
        let normed = rms_norm(hidden, &layer.attn_norm, config.rms_norm_eps)?;

        // 2. Full QKV projections (complete — no sharding)
        let q = matmul(&normed, &layer.w_q)?;
        let k = matmul(&normed, &layer.w_k)?;
        let v = matmul(&normed, &layer.w_v)?;

        // 3. Update KV cache
        self.kv_cache.update_layer(rel_layer_idx, k.clone(), v.clone())?;

        // 4. Compute attention
        let attn_scores = matmul(&q, &k.transpose())?;
        let scale = 1.0 / (config.hidden_dim as f32 / config.num_heads as f32).sqrt();
        let scaled_scores = attn_scores.scale(scale);
        let attn_probs = softmax(&scaled_scores);
        let attn_output = matmul(&attn_probs, &v)?;

        // 5. Output projection
        let o = matmul(&attn_output, &layer.w_o)?;

        // 6. Residual connection
        let post_attn = hidden.add(&o)?;

        // 7. MLP with SwiGLU
        let mlp_normed = rms_norm(&post_attn, &layer.mlp_norm, config.rms_norm_eps)?;
        let gate = matmul(&mlp_normed, &layer.w_gate)?;
        let up = matmul(&mlp_normed, &layer.w_up)?;
        let gate_activated = silu(&gate);
        let mlp_hidden = gate_activated.mul(&up)?;
        let down = matmul(&mlp_hidden, &layer.w_down)?;

        // 8. Final residual
        let output = post_attn.add(&down)?;

        Ok(output)
    }

    /// Run this stage's complete forward pass (all local layers, no network)
    ///
    /// Input: hidden states from previous stage (or embedded tokens for stage 0)
    /// Output: hidden states to send to next stage (or to compute logits for last stage)
    pub fn forward_stage(&mut self, input_hidden: &Tensor2D) -> Result<Tensor2D> {
        let start = Instant::now();
        let mut hidden = input_hidden.clone();

        for rel_idx in 0..self.weights.layers.len() {
            let abs_layer_idx = self.layer_start + rel_idx;
            hidden = self.forward_layer_local(&hidden, rel_idx)?;

            debug!(
                stage = self.stage_position,
                layer = abs_layer_idx,
                "Layer complete"
            );
        }

        info!(
            stage = self.stage_position,
            layers = format!("{}-{}", self.layer_start, self.layer_end),
            elapsed = ?start.elapsed(),
            "Stage forward pass complete"
        );

        Ok(hidden)
    }

    /// Compute logits from final hidden states (only called by last stage)
    pub fn compute_logits(&self, hidden: &Tensor2D) -> Result<Tensor1D> {
        let normed = rms_norm(hidden, &self.weights.final_norm, self.weights.config.rms_norm_eps)?;
        let last_row = normed.row(normed.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, normed.cols)?;
        let logits_2d = matmul(&last_hidden, &self.weights.lm_head)?;
        Ok(Tensor1D::new(logits_2d.data))
    }

    pub fn sample(&self, logits: &Tensor1D, temperature: f32, top_p: f32, seed: u64) -> u32 {
        if temperature <= 0.0 || temperature == 1.0 && top_p >= 1.0 {
            sample_greedy(logits)
        } else {
            sample_token(logits, temperature, top_p, seed)
        }
    }

    /// Run full forward pass for all layers (single-stage fallback, legacy API)
    pub async fn forward(
        &mut self,
        tokens: &[u32],
        _worker_ring: &mut crate::executor::ring_allreduce::WorkerRing<'_>,
        _job_id: uuid::Uuid,
    ) -> Result<Tensor2D> {
        let mut hidden = self.embed(tokens)?;
        hidden = self.forward_stage(&hidden)?;
        hidden = rms_norm(&hidden, &self.weights.final_norm, self.weights.config.rms_norm_eps)?;
        Ok(hidden)
    }

    /// Generate next token (single-stage fallback, legacy API)
    pub async fn generate_next_token(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut crate::executor::ring_allreduce::WorkerRing<'_>,
        job_id: uuid::Uuid,
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
        let hidden = self.forward(tokens, worker_ring, job_id).await?;
        let logits = self.compute_logits(&hidden)?;
        let seed = job_id.as_u128() as u64 ^ self.position as u64;
        let next_token = self.sample(&logits, temperature, top_p, seed);
        self.position += 1;
        Ok(next_token)
    }

    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
        self.position = 0;
    }

    pub fn cache_memory_usage(&self) -> usize {
        self.kv_cache.memory_usage()
    }
}

/// Simplified forward pass for testing without network
pub struct LocalForwardPass {
    pub weights: ModelWeights,
    pub kv_cache: KVCache,
    pub position: usize,
}

impl LocalForwardPass {
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

    pub fn forward(&mut self, tokens: &[u32]) -> Result<Tensor2D> {
        let mut hidden = embed_tokens(&self.weights.embedding, tokens)?;

        for layer_idx in 0..self.weights.config.num_layers {
            let layer = &self.weights.layers[layer_idx];
            let config = &self.weights.config;

            let normed = rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps)?;

            let gate = matmul(&normed, &layer.w_gate)?;
            let up = matmul(&normed, &layer.w_up)?;
            let gate_activated = silu(&gate);
            let mlp_hidden = gate_activated.mul(&up)?;
            let mlp_out = matmul(&mlp_hidden, &layer.w_down)?;

            hidden = hidden.add(&mlp_out)?;
        }

        hidden = rms_norm(&hidden, &self.weights.final_norm, self.weights.config.rms_norm_eps)?;
        Ok(hidden)
    }

    pub fn generate_next_token(&mut self, tokens: &[u32], temperature: f32, top_p: f32) -> Result<u32> {
        let hidden = self.forward(tokens)?;
        let last_row = hidden.row(hidden.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, hidden.cols)?;
        let logits_2d = matmul(&last_hidden, &self.weights.lm_head)?;
        let logits = Tensor1D::new(logits_2d.data);

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
        let weights = ModelWeights::placeholder(config.clone(), 64);
        assert_eq!(weights.layers.len(), config.num_layers);
        assert!(weights.memory_usage() > 0);
    }

    #[test]
    fn test_layer_weights_memory() {
        let layer = LayerWeights::placeholder(0, 64, 64);
        assert!(layer.memory_usage() > 0);
    }

    #[test]
    fn test_local_forward_pass() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let mut forward = LocalForwardPass::new(weights);

        let tokens = vec![1, 2, 3];
        let hidden = forward.forward(&tokens).unwrap();
        assert_eq!(hidden.rows, 3);
        assert_eq!(hidden.cols, 64);
    }

    #[test]
    fn test_local_generate() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let mut forward = LocalForwardPass::new(weights);

        let tokens = vec![1, 2, 3];
        let next_token = forward.generate_next_token(&tokens, 1.0, 0.9).unwrap();
        assert!(next_token < 100);
    }

    #[test]
    fn test_forward_pass_creation() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let forward = ForwardPass::new(weights, 0, 2, 1);

        assert_eq!(forward.layer_start, 0);
        assert_eq!(forward.layer_end, 2);
        assert_eq!(forward.total_workers, 1);
    }

    #[test]
    fn test_pipeline_stage_creation() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let forward = ForwardPass::new_pipeline_stage(weights, 0, 2, 0, 4);

        assert!(forward.is_first_stage());
        assert!(!forward.is_last_stage());
    }

    #[test]
    fn test_single_stage_forward() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let mut forward = ForwardPass::new_pipeline_stage(weights, 0, 2, 0, 1);

        let tokens = vec![1, 2, 3];
        let embedded = forward.embed(&tokens).unwrap();
        let output = forward.forward_stage(&embedded).unwrap();
        assert_eq!(output.rows, 3);
        assert_eq!(output.cols, 64);
    }

    #[test]
    fn test_two_stage_pipeline_matches_single_stage() {
        // Verify: running 2 layers in a single stage produces the same result
        // as running layer 0 in stage 0 and layer 1 in stage 1 with activation passing.
        let config = ModelConfig {
            hidden_dim: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 2,
            vocab_size: 50,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let full_weights = ModelWeights::placeholder(config.clone(), 32);
        let tokens = vec![1, 5, 10];

        // Single-stage: run all 2 layers together
        let mut single = ForwardPass::new_pipeline_stage(full_weights.clone(), 0, 2, 0, 1);
        let embedded = single.embed(&tokens).unwrap();
        let single_output = single.forward_stage(&embedded).unwrap();

        // Two-stage: split layers
        let mut stage0_weights = full_weights.clone();
        stage0_weights.layers = vec![full_weights.layers[0].clone()];
        stage0_weights.config.num_layers = 1;

        let mut stage1_weights = full_weights.clone();
        stage1_weights.layers = vec![full_weights.layers[1].clone()];
        stage1_weights.config.num_layers = 1;

        let mut stage0 = ForwardPass::new_pipeline_stage(stage0_weights, 0, 1, 0, 2);
        let mut stage1 = ForwardPass::new_pipeline_stage(stage1_weights, 1, 2, 1, 2);

        let embedded = stage0.embed(&tokens).unwrap();
        let activations = stage0.forward_stage(&embedded).unwrap();

        // Pass activations to stage 1 (simulating network transfer)
        let two_stage_output = stage1.forward_stage(&activations).unwrap();

        // Outputs must be identical
        assert_eq!(single_output.rows, two_stage_output.rows);
        assert_eq!(single_output.cols, two_stage_output.cols);
        assert_eq!(single_output.data.len(), two_stage_output.data.len());

        for (i, (a, b)) in single_output.data.iter().zip(&two_stage_output.data).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "Divergence at index {}: single={}, two_stage={} (diff={})",
                i, a, b, (a - b).abs()
            );
        }
    }

    #[test]
    fn test_pipeline_activation_dimensions_preserved() {
        // Verify activation dimensions are preserved across stages
        let config = ModelConfig {
            hidden_dim: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 4,
            vocab_size: 50,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let full_weights = ModelWeights::placeholder(config.clone(), 32);
        let tokens = vec![1, 2, 3, 4, 5];

        // Split into 4 stages of 1 layer each
        for stage_idx in 0..4 {
            let mut stage_weights = full_weights.clone();
            stage_weights.layers = vec![full_weights.layers[stage_idx].clone()];
            stage_weights.config.num_layers = 1;

            let mut stage = ForwardPass::new_pipeline_stage(
                stage_weights, stage_idx, stage_idx + 1, stage_idx as u32, 4,
            );

            let input = if stage_idx == 0 {
                stage.embed(&tokens).unwrap()
            } else {
                // Simulate receiving activations: [seq_len, hidden_dim]
                Tensor2D::filled(tokens.len(), 32, 0.5)
            };

            let output = stage.forward_stage(&input).unwrap();

            // Every stage must preserve [seq_len, hidden_dim]
            assert_eq!(
                output.rows, tokens.len(),
                "Stage {} changed seq_len: expected {}, got {}",
                stage_idx, tokens.len(), output.rows
            );
            assert_eq!(
                output.cols, 32,
                "Stage {} changed hidden_dim: expected 32, got {}",
                stage_idx, output.cols
            );
        }
    }

    #[test]
    fn test_logits_and_sampling_deterministic() {
        // Verify that same inputs + same seed produce same token
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config, 64);
        let mut forward = ForwardPass::new_pipeline_stage(weights, 0, 2, 0, 1);

        let tokens = vec![1, 2, 3];
        let embedded = forward.embed(&tokens).unwrap();
        let hidden = forward.forward_stage(&embedded).unwrap();
        let logits = forward.compute_logits(&hidden).unwrap();

        let seed = 12345u64;
        let token1 = forward.sample(&logits, 1.0, 0.9, seed);
        let token2 = forward.sample(&logits, 1.0, 0.9, seed);

        assert_eq!(token1, token2, "Same seed must produce same token");
        assert!(token1 < 100, "Token must be within vocab range");
    }

    #[test]
    fn test_greedy_sampling_selects_max() {
        // Verify greedy sampling picks the highest logit
        let logits = Tensor1D::new(vec![0.0, 0.0, 10.0, 0.0, 0.0]);
        let token = sample_greedy(&logits);
        assert_eq!(token, 2, "Greedy should select index of max logit");
    }

    #[test]
    fn test_embed_produces_correct_shape() {
        let config = create_test_config();
        let weights = ModelWeights::placeholder(config.clone(), 64);
        let forward = ForwardPass::new_pipeline_stage(weights, 0, 2, 0, 1);

        let tokens = vec![0, 50, 99];
        let embedded = forward.embed(&tokens).unwrap();
        assert_eq!(embedded.rows, 3);
        assert_eq!(embedded.cols, config.hidden_dim);
    }
}
