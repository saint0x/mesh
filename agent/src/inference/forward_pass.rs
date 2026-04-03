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
use candle_core::Tensor as CandleTensor;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

use super::kv_cache::KVCache;
use super::tensor_ops::{
    apply_rope, apply_rope_candle, causal_self_attention_candle, causal_self_attention_gpu,
    embed_tokens, from_candle_2d, matmul, rms_norm, rms_norm_candle, sample_greedy, sample_token,
    silu, silu_candle, to_candle_1d, to_candle_2d, Tensor1D, Tensor2D,
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
    /// Key projection weights [hidden_dim, kv_shard_cols]
    pub w_k: Tensor2D,
    /// Value projection weights [hidden_dim, kv_shard_cols]
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
    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let layer_mem: usize = self.layers.iter().map(|l| l.memory_usage()).sum();
        let embed_mem = self.embedding.len() * 4;
        let head_mem = self.lm_head.len() * 4;
        let norm_mem = self.final_norm.len() * 4;
        layer_mem + embed_mem + head_mem + norm_mem
    }
}

fn split_heads(tensor: &Tensor2D, num_heads: usize, head_dim: usize) -> Result<Vec<Tensor2D>> {
    if tensor.cols != num_heads * head_dim {
        return Err(crate::errors::AgentError::Execution(format!(
            "Head split mismatch: tensor cols {} vs num_heads {} * head_dim {}",
            tensor.cols, num_heads, head_dim
        )));
    }

    let mut heads = Vec::with_capacity(num_heads);
    for head_idx in 0..num_heads {
        let start = head_idx * head_dim;
        heads.push(tensor.column_slice(start, start + head_dim)?);
    }
    Ok(heads)
}

fn merge_heads(heads: &[Tensor2D]) -> Result<Tensor2D> {
    let Some(first) = heads.first() else {
        return Ok(Tensor2D::zeros(0, 0));
    };

    let seq_len = first.rows;
    let head_dim = first.cols;
    let num_heads = heads.len();

    for head in heads {
        if head.rows != seq_len || head.cols != head_dim {
            return Err(crate::errors::AgentError::Execution(
                "Cannot merge attention heads with mismatched shapes".to_string(),
            ));
        }
    }

    let mut merged = Tensor2D::zeros(seq_len, num_heads * head_dim);
    for row in 0..seq_len {
        for (head_idx, head) in heads.iter().enumerate() {
            let dst_offset = row * merged.cols + head_idx * head_dim;
            let src_offset = row * head.cols;
            merged.data[dst_offset..dst_offset + head_dim]
                .copy_from_slice(&head.data[src_offset..src_offset + head_dim]);
        }
    }

    Ok(merged)
}

fn partition_start(total_columns: usize, worker_position: u32, total_workers: u32) -> usize {
    if total_workers == 0 {
        return 0;
    }

    let total_workers = total_workers as usize;
    let worker_position = worker_position as usize;
    let columns_per_worker = total_columns / total_workers;
    let remainder = total_columns % total_workers;

    if worker_position < remainder {
        worker_position * (columns_per_worker + 1)
    } else {
        remainder * (columns_per_worker + 1) + (worker_position - remainder) * columns_per_worker
    }
}

fn partition_columns(total_columns: usize, worker_position: u32, total_workers: u32) -> usize {
    if total_workers == 0 {
        return total_columns;
    }

    let total_workers = total_workers as usize;
    let worker_position = worker_position as usize;
    let columns_per_worker = total_columns / total_workers;
    let remainder = total_columns % total_workers;

    if worker_position < remainder {
        columns_per_worker + 1
    } else {
        columns_per_worker
    }
}

fn causal_self_attention(q: &Tensor2D, k: &Tensor2D, v: &Tensor2D, scale: f32) -> Result<Tensor2D> {
    causal_self_attention_gpu(q, k, v, scale)
}

fn attention_output(
    kv_cache: &mut KVCache,
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    layer_idx: usize,
    q_local: Tensor2D,
    k_local: Tensor2D,
    v_local: Tensor2D,
) -> Result<Tensor2D> {
    if config.num_heads == 0 || config.hidden_dim % config.num_heads != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Unsupported attention geometry: hidden_dim {} num_heads {}",
            config.hidden_dim, config.num_heads
        )));
    }
    if config.num_kv_heads == 0 || config.num_heads % config.num_kv_heads != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Unsupported grouped-query attention geometry: num_heads {} num_kv_heads {}",
            config.num_heads, config.num_kv_heads
        )));
    }

    let head_dim = config.hidden_dim / config.num_heads;
    let kv_hidden_dim = config.num_kv_heads * head_dim;
    if q_local.cols % head_dim != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width {} is not a multiple of head_dim {}",
            q_local.cols, head_dim
        )));
    }
    if k_local.cols % head_dim != 0 || v_local.cols % head_dim != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection widths must be multiples of head_dim {}: k={} v={}",
            head_dim, k_local.cols, v_local.cols
        )));
    }
    let expected_q_cols = partition_columns(config.hidden_dim, worker_position, total_workers);
    if q_local.cols != expected_q_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width mismatch: expected {}, got {}",
            expected_q_cols, q_local.cols
        )));
    }
    let expected_kv_cols = partition_columns(kv_hidden_dim, worker_position, total_workers);
    if k_local.cols != expected_kv_cols || v_local.cols != expected_kv_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection width mismatch: expected {}, got k={} v={}",
            expected_kv_cols, k_local.cols, v_local.cols
        )));
    }

    let q_heads_per_kv_head = config.num_heads / config.num_kv_heads;
    let q_head_start =
        partition_start(config.hidden_dim, worker_position, total_workers) / head_dim;
    let kv_head_start = partition_start(kv_hidden_dim, worker_position, total_workers) / head_dim;

    let positions: Vec<u32> = (0..q_local.rows as u32).collect();
    let q_rope = apply_rope(&q_local, &positions, head_dim, config.rope_base)?;
    let k_rope = apply_rope(&k_local, &positions, head_dim, config.rope_base)?;

    kv_cache.update_layer(layer_idx, k_rope.clone(), v_local.clone())?;

    let local_q_heads = q_local.cols / head_dim;
    let local_kv_heads = k_local.cols / head_dim;
    let q_heads = split_heads(&q_rope, local_q_heads, head_dim)?;
    let k_heads = split_heads(&k_rope, local_kv_heads, head_dim)?;
    let v_heads = split_heads(&v_local, local_kv_heads, head_dim)?;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output_heads = Vec::with_capacity(local_q_heads);
    for (local_q_idx, q_head) in q_heads.iter().enumerate() {
        let global_q_head = q_head_start + local_q_idx;
        let global_kv_head = global_q_head / q_heads_per_kv_head;
        if global_kv_head < kv_head_start || global_kv_head >= kv_head_start + local_kv_heads {
            return Err(crate::errors::AgentError::Execution(format!(
                "Local KV head ownership mismatch: q_head {} maps to kv_head {}, local kv range {}..{}",
                global_q_head,
                global_kv_head,
                kv_head_start,
                kv_head_start + local_kv_heads
            )));
        }
        let local_kv_idx = global_kv_head - kv_head_start;
        output_heads.push(causal_self_attention(
            q_head,
            &k_heads[local_kv_idx],
            &v_heads[local_kv_idx],
            scale,
        )?);
    }

    merge_heads(&output_heads)
}

/// Forward pass state for tensor-parallel inference
#[derive(Clone)]
struct DeviceLayerWeights {
    w_q: CandleTensor,
    w_k: CandleTensor,
    w_v: CandleTensor,
    w_o: CandleTensor,
    w_up: CandleTensor,
    w_gate: CandleTensor,
    w_down: CandleTensor,
    attn_norm: CandleTensor,
    mlp_norm: CandleTensor,
}

#[derive(Clone)]
struct DeviceModelWeights {
    embedding: CandleTensor,
    layers: Vec<DeviceLayerWeights>,
    final_norm: CandleTensor,
    lm_head: CandleTensor,
}

impl DeviceModelWeights {
    fn from_host(weights: &ModelWeights) -> Result<Self> {
        let embedding = to_candle_2d(&weights.embedding)?;
        let final_norm = to_candle_1d(&weights.final_norm)?;
        let lm_head = to_candle_2d(&weights.lm_head)?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            layers.push(DeviceLayerWeights {
                w_q: to_candle_2d(&layer.w_q)?,
                w_k: to_candle_2d(&layer.w_k)?,
                w_v: to_candle_2d(&layer.w_v)?,
                w_o: to_candle_2d(&layer.w_o)?,
                w_up: to_candle_2d(&layer.w_up)?,
                w_gate: to_candle_2d(&layer.w_gate)?,
                w_down: to_candle_2d(&layer.w_down)?,
                attn_norm: to_candle_1d(&layer.attn_norm)?,
                mlp_norm: to_candle_1d(&layer.mlp_norm)?,
            });
        }
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

pub struct ForwardPass {
    device_weights: DeviceModelWeights,
    config: ModelConfig,
    allreduce_timeout: std::time::Duration,

    /// KV cache for attention
    pub kv_cache: KVCache,

    /// This worker's shard column range
    pub shard_start: usize,
    pub shard_end: usize,

    /// This worker's position in the tensor-parallel ring
    pub worker_position: u32,

    /// Total workers in ring
    pub total_workers: u32,

    /// Current position in sequence
    pub position: usize,
}

impl ForwardPass {
    /// Create a new forward pass state
    pub fn new(
        weights: ModelWeights,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        let config = weights.config.clone();
        let kv_config = super::kv_cache::KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 4096,
        };

        Ok(Self {
            device_weights: DeviceModelWeights::from_host(&weights)?,
            config,
            allreduce_timeout,
            kv_cache: KVCache::new(kv_config),
            shard_start,
            shard_end,
            worker_position,
            total_workers,
            position: 0,
        })
    }

    /// Embed input tokens
    fn embed(&self, tokens: &[u32]) -> Result<CandleTensor> {
        let ids = CandleTensor::from_vec(
            tokens.to_vec(),
            tokens.len(),
            self.device_weights.embedding.device(),
        )
        .map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        self.device_weights.embedding.embedding(&ids).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })
    }

    /// Run forward pass for a single layer with ring all-reduce
    ///
    /// This is the core tensor-parallel operation:
    /// 1. Compute partial attention/MLP with local shard
    /// 2. Ring all-reduce to combine results
    /// 3. Apply activation and normalization
    pub async fn forward_layer(
        &mut self,
        hidden: &CandleTensor,
        layer_idx: usize,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<CandleTensor> {
        let layer = self.device_weights.layers[layer_idx].clone();
        let config = self.config.clone();
        let start = Instant::now();
        let hidden_dims = hidden.dims();
        let hidden_rows = hidden_dims[0];
        let hidden_cols = hidden_dims[1];

        // 1. RMS Norm before attention
        let normed = rms_norm_candle(hidden, &layer.attn_norm, config.rms_norm_eps)?;

        // 2. Compute partial QKV projections (my columns only)
        let q_partial = normed.matmul(&layer.w_q).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let k_partial = normed.matmul(&layer.w_k).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let v_partial = normed.matmul(&layer.w_v).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        debug!(
            "Layer {} QKV partial computed: {}x{} -> {}x{}",
            layer_idx,
            hidden_rows,
            hidden_cols,
            q_partial.dims()[0],
            q_partial.dims()[1]
        );

        // 3. Compute causal attention on local heads only
        let attn_output = attention_output_device(
            &mut self.kv_cache,
            &config,
            self.worker_position,
            self.total_workers,
            layer_idx,
            q_partial,
            k_partial,
            v_partial,
        )?;

        // 6. Output projection (partial)
        let o_partial = attn_output.matmul(&layer.w_o).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let o_full = self
            .ring_allreduce_candle(&o_partial, worker_ring, job_id, layer_idx as u32)
            .await?;

        // 7. Residual connection
        let post_attn = hidden.broadcast_add(&o_full).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        // 8. MLP with SwiGLU
        let mlp_normed = rms_norm_candle(&post_attn, &layer.mlp_norm, config.rms_norm_eps)?;

        // Gate and up projections are column-parallel and stay local
        let gate_partial = mlp_normed.matmul(&layer.w_gate).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let up_partial = mlp_normed.matmul(&layer.w_up).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        // SwiGLU: silu(gate) * up
        let gate_activated = silu_candle(&gate_partial)?;
        let mlp_hidden = gate_activated.broadcast_mul(&up_partial).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        // Down projection (partial)
        let down_partial = mlp_hidden.matmul(&layer.w_down).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let down_full = self
            .ring_allreduce_candle(&down_partial, worker_ring, job_id, layer_idx as u32)
            .await?;

        // 9. Final residual
        let output = post_attn.broadcast_add(&down_full).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        debug!(
            "Layer {} forward complete in {:?}",
            layer_idx,
            start.elapsed()
        );

        Ok(output)
    }

    /// Compute logits from final hidden states
    pub fn compute_logits(&self, hidden: &CandleTensor) -> Result<Tensor1D> {
        // Take last token's hidden state
        let last_hidden = hidden.narrow(0, hidden.dims()[0] - 1, 1).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        // Project to vocabulary
        let logits_2d = last_hidden
            .matmul(&self.device_weights.lm_head)
            .map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;

        // Flatten to 1D
        Ok(Tensor1D::new(
            logits_2d
                .flatten_all()
                .map_err(|e| {
                    crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
                })?
                .to_vec1::<f32>()
                .map_err(|e| {
                    crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
                })?,
        ))
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
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<CandleTensor> {
        let start = Instant::now();
        self.kv_cache.clear();

        // 1. Embed tokens
        let mut hidden = self.embed(tokens)?;
        let hidden_dims = hidden.dims();
        debug!(
            "Embedded {} tokens -> {:?}",
            tokens.len(),
            (hidden_dims[0], hidden_dims[1])
        );

        // 2. Run through all layers
        for layer_idx in 0..self.config.num_layers {
            hidden = self
                .forward_layer(&hidden, layer_idx, worker_ring, job_id)
                .await?;
        }

        // 3. Apply final norm
        hidden = rms_norm_candle(
            &hidden,
            &self.device_weights.final_norm,
            self.config.rms_norm_eps,
        )?;

        info!(
            "Full forward pass complete: {} layers in {:?}",
            self.config.num_layers,
            start.elapsed()
        );

        Ok(hidden)
    }

    /// Generate next token using full forward pass
    pub async fn generate_next_token(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
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
    async fn ring_allreduce_candle(
        &self,
        tensor: &CandleTensor,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        layer_idx: u32,
    ) -> Result<CandleTensor> {
        // Convert to flat tensor for all-reduce
        let host = from_candle_2d(tensor)?;
        let flat = host.to_allreduce_tensor();

        // Perform ring all-reduce
        let reduced = worker_ring
            .ring_all_reduce_with_timeout(flat, job_id, layer_idx, self.allreduce_timeout)
            .await?;

        // Convert back to 2D
        let host = Tensor2D::from_allreduce_tensor(&reduced)?;
        to_candle_2d(&host)
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

fn attention_output_device(
    kv_cache: &mut KVCache,
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    layer_idx: usize,
    q_local: CandleTensor,
    k_local: CandleTensor,
    v_local: CandleTensor,
) -> Result<CandleTensor> {
    if config.num_heads == 0 || config.hidden_dim % config.num_heads != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Unsupported attention geometry: hidden_dim {} num_heads {}",
            config.hidden_dim, config.num_heads
        )));
    }
    if config.num_kv_heads == 0 || config.num_heads % config.num_kv_heads != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Unsupported grouped-query attention geometry: num_heads {} num_kv_heads {}",
            config.num_heads, config.num_kv_heads
        )));
    }

    let head_dim = config.hidden_dim / config.num_heads;
    let kv_hidden_dim = config.num_kv_heads * head_dim;
    let q_dims = q_local.dims();
    let k_dims = k_local.dims();
    let v_dims = v_local.dims();
    let q_rows = q_dims[0];
    let q_cols = q_dims[1];
    let k_cols = k_dims[1];
    let v_cols = v_dims[1];

    if q_cols % head_dim != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width {} is not a multiple of head_dim {}",
            q_cols, head_dim
        )));
    }
    if k_cols % head_dim != 0 || v_cols % head_dim != 0 {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection widths must be multiples of head_dim {}: k={} v={}",
            head_dim, k_cols, v_cols
        )));
    }
    let expected_q_cols = partition_columns(config.hidden_dim, worker_position, total_workers);
    if q_cols != expected_q_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width mismatch: expected {}, got {}",
            expected_q_cols, q_cols
        )));
    }
    let expected_kv_cols = partition_columns(kv_hidden_dim, worker_position, total_workers);
    if k_cols != expected_kv_cols || v_cols != expected_kv_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection width mismatch: expected {}, got k={} v={}",
            expected_kv_cols, k_cols, v_cols
        )));
    }

    let q_heads_per_kv_head = config.num_heads / config.num_kv_heads;
    let q_head_start =
        partition_start(config.hidden_dim, worker_position, total_workers) / head_dim;
    let kv_head_start = partition_start(kv_hidden_dim, worker_position, total_workers) / head_dim;
    let positions: Vec<u32> = (0..q_rows as u32).collect();
    let q_rope = apply_rope_candle(
        &q_local,
        q_rows,
        q_cols,
        &positions,
        head_dim,
        config.rope_base,
    )?;
    let k_rope = apply_rope_candle(
        &k_local,
        q_rows,
        k_cols,
        &positions,
        head_dim,
        config.rope_base,
    )?;

    kv_cache.update_layer(
        layer_idx,
        from_candle_2d(&k_rope)?,
        from_candle_2d(&v_local)?,
    )?;

    let local_q_heads = q_cols / head_dim;
    let local_kv_heads = k_cols / head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output_heads = Vec::with_capacity(local_q_heads);

    for local_q_idx in 0..local_q_heads {
        let global_q_head = q_head_start + local_q_idx;
        let global_kv_head = global_q_head / q_heads_per_kv_head;
        if global_kv_head < kv_head_start || global_kv_head >= kv_head_start + local_kv_heads {
            return Err(crate::errors::AgentError::Execution(format!(
                "Local KV head ownership mismatch: q_head {} maps to kv_head {}, local kv range {}..{}",
                global_q_head,
                global_kv_head,
                kv_head_start,
                kv_head_start + local_kv_heads
            )));
        }
        let local_kv_idx = global_kv_head - kv_head_start;
        let q_head = q_rope
            .narrow(1, local_q_idx * head_dim, head_dim)
            .map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
        let k_head = k_rope
            .narrow(1, local_kv_idx * head_dim, head_dim)
            .map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
        let v_head = v_local
            .narrow(1, local_kv_idx * head_dim, head_dim)
            .map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
        output_heads.push(causal_self_attention_candle(
            &q_head, &k_head, &v_head, q_rows, head_dim, scale,
        )?);
    }

    let refs: Vec<&CandleTensor> = output_heads.iter().collect();
    CandleTensor::cat(&refs, 1).map_err(|e| {
        crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
    })
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
            num_heads: weights.config.num_kv_heads,
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
        self.kv_cache.clear();
        // Embed tokens
        let mut hidden = embed_tokens(&self.weights.embedding, tokens)?;

        // Run through all layers with the same causal attention block used in distributed mode
        for layer_idx in 0..self.weights.config.num_layers {
            let layer = &self.weights.layers[layer_idx];
            let config = &self.weights.config;

            // Attention block
            let normed = rms_norm(&hidden, &layer.attn_norm, config.rms_norm_eps)?;
            let q_local = matmul(&normed, &layer.w_q)?;
            let k_local = matmul(&normed, &layer.w_k)?;
            let v_local = matmul(&normed, &layer.w_v)?;
            let attn_output = attention_output(
                &mut self.kv_cache,
                config,
                0,
                1,
                layer_idx,
                q_local,
                k_local,
                v_local,
            )?;
            let o_partial = matmul(&attn_output, &layer.w_o)?;
            let post_attn = hidden.add(&o_partial)?;

            // MLP block
            let mlp_normed = rms_norm(&post_attn, &layer.mlp_norm, config.rms_norm_eps)?;
            let gate_partial = matmul(&mlp_normed, &layer.w_gate)?;
            let up_partial = matmul(&mlp_normed, &layer.w_up)?;
            let gate_activated = silu(&gate_partial);
            let mlp_hidden = gate_activated.mul(&up_partial)?;
            let mlp_out = matmul(&mlp_hidden, &layer.w_down)?;
            hidden = post_attn.add(&mlp_out)?;
        }

        // Final norm
        hidden = rms_norm(
            &hidden,
            &self.weights.final_norm,
            self.weights.config.rms_norm_eps,
        )?;

        Ok(hidden)
    }

    /// Generate next token locally
    pub fn generate_next_token(
        &mut self,
        tokens: &[u32],
        temperature: f32,
        top_p: f32,
    ) -> Result<u32> {
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

    fn create_test_weights(config: &ModelConfig, shard_cols: usize) -> ModelWeights {
        let mlp_shard_cols = shard_cols * config.intermediate_size / config.hidden_dim;
        let kv_shard_cols = config.num_kv_heads * (config.hidden_dim / config.num_heads);
        let layers = (0..config.num_layers)
            .map(|layer_idx| LayerWeights {
                layer_idx,
                w_q: Tensor2D::filled(config.hidden_dim, shard_cols, 0.1),
                w_k: Tensor2D::filled(config.hidden_dim, kv_shard_cols, 0.2),
                w_v: Tensor2D::filled(config.hidden_dim, kv_shard_cols, 0.3),
                w_o: Tensor2D::filled(shard_cols, config.hidden_dim, 0.4),
                w_up: Tensor2D::filled(config.hidden_dim, mlp_shard_cols, 0.5),
                w_gate: Tensor2D::filled(config.hidden_dim, mlp_shard_cols, 0.6),
                w_down: Tensor2D::filled(mlp_shard_cols, config.hidden_dim, 0.7),
                attn_norm: Tensor1D::new(vec![1.0; config.hidden_dim]),
                mlp_norm: Tensor1D::new(vec![1.0; config.hidden_dim]),
            })
            .collect();

        ModelWeights {
            model_id: "test-model".to_string(),
            embedding: Tensor2D::filled(config.vocab_size, config.hidden_dim, 0.05),
            layers,
            final_norm: Tensor1D::new(vec![1.0; config.hidden_dim]),
            lm_head: Tensor2D::filled(config.hidden_dim, config.vocab_size, 0.08),
            config: config.clone(),
        }
    }

    #[test]
    fn test_weight_memory_accounting() {
        let config = create_test_config();
        let weights = create_test_weights(&config, 32);

        assert_eq!(weights.layers.len(), config.num_layers);
        assert!(weights.memory_usage() > 0);
    }

    #[test]
    fn test_layer_weights_memory() {
        let layer = LayerWeights {
            layer_idx: 0,
            w_q: Tensor2D::filled(64, 32, 0.1),
            w_k: Tensor2D::filled(64, 32, 0.1),
            w_v: Tensor2D::filled(64, 32, 0.1),
            w_o: Tensor2D::filled(32, 64, 0.1),
            w_up: Tensor2D::filled(64, 64, 0.1),
            w_gate: Tensor2D::filled(64, 64, 0.1),
            w_down: Tensor2D::filled(64, 64, 0.1),
            attn_norm: Tensor1D::new(vec![1.0; 64]),
            mlp_norm: Tensor1D::new(vec![1.0; 64]),
        };
        let mem = layer.memory_usage();

        assert!(mem > 0);
    }

    #[test]
    fn test_local_forward_pass() {
        let config = create_test_config();
        let weights = create_test_weights(&config, 64);
        let mut forward = LocalForwardPass::new(weights);

        let tokens = vec![1, 2, 3];
        let hidden = forward.forward(&tokens).unwrap();

        assert_eq!(hidden.rows, 3); // seq_len
        assert_eq!(hidden.cols, 64); // hidden_dim
    }

    #[test]
    fn test_local_forward_supports_gqa() {
        let mut config = create_test_config();
        config.num_kv_heads = 2;
        let weights = create_test_weights(&config, 64);
        let mut forward = LocalForwardPass::new(weights);

        let hidden = forward.forward(&[1, 2, 3]).unwrap();
        assert_eq!(hidden.rows, 3);
        assert_eq!(hidden.cols, 64);
    }

    #[test]
    fn test_local_generate() {
        let config = create_test_config();
        let weights = create_test_weights(&config, 64);
        let mut forward = LocalForwardPass::new(weights);

        let tokens = vec![1, 2, 3];
        let next_token = forward.generate_next_token(&tokens, 1.0, 0.9).unwrap();

        // Token should be in vocab range
        assert!(next_token < 100);
    }

    #[test]
    fn test_forward_pass_creation() {
        let config = create_test_config();
        let weights = create_test_weights(&config, 16);
        let forward =
            ForwardPass::new(weights, 0, 0, 16, 4, std::time::Duration::from_secs(30)).unwrap();

        assert_eq!(forward.worker_position, 0);
        assert_eq!(forward.shard_start, 0);
        assert_eq!(forward.shard_end, 16);
        assert_eq!(forward.total_workers, 4);
    }

    #[test]
    fn test_local_forward_and_logits_match_single_final_norm_path() {
        let config = create_test_config();
        let weights = create_test_weights(&config, config.hidden_dim);
        let mut forward = LocalForwardPass::new(weights.clone());
        let tokens = vec![1, 2, 3];

        let hidden = forward.forward(&tokens).unwrap();
        let last_row = hidden.row(hidden.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, hidden.cols).unwrap();
        let logits_from_forward = matmul(&last_hidden, &weights.lm_head).unwrap();

        let hidden_double_norm =
            rms_norm(&hidden, &weights.final_norm, weights.config.rms_norm_eps).unwrap();
        let last_row_double_norm = hidden_double_norm.row(hidden_double_norm.rows - 1);
        let last_hidden_double_norm =
            Tensor2D::new(last_row_double_norm.to_vec(), 1, hidden_double_norm.cols).unwrap();
        let logits_double_norm = matmul(&last_hidden_double_norm, &weights.lm_head).unwrap();

        assert_eq!(logits_from_forward.rows, logits_double_norm.rows);
        assert_eq!(logits_from_forward.cols, logits_double_norm.cols);
        assert_ne!(
            logits_from_forward.data, logits_double_norm.data,
            "Applying final norm twice should change logits and must not happen in the distributed path"
        );
    }
}
