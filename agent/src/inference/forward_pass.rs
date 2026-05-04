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

use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::{RingAllReduceMetrics, WorkerRing};
use candle_core::{DType, Tensor as CandleTensor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::kv_cache::{KVCache, KVCacheConfig, KVCacheSnapshot};
use super::tensor_ops::{
    apply_rope, apply_rope_candle, candle_2d_from_collective_buffer_owned_like,
    collective_buffer_from_candle_2d, embed_tokens, from_candle_2d, matmul, rms_norm,
    rms_norm_candle, sample_greedy, sample_token, sample_token_device, silu, silu_candle,
    to_candle_1d, to_candle_2d, Tensor1D, Tensor2D,
};
use crate::inference::backend::BackendLogits;

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

#[derive(Clone)]
struct DeviceKVPage {
    keys: CandleTensor,
    values: CandleTensor,
    used_rows: usize,
}

impl DeviceKVPage {
    fn new(page_rows: usize, cols: usize, device: &candle_core::Device) -> Result<Self> {
        Ok(Self {
            keys: CandleTensor::zeros((page_rows, cols), DType::F32, device)
                .map_err(device_error)?,
            values: CandleTensor::zeros((page_rows, cols), DType::F32, device)
                .map_err(device_error)?,
            used_rows: 0,
        })
    }

    fn free_rows(&self, page_rows: usize) -> usize {
        page_rows.saturating_sub(self.used_rows)
    }
}

#[derive(Clone)]
struct DeviceLayerKVCache {
    pages: Vec<DeviceKVPage>,
    block_table: Vec<usize>,
    seq_len: usize,
    cols: usize,
    page_rows: usize,
}

impl DeviceLayerKVCache {
    fn new(page_rows: usize) -> Self {
        Self {
            pages: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            cols: 0,
            page_rows,
        }
    }

    fn ensure_page_shape(&mut self, cols: usize) -> Result<()> {
        if self.cols == 0 {
            self.cols = cols;
            return Ok(());
        }
        if self.cols != cols {
            return Err(AgentError::Execution(format!(
                "Device KV width mismatch: existing {} vs requested {}",
                self.cols, cols
            )));
        }
        Ok(())
    }

    fn ensure_writable_tail(&mut self, cols: usize, device: &candle_core::Device) -> Result<usize> {
        self.ensure_page_shape(cols)?;

        if self.block_table.is_empty() {
            if self.pages.is_empty() {
                self.pages
                    .push(DeviceKVPage::new(self.page_rows, cols, device)?);
            }
            self.block_table.push(0);
            return Ok(0);
        }

        let tail_index = *self
            .block_table
            .last()
            .ok_or_else(|| AgentError::Execution("Device KV block table is empty".to_string()))?;
        if self.pages[tail_index].free_rows(self.page_rows) > 0 {
            return Ok(tail_index);
        }

        let new_index = self.pages.len();
        self.pages
            .push(DeviceKVPage::new(self.page_rows, cols, device)?);
        self.block_table.push(new_index);
        Ok(new_index)
    }

    fn append(&mut self, new_keys: &CandleTensor, new_values: &CandleTensor) -> Result<()> {
        let key_dims = new_keys.dims();
        let value_dims = new_values.dims();
        if key_dims.len() != 2 || value_dims.len() != 2 {
            return Err(AgentError::Execution(format!(
                "Device KV append expects rank-2 tensors, got keys {:?} values {:?}",
                key_dims, value_dims
            )));
        }
        if key_dims != value_dims {
            return Err(AgentError::Execution(format!(
                "Device KV append shape mismatch: keys {:?} values {:?}",
                key_dims, value_dims
            )));
        }

        let new_rows = key_dims[0];
        let cols = key_dims[1];
        let mut appended_rows = 0;
        while appended_rows < new_rows {
            let page_index = self.ensure_writable_tail(cols, new_keys.device())?;
            let page = self.pages.get_mut(page_index).ok_or_else(|| {
                AgentError::Execution(format!("Invalid device KV page index {}", page_index))
            })?;
            let copy_rows = page
                .free_rows(self.page_rows)
                .min(new_rows.saturating_sub(appended_rows));
            let key_chunk = new_keys
                .narrow(0, appended_rows, copy_rows)
                .map_err(device_error)?;
            let value_chunk = new_values
                .narrow(0, appended_rows, copy_rows)
                .map_err(device_error)?;
            let row_ranges = &[page.used_rows..page.used_rows + copy_rows, 0..cols];
            page.keys = page
                .keys
                .slice_assign(row_ranges, &key_chunk)
                .map_err(device_error)?;
            page.values = page
                .values
                .slice_assign(row_ranges, &value_chunk)
                .map_err(device_error)?;
            page.used_rows += copy_rows;
            appended_rows += copy_rows;
            self.seq_len += copy_rows;
        }
        Ok(())
    }

    fn active_tensors(&self) -> Result<(CandleTensor, CandleTensor, usize)> {
        if self.seq_len == 0 || self.block_table.is_empty() {
            return Err(AgentError::Execution(
                "Requested device KV state before cache initialization".to_string(),
            ));
        }

        let mut key_chunks = Vec::with_capacity(self.block_table.len());
        let mut value_chunks = Vec::with_capacity(self.block_table.len());
        for &page_index in &self.block_table {
            let page = self.pages.get(page_index).ok_or_else(|| {
                AgentError::Execution(format!("Invalid device KV page index {}", page_index))
            })?;
            if page.used_rows == 0 {
                continue;
            }
            key_chunks.push(
                page.keys
                    .narrow(0, 0, page.used_rows)
                    .map_err(device_error)?,
            );
            value_chunks.push(
                page.values
                    .narrow(0, 0, page.used_rows)
                    .map_err(device_error)?,
            );
        }

        if key_chunks.is_empty() || value_chunks.is_empty() {
            return Err(AgentError::Execution(
                "Device KV block table contained no active rows".to_string(),
            ));
        }

        let active_keys = if key_chunks.len() == 1 {
            key_chunks[0].clone()
        } else {
            let key_refs = key_chunks.iter().collect::<Vec<_>>();
            CandleTensor::cat(&key_refs, 0).map_err(device_error)?
        };
        let active_values = if value_chunks.len() == 1 {
            value_chunks[0].clone()
        } else {
            let value_refs = value_chunks.iter().collect::<Vec<_>>();
            CandleTensor::cat(&value_refs, 0).map_err(device_error)?
        };
        Ok((active_keys, active_values, self.seq_len))
    }

    fn block_table_len(&self) -> usize {
        self.block_table.len()
    }

    fn clear(&mut self) {
        for page in &mut self.pages {
            page.used_rows = 0;
        }
        self.block_table.clear();
        self.seq_len = 0;
    }

    fn retain_suffix(&mut self, keep_rows: usize) -> Result<()> {
        if self.seq_len <= keep_rows {
            return Ok(());
        }
        if keep_rows == 0 {
            self.clear();
            return Ok(());
        }
        let (active_keys, active_values, active_rows) = self.active_tensors()?;
        let start_row = active_rows - keep_rows;
        let kept_keys = active_keys
            .narrow(0, start_row, keep_rows)
            .map_err(device_error)?;
        let kept_values = active_values
            .narrow(0, start_row, keep_rows)
            .map_err(device_error)?;
        self.clear();
        self.append(&kept_keys, &kept_values)?;
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.pages
            .len()
            .saturating_mul(self.page_rows)
            .saturating_mul(self.cols)
            .saturating_mul(std::mem::size_of::<f32>())
            .saturating_mul(2)
    }
}

#[derive(Clone)]
struct DeviceKVCache {
    layers: Vec<DeviceLayerKVCache>,
    base_position: usize,
    config: KVCacheConfig,
}

impl DeviceKVCache {
    fn new(config: KVCacheConfig) -> Self {
        let page_rows = config.live_page_tokens();
        Self {
            layers: (0..config.num_layers)
                .map(|_| DeviceLayerKVCache::new(page_rows))
                .collect(),
            base_position: 0,
            config,
        }
    }

    fn append_layer(
        &mut self,
        layer_idx: usize,
        keys: &CandleTensor,
        values: &CandleTensor,
    ) -> Result<()> {
        let rows = keys.dims().first().copied().ok_or_else(|| {
            AgentError::Execution("Device KV append received an empty key shape".to_string())
        })?;
        let max_seq_len = self.config.max_seq_len.max(1);
        let (keys, values, rows) = if rows > max_seq_len {
            let keep_rows = max_seq_len;
            let start_row = rows - keep_rows;
            (
                keys.narrow(0, start_row, keep_rows).map_err(device_error)?,
                values
                    .narrow(0, start_row, keep_rows)
                    .map_err(device_error)?,
                keep_rows,
            )
        } else {
            (keys.clone(), values.clone(), rows)
        };
        let target_layer_seq_len = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?
            .seq_len;
        if target_layer_seq_len.saturating_add(rows) > max_seq_len {
            let drop_rows = target_layer_seq_len.saturating_add(rows) - max_seq_len;
            self.retain_suffix(target_layer_seq_len.saturating_sub(drop_rows))?;
        }
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.append(&keys, &values)
    }

    fn layer_seq_len(&self, layer_idx: usize) -> Result<usize> {
        self.layers
            .get(layer_idx)
            .map(|layer| layer.seq_len)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))
    }

    fn get_layer_active_kv(&self, layer_idx: usize) -> Result<(CandleTensor, CandleTensor, usize)> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.active_tensors()
    }

    fn seq_len(&self) -> usize {
        self.layers.first().map(|layer| layer.seq_len).unwrap_or(0)
    }

    fn base_position(&self) -> usize {
        self.base_position
    }

    fn next_position(&self) -> usize {
        self.base_position.saturating_add(self.seq_len())
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.base_position = 0;
    }

    fn retain_suffix(&mut self, keep_rows: usize) -> Result<()> {
        let seq_len = self.seq_len();
        if seq_len <= keep_rows {
            return Ok(());
        }
        for layer in &mut self.layers {
            layer.retain_suffix(keep_rows)?;
        }
        self.base_position = self.base_position.saturating_add(seq_len - keep_rows);
        Ok(())
    }

    fn live_page_tokens(&self) -> usize {
        self.config.live_page_tokens()
    }

    fn live_block_table_len(&self, layer_idx: usize) -> Result<usize> {
        self.layers
            .get(layer_idx)
            .map(DeviceLayerKVCache::block_table_len)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))
    }

    fn memory_usage(&self) -> usize {
        self.layers
            .iter()
            .map(DeviceLayerKVCache::memory_usage)
            .sum()
    }

    fn export_snapshot(
        &self,
        next_position: u32,
        max_cached_tokens: Option<usize>,
    ) -> Result<Option<KVCacheSnapshot>> {
        if self.next_position() as u32 != next_position {
            return Err(AgentError::Execution(format!(
                "Device KV export next_position {} does not match cache next position {}",
                next_position,
                self.next_position()
            )));
        }
        if self.seq_len() == 0 {
            return Ok(None);
        }
        let mut host_cache = KVCache::new(self.config.clone());
        host_cache.set_base_position_for_restore(self.base_position);
        for layer_idx in 0..self.layers.len() {
            let layer_seq_len = self.layer_seq_len(layer_idx)?;
            if layer_seq_len == 0 {
                continue;
            }
            let (keys, values, _) = self.get_layer_active_kv(layer_idx)?;
            host_cache.append_layer(layer_idx, from_candle_2d(&keys)?, from_candle_2d(&values)?)?;
        }
        if let Some(limit) = max_cached_tokens {
            host_cache.retain_suffix(limit.max(1));
        }
        KVCacheSnapshot::from_cache(&host_cache, next_position).map(Some)
    }

    fn import_snapshot(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        snapshot.validate()?;
        let host_cache = snapshot.decode_cache()?;
        self.clear();
        self.base_position = host_cache.base_position();
        for (layer_idx, layer) in host_cache.layers.iter().enumerate() {
            match (&layer.keys, &layer.values) {
                (Some(keys), Some(values)) if layer.seq_len > 0 => {
                    let device_keys = to_candle_2d(keys)?;
                    let device_values = to_candle_2d(values)?;
                    self.append_layer(layer_idx, &device_keys, &device_values)?;
                }
                (None, None) => {}
                _ => {
                    return Err(AgentError::Execution(format!(
                        "Host KV snapshot is inconsistent on layer {}",
                        layer_idx
                    )))
                }
            }
        }
        if self.seq_len() as u32 != snapshot.sequence.cached_tokens {
            return Err(AgentError::Execution(format!(
                "Recovered device KV cache length {} does not match snapshot cached token count {}",
                self.seq_len(),
                snapshot.sequence.cached_tokens
            )));
        }
        if self.base_position as u32 != snapshot.sequence.first_cached_position() {
            return Err(AgentError::Execution(format!(
                "Recovered device KV cache base position {} does not match snapshot first cached position {}",
                self.base_position,
                snapshot.sequence.first_cached_position()
            )));
        }
        Ok(())
    }
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

fn causal_attention_with_prefix(
    q: &Tensor2D,
    k: &Tensor2D,
    v: &Tensor2D,
    scale: f32,
    prefix_len: usize,
) -> Result<Tensor2D> {
    if q.cols != k.cols || k.cols != v.cols || k.rows != v.rows {
        return Err(crate::errors::AgentError::Execution(format!(
            "Attention shape mismatch: q {}x{}, k {}x{}, v {}x{}",
            q.rows, q.cols, k.rows, k.cols, v.rows, v.cols
        )));
    }

    let mut out = Tensor2D::zeros(q.rows, v.cols);
    for q_row in 0..q.rows {
        let q_slice = q.row(q_row);
        let max_k = prefix_len + q_row + 1;
        if max_k > k.rows {
            return Err(crate::errors::AgentError::Execution(format!(
                "Attention prefix exceeds cached sequence: prefix {} query_row {} cache_rows {}",
                prefix_len, q_row, k.rows
            )));
        }

        let mut scores = Vec::with_capacity(max_k);
        let mut max_score = f32::NEG_INFINITY;
        for k_row in 0..max_k {
            let k_slice = k.row(k_row);
            let score = q_slice
                .iter()
                .zip(k_slice.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>()
                * scale;
            max_score = max_score.max(score);
            scores.push(score);
        }

        let mut exp_sum = 0.0;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            exp_sum += *score;
        }

        for (k_row, prob) in scores.into_iter().enumerate() {
            let weight = prob / exp_sum;
            let v_slice = v.row(k_row);
            for (col_idx, value) in v_slice.iter().enumerate() {
                out.data[q_row * out.cols + col_idx] += weight * value;
            }
        }
    }

    Ok(out)
}

fn validate_positions(absolute_positions: &[u32], expected_start: usize) -> Result<()> {
    if absolute_positions.is_empty() {
        return Err(crate::errors::AgentError::Execution(
            "Attention positions cannot be empty".to_string(),
        ));
    }

    for (offset, position) in absolute_positions.iter().enumerate() {
        let expected = expected_start + offset;
        if *position as usize != expected {
            return Err(crate::errors::AgentError::Execution(format!(
                "Non-contiguous attention positions: expected {}, got {}",
                expected, position
            )));
        }
    }

    Ok(())
}

fn build_positions(start: usize, count: usize) -> Vec<u32> {
    (start..start + count)
        .map(|position| position as u32)
        .collect()
}

fn attention_output(
    kv_cache: &mut KVCache,
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    layer_idx: usize,
    absolute_positions: &[u32],
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

    let current_layer_seq_len = kv_cache.layer_seq_len(layer_idx)?;
    let incoming_rows = absolute_positions.len();
    let max_seq_len = kv_cache.config.max_seq_len.max(1);
    if current_layer_seq_len.saturating_add(incoming_rows) > max_seq_len {
        let drop_rows = current_layer_seq_len.saturating_add(incoming_rows) - max_seq_len;
        kv_cache.retain_suffix(current_layer_seq_len.saturating_sub(drop_rows));
    }
    let current_layer_seq_len = kv_cache.layer_seq_len(layer_idx)?;
    let incoming_rows = absolute_positions.len();
    let max_seq_len = kv_cache.config.max_seq_len.max(1);
    if current_layer_seq_len.saturating_add(incoming_rows) > max_seq_len {
        let drop_rows = current_layer_seq_len.saturating_add(incoming_rows) - max_seq_len;
        kv_cache.retain_suffix(current_layer_seq_len.saturating_sub(drop_rows));
    }
    let cache_prefix_len = kv_cache.layer_seq_len(layer_idx)?;
    validate_positions(
        absolute_positions,
        kv_cache.base_position().saturating_add(cache_prefix_len),
    )?;
    let q_rope = apply_rope(&q_local, absolute_positions, head_dim, config.rope_base)?;
    let k_rope = apply_rope(&k_local, absolute_positions, head_dim, config.rope_base)?;

    kv_cache.append_layer(layer_idx, k_rope, v_local)?;
    let (cached_k, cached_v) = kv_cache.get_layer_kv(layer_idx)?;

    let local_q_heads = q_local.cols / head_dim;
    let local_kv_heads = cached_k.cols / head_dim;
    let q_heads = split_heads(&q_rope, local_q_heads, head_dim)?;
    let k_heads = split_heads(cached_k, local_kv_heads, head_dim)?;
    let v_heads = split_heads(cached_v, local_kv_heads, head_dim)?;
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
        output_heads.push(causal_attention_with_prefix(
            q_head,
            &k_heads[local_kv_idx],
            &v_heads[local_kv_idx],
            scale,
            cache_prefix_len,
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

fn candle_tensor_memory_usage_bytes(tensor: &CandleTensor) -> usize {
    tensor
        .dims()
        .iter()
        .copied()
        .product::<usize>()
        .saturating_mul(std::mem::size_of::<f32>())
}

impl DeviceLayerWeights {
    fn memory_usage_bytes(&self) -> usize {
        candle_tensor_memory_usage_bytes(&self.w_q)
            .saturating_add(candle_tensor_memory_usage_bytes(&self.w_k))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.w_v))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.w_o))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.w_up))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.w_gate))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.w_down))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.attn_norm))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.mlp_norm))
    }
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

    fn memory_usage_bytes(&self) -> usize {
        candle_tensor_memory_usage_bytes(&self.embedding)
            .saturating_add(
                self.layers
                    .iter()
                    .map(DeviceLayerWeights::memory_usage_bytes)
                    .sum(),
            )
            .saturating_add(candle_tensor_memory_usage_bytes(&self.final_norm))
            .saturating_add(candle_tensor_memory_usage_bytes(&self.lm_head))
    }
}

#[derive(Clone)]
pub struct SharedModelResidency {
    model_id: String,
    config: ModelConfig,
    device_weights: Arc<DeviceModelWeights>,
    resident_bytes: usize,
}

impl SharedModelResidency {
    pub fn from_host(weights: ModelWeights) -> Result<Self> {
        let model_id = weights.model_id.clone();
        let config = weights.config.clone();
        let device_weights = Arc::new(DeviceModelWeights::from_host(&weights)?);
        let resident_bytes = device_weights.memory_usage_bytes();
        Ok(Self {
            model_id,
            config,
            device_weights,
            resident_bytes,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

pub struct ForwardPass {
    device_weights: Arc<DeviceModelWeights>,
    config: ModelConfig,
    allreduce_timeout: std::time::Duration,

    /// KV cache for attention
    device_kv_cache: DeviceKVCache,

    /// This worker's shard column range
    pub shard_start: usize,
    pub shard_end: usize,

    /// This worker's position in the tensor-parallel ring
    pub worker_position: u32,

    /// Total workers in ring
    pub total_workers: u32,

    /// Current position in sequence
    pub position: usize,

    /// Aggregated ring all-reduce timings from the last forward invocation.
    pub last_allreduce_metrics: RingAllReduceMetrics,
}

impl ForwardPass {
    pub fn from_residency(
        residency: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        let config = residency.config().clone();
        let kv_config = KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 4096,
        };

        Ok(Self {
            device_weights: Arc::clone(&residency.device_weights),
            config,
            allreduce_timeout,
            device_kv_cache: DeviceKVCache::new(kv_config),
            shard_start,
            shard_end,
            worker_position,
            total_workers,
            position: 0,
            last_allreduce_metrics: RingAllReduceMetrics::default(),
        })
    }

    /// Create a new forward pass state
    pub fn new(
        weights: ModelWeights,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        let residency = Arc::new(SharedModelResidency::from_host(weights)?);
        Self::from_residency(
            residency,
            worker_position,
            shard_start,
            shard_end,
            total_workers,
            allreduce_timeout,
        )
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
        absolute_positions: &[u32],
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
            &mut self.device_kv_cache,
            &config,
            self.worker_position,
            self.total_workers,
            layer_idx,
            absolute_positions,
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
    pub fn compute_logits_tensor(&self, hidden: &CandleTensor) -> Result<CandleTensor> {
        // Take last token's hidden state
        let last_hidden = hidden.narrow(0, hidden.dims()[0] - 1, 1).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        last_hidden
            .matmul(&self.device_weights.lm_head)
            .map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })
    }

    pub fn compute_logits(&self, hidden: &CandleTensor) -> Result<BackendLogits> {
        Ok(BackendLogits::Device(self.compute_logits_tensor(hidden)?))
    }

    /// Sample next token from logits
    pub fn sample(
        &self,
        logits: &BackendLogits,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<u32> {
        match logits {
            BackendLogits::Host(logits) => Ok({
                if temperature <= 0.0 || temperature == 1.0 && top_p >= 1.0 {
                    sample_greedy(logits)
                } else {
                    sample_token(logits, temperature, top_p, seed)
                }
            }),
            BackendLogits::Device(logits) => {
                if temperature <= 0.0 || temperature == 1.0 && top_p >= 1.0 {
                    let token_ids = logits
                        .argmax(1)
                        .and_then(|idx| idx.to_vec1::<u32>())
                        .map_err(device_error)
                        .or_else(|device_err| {
                            let host_logits = Tensor1D::new(
                                logits
                                    .flatten_all()
                                    .map_err(device_error)?
                                    .to_vec1::<f32>()
                                    .map_err(device_error)?,
                            );
                            warn!(
                                error = %device_err,
                                "device argmax sampling failed, falling back to host logits"
                            );
                            Ok::<Vec<u32>, AgentError>(vec![sample_greedy(&host_logits)])
                        })?;
                    token_ids.into_iter().next().ok_or_else(|| {
                        AgentError::Execution("Device argmax produced no token ids".to_string())
                    })
                } else {
                    sample_token_device(logits, temperature, top_p, seed).or_else(|device_err| {
                        let host_logits = Tensor1D::new(
                            logits
                                .flatten_all()
                                .map_err(device_error)?
                                .to_vec1::<f32>()
                                .map_err(device_error)?,
                        );
                        warn!(
                            error = %device_err,
                            "device stochastic sampling failed, falling back to host logits"
                        );
                        Ok(sample_token(&host_logits, temperature, top_p, seed))
                    })
                }
            }
        }
    }

    /// Run full forward pass for all layers
    pub async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits> {
        if tokens.is_empty() {
            return Err(crate::errors::AgentError::Execution(
                "Cannot prefill an empty prompt without an explicit BOS policy".to_string(),
            ));
        }

        self.clear_cache();
        let window = self.device_kv_cache.config.max_seq_len.max(1);
        let start = tokens.len().saturating_sub(window);
        self.device_kv_cache.base_position = start;
        let hidden = self
            .forward_tokens(&tokens[start..], start, worker_ring, job_id)
            .await?;
        self.position = tokens.len();
        debug_assert_eq!(self.device_kv_cache.next_position(), self.position);
        self.compute_logits(&hidden)
    }

    pub async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits> {
        if self.position == 0 {
            return Err(crate::errors::AgentError::Execution(
                "Decode step requested before prompt prefill".to_string(),
            ));
        }
        if self.device_kv_cache.next_position() != self.position {
            return Err(crate::errors::AgentError::Execution(format!(
                "Forward pass position {} diverged from KV cache next position {}",
                self.position,
                self.device_kv_cache.next_position()
            )));
        }

        let hidden = self
            .forward_tokens(&[token], self.position, worker_ring, job_id)
            .await?;
        self.position += 1;
        self.compute_logits(&hidden)
    }

    pub async fn decode_microbatch(
        backends: &mut [&mut crate::inference::backend::CandleExecutionBackend],
        tokens: &[u32],
        job_ids: &[Uuid],
        worker_ring: &mut WorkerRing<'_>,
    ) -> Result<Vec<BackendLogits>> {
        if backends.is_empty() || tokens.is_empty() {
            return Ok(Vec::new());
        }
        if backends.len() != tokens.len() || backends.len() != job_ids.len() {
            return Err(crate::errors::AgentError::Execution(format!(
                "Decode microbatch shape mismatch: backends={} tokens={} job_ids={}",
                backends.len(),
                tokens.len(),
                job_ids.len()
            )));
        }

        let (config, device_weights, worker_position, total_workers, allreduce_timeout) = {
            let template = &backends[0].forward_pass;
            (
                template.config.clone(),
                template.device_weights.clone(),
                template.worker_position,
                template.total_workers,
                template.allreduce_timeout,
            )
        };
        let batch_job_id = job_ids[0];

        let mut positions = Vec::with_capacity(backends.len());
        for backend in backends.iter() {
            let forward_pass = &backend.forward_pass;
            if forward_pass.position == 0 {
                return Err(crate::errors::AgentError::Execution(
                    "Decode microbatch requested before prompt prefill".to_string(),
                ));
            }
            if forward_pass.device_kv_cache.next_position() != forward_pass.position {
                return Err(crate::errors::AgentError::Execution(format!(
                    "Forward pass position {} diverged from KV cache next position {}",
                    forward_pass.position,
                    forward_pass.device_kv_cache.next_position()
                )));
            }
            positions.push(forward_pass.position as u32);
        }

        let ids = CandleTensor::from_vec(
            tokens.to_vec(),
            tokens.len(),
            device_weights.embedding.device(),
        )
        .map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let mut hidden = device_weights.embedding.embedding(&ids).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let mut metrics = RingAllReduceMetrics::default();

        for layer_idx in 0..config.num_layers {
            let layer = device_weights.layers[layer_idx].clone();
            let normed = rms_norm_candle(&hidden, &layer.attn_norm, config.rms_norm_eps)?;
            let q_partial = normed.matmul(&layer.w_q).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let k_partial = normed.matmul(&layer.w_k).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let v_partial = normed.matmul(&layer.w_v).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;

            let mut kv_caches = backends
                .iter_mut()
                .map(|backend| &mut backend.forward_pass.device_kv_cache)
                .collect::<Vec<_>>();
            let attn_output = attention_output_device_batch(
                &mut kv_caches,
                &config,
                worker_position,
                total_workers,
                layer_idx,
                &positions,
                q_partial,
                k_partial,
                v_partial,
            )?;

            let o_partial = attn_output.matmul(&layer.w_o).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let o_full = Self::ring_allreduce_candle_batch(
                &o_partial,
                worker_ring,
                batch_job_id,
                layer_idx as u32,
                allreduce_timeout,
                &mut metrics,
            )
            .await?;
            let post_attn = hidden.broadcast_add(&o_full).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;

            let mlp_normed = rms_norm_candle(&post_attn, &layer.mlp_norm, config.rms_norm_eps)?;
            let gate_partial = mlp_normed.matmul(&layer.w_gate).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let up_partial = mlp_normed.matmul(&layer.w_up).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let gate_activated = silu_candle(&gate_partial)?;
            let mlp_hidden = gate_activated.broadcast_mul(&up_partial).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let down_partial = mlp_hidden.matmul(&layer.w_down).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let down_full = Self::ring_allreduce_candle_batch(
                &down_partial,
                worker_ring,
                batch_job_id,
                layer_idx as u32,
                allreduce_timeout,
                &mut metrics,
            )
            .await?;
            hidden = post_attn.broadcast_add(&down_full).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
        }

        hidden = rms_norm_candle(&hidden, &device_weights.final_norm, config.rms_norm_eps)?;
        let logits_2d = hidden.matmul(&device_weights.lm_head).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let dims = logits_2d.dims();
        let mut logits = Vec::with_capacity(dims[0]);
        for row_idx in 0..dims[0] {
            let row = logits_2d.narrow(0, row_idx, 1).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            logits.push(BackendLogits::Device(row));
        }
        for backend in backends.iter_mut() {
            backend.forward_pass.position += 1;
            backend.forward_pass.last_allreduce_metrics = metrics;
        }
        Ok(logits)
    }

    /// Materialize a single collective buffer from a device tensor and restore
    /// the reduced result directly back onto the execution device.
    async fn ring_allreduce_candle(
        &mut self,
        tensor: &CandleTensor,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        layer_idx: u32,
    ) -> Result<CandleTensor> {
        if worker_ring.total_workers <= 1 {
            return Ok(tensor.clone());
        }
        let flat = collective_buffer_from_candle_2d(tensor)?;
        let reduced = worker_ring
            .ring_all_reduce_matrix_with_timeout(flat, job_id, layer_idx, self.allreduce_timeout)
            .await?;
        self.last_allreduce_metrics
            .accumulate(worker_ring.last_run_metrics());
        candle_2d_from_collective_buffer_owned_like(reduced, tensor)
    }

    async fn ring_allreduce_candle_batch(
        tensor: &CandleTensor,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        layer_idx: u32,
        allreduce_timeout: std::time::Duration,
        metrics: &mut RingAllReduceMetrics,
    ) -> Result<CandleTensor> {
        if worker_ring.total_workers <= 1 {
            return Ok(tensor.clone());
        }
        let flat = collective_buffer_from_candle_2d(tensor)?;
        let reduced = worker_ring
            .ring_all_reduce_matrix_with_timeout(flat, job_id, layer_idx, allreduce_timeout)
            .await?;
        metrics.accumulate(worker_ring.last_run_metrics());
        candle_2d_from_collective_buffer_owned_like(reduced, tensor)
    }

    /// Clear KV cache (for new sequence)
    pub fn clear_cache(&mut self) {
        self.device_kv_cache.clear();
        self.position = 0;
    }

    /// Get current KV cache memory usage
    pub fn live_kv_cache_bytes(&self) -> usize {
        self.device_kv_cache.memory_usage()
    }

    pub fn logical_kv_tokens(&self) -> usize {
        self.position
            .saturating_sub(self.device_kv_cache.base_position())
    }

    pub fn cache_seq_len(&self) -> usize {
        self.device_kv_cache.seq_len()
    }

    pub fn export_kv_cache_snapshot(
        &self,
        max_cached_tokens: Option<usize>,
    ) -> Result<Option<KVCacheSnapshot>> {
        self.device_kv_cache
            .export_snapshot(self.position as u32, max_cached_tokens)
    }

    pub fn import_kv_cache_snapshot(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        self.device_kv_cache.import_snapshot(snapshot)?;
        self.position = snapshot.sequence.next_position as usize;
        Ok(())
    }

    async fn forward_tokens(
        &mut self,
        tokens: &[u32],
        absolute_position_start: usize,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<CandleTensor> {
        if tokens.is_empty() {
            return Err(crate::errors::AgentError::Execution(
                "Forward pass requires at least one token".to_string(),
            ));
        }
        let start = Instant::now();
        let absolute_positions = build_positions(absolute_position_start, tokens.len());
        self.last_allreduce_metrics = RingAllReduceMetrics::default();

        let mut hidden = self.embed(tokens)?;
        let hidden_dims = hidden.dims();
        debug!(
            "Embedded {} tokens -> {:?} at positions {}..{}",
            tokens.len(),
            (hidden_dims[0], hidden_dims[1]),
            absolute_position_start,
            absolute_position_start + tokens.len() - 1
        );

        for layer_idx in 0..self.config.num_layers {
            hidden = self
                .forward_layer(&hidden, layer_idx, &absolute_positions, worker_ring, job_id)
                .await?;
        }

        hidden = rms_norm_candle(
            &hidden,
            &self.device_weights.final_norm,
            self.config.rms_norm_eps,
        )?;

        info!(
            positions = format!(
                "{}..{}",
                absolute_position_start,
                absolute_position_start + tokens.len() - 1
            ),
            layers = self.config.num_layers,
            elapsed_ms = start.elapsed().as_millis(),
            "Forward pass segment complete"
        );

        Ok(hidden)
    }
}

fn attention_output_device(
    kv_cache: &mut DeviceKVCache,
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    layer_idx: usize,
    absolute_positions: &[u32],
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

    let cache_prefix_len = kv_cache.layer_seq_len(layer_idx)?;
    validate_positions(
        absolute_positions,
        kv_cache.base_position().saturating_add(cache_prefix_len),
    )?;
    let q_rope = apply_rope_candle(
        &q_local,
        q_rows,
        q_cols,
        absolute_positions,
        head_dim,
        config.rope_base,
    )?;
    let k_rope = apply_rope_candle(
        &k_local,
        q_rows,
        k_cols,
        absolute_positions,
        head_dim,
        config.rope_base,
    )?;

    kv_cache.append_layer(layer_idx, &k_rope, &v_local)?;
    attention_output_device_cached(
        kv_cache,
        config,
        worker_position,
        total_workers,
        layer_idx,
        cache_prefix_len,
        &q_rope,
    )
}

fn attention_output_device_batch(
    kv_caches: &mut [&mut DeviceKVCache],
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    layer_idx: usize,
    absolute_positions: &[u32],
    q_local: CandleTensor,
    k_local: CandleTensor,
    v_local: CandleTensor,
) -> Result<CandleTensor> {
    let batch_size = q_local.dims()[0];
    if kv_caches.len() != batch_size || absolute_positions.len() != batch_size {
        return Err(crate::errors::AgentError::Execution(format!(
            "Batched attention state mismatch: caches={} positions={} batch={}",
            kv_caches.len(),
            absolute_positions.len(),
            batch_size
        )));
    }
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

    let q_rope = apply_rope_candle(
        &q_local,
        batch_size,
        q_cols,
        absolute_positions,
        head_dim,
        config.rope_base,
    )?;
    let k_rope = apply_rope_candle(
        &k_local,
        batch_size,
        k_cols,
        absolute_positions,
        head_dim,
        config.rope_base,
    )?;

    let mut outputs = Vec::with_capacity(batch_size);
    for row_idx in 0..batch_size {
        let current_layer_seq_len = kv_caches[row_idx].layer_seq_len(layer_idx)?;
        let max_seq_len = kv_caches[row_idx].config.max_seq_len.max(1);
        if current_layer_seq_len.saturating_add(1) > max_seq_len {
            let drop_rows = current_layer_seq_len.saturating_add(1) - max_seq_len;
            kv_caches[row_idx].retain_suffix(current_layer_seq_len.saturating_sub(drop_rows))?;
        }
        let prefix_len = kv_caches[row_idx].layer_seq_len(layer_idx)?;
        validate_positions(
            &[absolute_positions[row_idx]],
            kv_caches[row_idx]
                .base_position()
                .saturating_add(prefix_len),
        )?;
        let q_row = q_rope.narrow(0, row_idx, 1).map_err(device_error)?;
        let k_row = k_rope.narrow(0, row_idx, 1).map_err(device_error)?;
        let v_row = v_local.narrow(0, row_idx, 1).map_err(device_error)?;
        kv_caches[row_idx].append_layer(layer_idx, &k_row, &v_row)?;
        outputs.push(attention_output_device_cached(
            kv_caches[row_idx],
            config,
            worker_position,
            total_workers,
            layer_idx,
            prefix_len,
            &q_row,
        )?);
    }

    let output_refs = outputs.iter().collect::<Vec<_>>();
    CandleTensor::cat(&output_refs, 0).map_err(device_error)
}

fn attention_output_device_cached(
    kv_cache: &DeviceKVCache,
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    layer_idx: usize,
    cache_prefix_len: usize,
    q_rope: &CandleTensor,
) -> Result<CandleTensor> {
    let q_dims = q_rope.dims();
    if q_dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Device attention expects rank-2 query tensor, got {:?}",
            q_dims
        )));
    }
    let q_rows = q_dims[0];
    let q_cols = q_dims[1];
    let head_dim = config.hidden_dim / config.num_heads;
    let local_q_heads = q_cols / head_dim;
    let q_head_start =
        partition_start(config.hidden_dim, worker_position, total_workers) / head_dim;
    let local_kv_indices = build_local_kv_head_indices(
        config,
        worker_position,
        total_workers,
        local_q_heads,
        q_head_start,
    )?;

    let (cached_k, cached_v, cached_seq_len) = kv_cache.get_layer_active_kv(layer_idx)?;
    let k_dims = cached_k.dims();
    let local_kv_heads = k_dims[1] / head_dim;
    if local_kv_heads == 0 {
        return Err(AgentError::Execution(format!(
            "Device attention found zero local KV heads on layer {}",
            layer_idx
        )));
    }

    let q_heads = q_rope
        .reshape((q_rows, local_q_heads, head_dim))
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)?;
    let k_heads = cached_k
        .reshape((cached_seq_len, local_kv_heads, head_dim))
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)?;
    let v_heads = cached_v
        .reshape((cached_seq_len, local_kv_heads, head_dim))
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)?;

    let kv_indices = CandleTensor::from_vec(local_kv_indices, local_q_heads, q_rope.device())
        .map_err(device_error)?;
    let selected_k = k_heads
        .index_select(&kv_indices, 0)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)?;
    let selected_v = v_heads
        .index_select(&kv_indices, 0)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)?;
    let scores = q_heads
        .matmul(&selected_k.transpose(1, 2).map_err(device_error)?)
        .map_err(device_error)?
        .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
        .map_err(device_error)?;

    let mask = causal_attention_mask(q_rows, cached_seq_len, cache_prefix_len, q_rope.device())?;
    let masked = scores.broadcast_add(&mask).map_err(device_error)?;
    let probs = candle_nn::ops::softmax(&masked, 2).map_err(device_error)?;
    probs
        .matmul(&selected_v)
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .reshape((q_rows, local_q_heads * head_dim))
        .map_err(device_error)
}

fn build_local_kv_head_indices(
    config: &ModelConfig,
    worker_position: u32,
    total_workers: u32,
    local_q_heads: usize,
    q_head_start: usize,
) -> Result<Vec<u32>> {
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_hidden_dim = config.num_kv_heads * head_dim;
    let local_kv_heads =
        partition_columns(kv_hidden_dim, worker_position, total_workers) / head_dim;
    let q_heads_per_kv_head = config.num_heads / config.num_kv_heads;
    let kv_head_start = partition_start(kv_hidden_dim, worker_position, total_workers) / head_dim;
    let mut indices = Vec::with_capacity(local_q_heads);
    for local_q_idx in 0..local_q_heads {
        let global_q_head = q_head_start + local_q_idx;
        let global_kv_head = global_q_head / q_heads_per_kv_head;
        if global_kv_head < kv_head_start || global_kv_head >= kv_head_start + local_kv_heads {
            return Err(AgentError::Execution(format!(
                "Local KV head ownership mismatch: q_head {} maps to kv_head {}, local kv range {}..{}",
                global_q_head,
                global_kv_head,
                kv_head_start,
                kv_head_start + local_kv_heads
            )));
        }
        indices.push((global_kv_head - kv_head_start) as u32);
    }
    Ok(indices)
}

fn causal_attention_mask(
    q_rows: usize,
    cached_seq_len: usize,
    cache_prefix_len: usize,
    device: &candle_core::Device,
) -> Result<CandleTensor> {
    let mut mask = vec![f32::NEG_INFINITY; q_rows * cached_seq_len];
    for q_row in 0..q_rows {
        let max_k = cache_prefix_len + q_row + 1;
        if max_k > cached_seq_len {
            return Err(AgentError::Execution(format!(
                "Attention prefix exceeds cache: prefix {} + query_row {} > cached_seq_len {}",
                cache_prefix_len, q_row, cached_seq_len
            )));
        }
        for k_row in 0..max_k {
            mask[q_row * cached_seq_len + k_row] = 0.0;
        }
    }
    CandleTensor::from_vec(mask, (1, q_rows, cached_seq_len), device).map_err(device_error)
}

fn device_error(err: candle_core::Error) -> AgentError {
    AgentError::Execution(format!("GPU tensor backend error: {}", err))
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
        let window = self.kv_cache.config.max_seq_len.max(1);
        let start = tokens.len().saturating_sub(window);
        self.kv_cache.set_base_position_for_restore(start);
        let hidden = self.forward_tokens(&tokens[start..], start)?;
        self.position = tokens.len();
        debug_assert_eq!(self.kv_cache.next_position(), self.position);
        Ok(hidden)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor1D> {
        if tokens.is_empty() {
            return Err(crate::errors::AgentError::Execution(
                "Cannot prefill an empty prompt without an explicit BOS policy".to_string(),
            ));
        }

        self.kv_cache.clear();
        let window = self.kv_cache.config.max_seq_len.max(1);
        let start = tokens.len().saturating_sub(window);
        self.kv_cache.set_base_position_for_restore(start);
        let hidden = self.forward_tokens(&tokens[start..], start)?;
        self.position = tokens.len();
        debug_assert_eq!(self.kv_cache.next_position(), self.position);
        let last_row = hidden.row(hidden.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, hidden.cols)?;
        let logits_2d = matmul(&last_hidden, &self.weights.lm_head)?;
        Ok(Tensor1D::new(logits_2d.data))
    }

    pub fn decode_step(&mut self, token: u32) -> Result<Tensor1D> {
        if self.position == 0 {
            return Err(crate::errors::AgentError::Execution(
                "Decode step requested before prompt prefill".to_string(),
            ));
        }
        if self.kv_cache.next_position() != self.position {
            return Err(crate::errors::AgentError::Execution(format!(
                "Local forward pass position {} diverged from KV cache next position {}",
                self.position,
                self.kv_cache.next_position()
            )));
        }

        let hidden = self.forward_tokens(&[token], self.position)?;
        self.position += 1;
        let last_row = hidden.row(hidden.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, hidden.cols)?;
        let logits_2d = matmul(&last_hidden, &self.weights.lm_head)?;
        Ok(Tensor1D::new(logits_2d.data))
    }

    fn forward_tokens(
        &mut self,
        tokens: &[u32],
        absolute_position_start: usize,
    ) -> Result<Tensor2D> {
        if tokens.is_empty() {
            return Err(crate::errors::AgentError::Execution(
                "Forward pass requires at least one token".to_string(),
            ));
        }
        let absolute_positions = build_positions(absolute_position_start, tokens.len());

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
                &absolute_positions,
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
        let logits = self.prefill(tokens)?;

        // Sample
        let seed = self.position as u64;
        let next_token = if temperature <= 0.0 {
            sample_greedy(&logits)
        } else {
            sample_token(&logits, temperature, top_p, seed)
        };

        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::CandleExecutionBackend;
    use crate::inference::engine::InferenceRuntimeMode;
    use crate::network::{TensorPlane, TensorPlaneConfig};
    use crate::provider::ExecutionProviderKind;
    use libp2p::PeerId;
    use uuid::Uuid;

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
    fn test_local_incremental_decode_matches_full_recompute() {
        let config = create_test_config();
        let weights = create_test_weights(&config, config.hidden_dim);
        let prompt = vec![1, 2, 3];

        let mut incremental = LocalForwardPass::new(weights.clone());
        let first_logits = incremental.prefill(&prompt).unwrap();
        let first_token = sample_greedy(&first_logits);
        let decode_logits = incremental.decode_step(first_token).unwrap();

        let mut reference = LocalForwardPass::new(weights.clone());
        let hidden = reference
            .forward(&[prompt.clone(), vec![first_token]].concat())
            .unwrap();
        let last_row = hidden.row(hidden.rows - 1);
        let last_hidden = Tensor2D::new(last_row.to_vec(), 1, hidden.cols).unwrap();
        let reference_logits = Tensor1D::new(matmul(&last_hidden, &weights.lm_head).unwrap().data);

        assert_eq!(decode_logits.data, reference_logits.data);
        assert_eq!(incremental.kv_cache.seq_len(), prompt.len() + 1);
        assert_eq!(incremental.position, prompt.len() + 1);
    }

    #[test]
    fn test_local_incremental_decode_slides_live_kv_window() {
        let config = create_test_config();
        let weights = create_test_weights(&config, config.hidden_dim);
        let mut incremental = LocalForwardPass::new(weights);
        incremental.kv_cache.config.max_seq_len = 2;

        let prompt = vec![1, 2, 3];
        let first_logits = incremental.prefill(&prompt).unwrap();
        let first_token = sample_greedy(&first_logits);
        let _decode_logits = incremental.decode_step(first_token).unwrap();

        assert_eq!(incremental.position, 4);
        assert_eq!(incremental.kv_cache.seq_len(), 2);
        assert_eq!(incremental.kv_cache.base_position(), 2);
        assert_eq!(incremental.kv_cache.next_position(), 4);
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
    fn test_forward_pass_kv_snapshot_round_trip() {
        let config = create_test_config();
        let weights = create_test_weights(&config, config.hidden_dim);
        let mut forward = ForwardPass::new(
            weights,
            0,
            0,
            config.hidden_dim,
            1,
            std::time::Duration::from_secs(30),
        )
        .unwrap();

        let kv_cols = config.num_kv_heads * (config.hidden_dim / config.num_heads);
        let mut host_cache = KVCache::new(KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 16,
        });
        for layer_idx in 0..config.num_layers {
            host_cache
                .append_layer(
                    layer_idx,
                    Tensor2D::filled(2, kv_cols, 0.2 + layer_idx as f32),
                    Tensor2D::filled(2, kv_cols, 0.4 + layer_idx as f32),
                )
                .unwrap();
        }

        let snapshot = KVCacheSnapshot::from_cache(&host_cache, 2).unwrap();
        forward.import_kv_cache_snapshot(&snapshot).unwrap();

        assert_eq!(forward.position, 2);
        assert_eq!(forward.cache_seq_len(), 2);
        assert!(forward.live_kv_cache_bytes() > 0);
        assert_eq!(forward.logical_kv_tokens(), 2);

        let exported = forward
            .export_kv_cache_snapshot(None)
            .unwrap()
            .expect("snapshot should be present");
        assert_eq!(exported.sequence, snapshot.sequence);

        let restored = exported.decode_cache().unwrap();
        assert_eq!(restored.seq_len(), host_cache.seq_len());
        for layer_idx in 0..config.num_layers {
            let (expected_k, expected_v) = host_cache.get_layer_kv(layer_idx).unwrap();
            let (actual_k, actual_v) = restored.get_layer_kv(layer_idx).unwrap();
            assert_eq!(actual_k.data, expected_k.data);
            assert_eq!(actual_v.data, expected_v.data);
        }
    }

    #[test]
    fn test_forward_pass_exports_segmented_kv_snapshot_window() {
        let config = create_test_config();
        let weights = create_test_weights(&config, config.hidden_dim);
        let mut forward = ForwardPass::new(
            weights,
            0,
            0,
            config.hidden_dim,
            1,
            std::time::Duration::from_secs(30),
        )
        .unwrap();

        let kv_cols = config.num_kv_heads * (config.hidden_dim / config.num_heads);
        let mut host_cache = KVCache::new(KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 16,
        });
        for layer_idx in 0..config.num_layers {
            host_cache
                .append_layer(
                    layer_idx,
                    Tensor2D::filled(4, kv_cols, 0.2 + layer_idx as f32),
                    Tensor2D::filled(4, kv_cols, 0.4 + layer_idx as f32),
                )
                .unwrap();
        }

        let snapshot = KVCacheSnapshot::from_cache(&host_cache, 4).unwrap();
        forward.import_kv_cache_snapshot(&snapshot).unwrap();

        let exported = forward
            .export_kv_cache_snapshot(Some(2))
            .unwrap()
            .expect("segmented snapshot should be present");
        assert_eq!(exported.sequence.next_position, 4);
        assert_eq!(exported.sequence.cached_tokens, 2);
        assert_eq!(exported.sequence.first_cached_position(), 2);

        let restored = exported.decode_cache().unwrap();
        assert_eq!(restored.base_position(), 2);
        assert_eq!(restored.seq_len(), 2);
    }

    #[test]
    fn test_forward_pass_imports_paged_live_kv_layout() {
        let config = create_test_config();
        let weights = create_test_weights(&config, config.hidden_dim);
        let mut forward = ForwardPass::new(
            weights,
            0,
            0,
            config.hidden_dim,
            1,
            std::time::Duration::from_secs(30),
        )
        .unwrap();

        let kv_cols = config.num_kv_heads * (config.hidden_dim / config.num_heads);
        let mut host_cache = KVCache::new(KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 64,
        });
        for layer_idx in 0..config.num_layers {
            host_cache
                .append_layer(
                    layer_idx,
                    Tensor2D::filled(17, kv_cols, 0.25 + layer_idx as f32),
                    Tensor2D::filled(17, kv_cols, 0.5 + layer_idx as f32),
                )
                .unwrap();
        }

        let snapshot = KVCacheSnapshot::from_cache(&host_cache, 17).unwrap();
        forward.import_kv_cache_snapshot(&snapshot).unwrap();

        assert_eq!(forward.device_kv_cache.live_page_tokens(), 16);
        assert_eq!(forward.device_kv_cache.live_block_table_len(0).unwrap(), 2);

        let exported = forward
            .export_kv_cache_snapshot(None)
            .unwrap()
            .expect("paged snapshot should be present");
        let metadata = exported.live_metadata(
            &forward.device_kv_cache.config,
            crate::inference::kv_cache::LiveKVResidency::ImportedFromSnapshot,
            Some("session-17".to_string()),
            Some("worker-0".to_string()),
        );
        assert_eq!(metadata.block_table_len, 2);
        assert_eq!(metadata.tail_tokens, 1);
    }

    #[tokio::test]
    async fn test_decode_microbatch_respects_sliding_device_kv_window() {
        let config = create_test_config();
        let residency = Arc::new(
            SharedModelResidency::from_host(create_test_weights(&config, config.hidden_dim))
                .unwrap(),
        );
        let mut backend_a = CandleExecutionBackend::new(
            Arc::clone(&residency),
            0,
            0,
            config.hidden_dim,
            1,
            std::time::Duration::from_secs(30),
        )
        .unwrap();
        let mut backend_b = CandleExecutionBackend::new(
            Arc::clone(&residency),
            0,
            0,
            config.hidden_dim,
            1,
            std::time::Duration::from_secs(30),
        )
        .unwrap();
        backend_a.forward_pass.device_kv_cache.config.max_seq_len = 2;
        backend_b.forward_pass.device_kv_cache.config.max_seq_len = 2;

        let mut tensor_plane = TensorPlane::bind(TensorPlaneConfig::default())
            .await
            .unwrap();
        let peer_id = PeerId::random();
        let local_addr = tensor_plane.local_addr();
        let mut worker_ring = WorkerRing::new(
            0,
            1,
            peer_id,
            peer_id,
            local_addr,
            local_addr,
            InferenceRuntimeMode::ThroughputFirst,
            ExecutionProviderKind::Cpu,
            &mut tensor_plane,
        );

        let prompt = vec![1, 2, 3];
        backend_a
            .forward_pass
            .prefill(&prompt, &mut worker_ring, Uuid::new_v4())
            .await
            .unwrap();
        backend_b
            .forward_pass
            .prefill(&prompt, &mut worker_ring, Uuid::new_v4())
            .await
            .unwrap();

        assert_eq!(backend_a.forward_pass.position, 3);
        assert_eq!(backend_a.forward_pass.device_kv_cache.seq_len(), 2);
        assert_eq!(backend_a.forward_pass.device_kv_cache.next_position(), 3);

        let mut batch = vec![&mut backend_a, &mut backend_b];
        let logits = ForwardPass::decode_microbatch(
            &mut batch,
            &[7, 8],
            &[Uuid::new_v4(), Uuid::new_v4()],
            &mut worker_ring,
        )
        .await
        .unwrap();

        assert_eq!(logits.len(), 2);
        assert_eq!(backend_a.forward_pass.position, 4);
        assert_eq!(backend_b.forward_pass.position, 4);
        assert_eq!(backend_a.forward_pass.device_kv_cache.seq_len(), 2);
        assert_eq!(backend_b.forward_pass.device_kv_cache.seq_len(), 2);
        assert_eq!(backend_a.forward_pass.device_kv_cache.base_position(), 2);
        assert_eq!(backend_b.forward_pass.device_kv_cache.base_position(), 2);
        assert_eq!(backend_a.forward_pass.device_kv_cache.next_position(), 4);
        assert_eq!(backend_b.forward_pass.device_kv_cache.next_position(), 4);
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
