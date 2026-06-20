//! Tensor-Parallel Forward Pass Implementation
//!
//! This module implements the tensor-parallel forward pass used by the production inference
//! runtime. Each worker holds a column shard of the model weights, executes backend-native local
//! projection and KV work, and only synchronizes at the explicit residual boundaries that require
//! globally identical activations.
//!
//! ## Forward Pass Flow (per layer)
//!
//! ```text
//! Input: hidden_states [seq_len, hidden_dim]
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 1. Local Attention Preparation                      │
//! │    normed = rms_norm(hidden)                        │
//! │    Q_local, K_local, V_local = normed @ W_qkv       │
//! │    apply RoPE + append paged KV cache               │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 2. Local Attention Compute                          │
//! │    attn_output = attention(Q_local, cached KV)      │
//! │    O_partial = attn_output @ W_o[my_cols]           │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 3. Residual Synchronization                         │
//! │    O_full = ring_allreduce(O_partial)               │
//! │    post_attn = hidden + O_full                      │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! ┌─────────────────────────────────────────────────────┐
//! │ 4. Local MLP + Residual Synchronization             │
//! │    down_partial = swiglu(rms_norm(post_attn))       │
//! │                   @ W_down[my_cols]                 │
//! │    down_full = ring_allreduce(down_partial)         │
//! │    output = post_attn + down_full                   │
//! └─────────────────────────────────────────────────────┘
//!        │
//!        ▼
//! Output: next_hidden_states [seq_len, hidden_dim]
//! ```

use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::{RingAllReduceMetrics, WorkerRing};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

use super::engine::CollectiveResidency;
use super::kv_cache::{KVCache, KVCacheConfig, KVCacheSnapshot};
use super::runtime::{
    apply_rope_device as apply_rope_candle, device_tensor_from_1d as to_candle_1d,
    device_tensor_from_2d as to_candle_2d, host_tensor_2d_from_device as from_candle_2d,
    rms_norm_device as rms_norm_candle, runtime_error as device_error, sample_token_device,
    silu_device as silu_candle, softmax_device, DeviceCollectiveBuffer, DeviceDType as DType,
    DeviceTensor as CandleTensor, RuntimeDevice,
};
use super::stats::{
    record_runtime_device_kv_active_view_cache_hit,
    record_runtime_device_kv_active_view_cache_miss, record_runtime_device_kv_head_view_cache_hit,
    record_runtime_device_kv_head_view_cache_miss,
    record_runtime_device_kv_selected_head_view_cache_hit,
    record_runtime_device_kv_selected_head_view_cache_miss, record_runtime_device_sampling,
    record_runtime_host_sampling,
};
use super::tensor_ops::{
    apply_rope, embed_tokens, matmul, rms_norm, sample_greedy, sample_token, silu, Tensor1D,
    Tensor2D,
};
use crate::inference::backend::BackendLogits;
use crate::inference::fast_path::{DecodeWorkspaceLease, PrefillWorkspaceLease};

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
    fn new(page_rows: usize, cols: usize, device: &RuntimeDevice) -> Result<Self> {
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
struct SelectedDeviceKvHeads {
    keys_for_scores: CandleTensor,
    values: CandleTensor,
    seq_len: usize,
}

#[derive(Clone)]
struct CachedSelectedDeviceKvHeads {
    selection_signature: u64,
    local_kv_head_indices: CandleTensor,
    local_kv_heads: usize,
    head_dim: usize,
    keys_for_scores: CandleTensor,
    values: CandleTensor,
    seq_len: usize,
}

#[derive(Clone, Copy, Debug)]
struct DevicePageSpan {
    page_index: usize,
    start_row: usize,
    row_count: usize,
}

#[derive(Clone)]
struct DeviceLayerKVCache {
    pages: Vec<DeviceKVPage>,
    free_pages: Vec<usize>,
    block_table: Vec<DevicePageSpan>,
    seq_len: usize,
    cols: usize,
    page_rows: usize,
    cached_active: Option<(CandleTensor, CandleTensor, usize)>,
    cached_heads: Option<(CandleTensor, CandleTensor, usize, usize)>,
    cached_selected_heads: Option<CachedSelectedDeviceKvHeads>,
}

impl DeviceLayerKVCache {
    fn new(page_rows: usize) -> Self {
        Self {
            pages: Vec::new(),
            free_pages: Vec::new(),
            block_table: Vec::new(),
            seq_len: 0,
            cols: 0,
            page_rows,
            cached_active: None,
            cached_heads: None,
            cached_selected_heads: None,
        }
    }

    fn invalidate_active_cache(&mut self) {
        self.cached_active = None;
        self.cached_heads = None;
        self.cached_selected_heads = None;
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

    fn ensure_writable_tail(&mut self, cols: usize, device: &RuntimeDevice) -> Result<usize> {
        self.ensure_page_shape(cols)?;

        if self.block_table.is_empty() {
            let page_index = self.allocate_page(cols, device)?;
            self.block_table.push(DevicePageSpan {
                page_index,
                start_row: 0,
                row_count: 0,
            });
            return Ok(page_index);
        }

        let tail_span =
            self.block_table.last().copied().ok_or_else(|| {
                AgentError::Execution("Device KV block table is empty".to_string())
            })?;
        let tail_page = self.pages.get(tail_span.page_index).ok_or_else(|| {
            AgentError::Execution(format!(
                "Invalid device KV page index {}",
                tail_span.page_index
            ))
        })?;
        let tail_end = tail_span.start_row.saturating_add(tail_span.row_count);
        if tail_end != tail_page.used_rows {
            return Err(AgentError::Execution(format!(
                "Device KV tail span for page {} is inconsistent: span end {} vs used_rows {}",
                tail_span.page_index, tail_end, tail_page.used_rows
            )));
        }
        if tail_page.free_rows(self.page_rows) > 0 {
            return Ok(tail_span.page_index);
        }

        let page_index = self.allocate_page(cols, device)?;
        self.block_table.push(DevicePageSpan {
            page_index,
            start_row: 0,
            row_count: 0,
        });
        Ok(page_index)
    }

    fn allocate_page(&mut self, cols: usize, device: &RuntimeDevice) -> Result<usize> {
        self.ensure_page_shape(cols)?;
        if let Some(page_index) = self.free_pages.pop() {
            let page = self.pages.get_mut(page_index).ok_or_else(|| {
                AgentError::Execution(format!("Invalid free device KV page index {}", page_index))
            })?;
            page.used_rows = 0;
            return Ok(page_index);
        }

        let page_index = self.pages.len();
        self.pages
            .push(DeviceKVPage::new(self.page_rows, cols, device)?);
        Ok(page_index)
    }

    fn release_page(&mut self, page_index: usize) -> Result<()> {
        let page = self.pages.get_mut(page_index).ok_or_else(|| {
            AgentError::Execution(format!("Invalid device KV page index {}", page_index))
        })?;
        page.used_rows = 0;
        if !self.free_pages.contains(&page_index) {
            self.free_pages.push(page_index);
        }
        Ok(())
    }

    fn reclaim_unused_pages(&mut self) -> Result<()> {
        let referenced = self
            .block_table
            .iter()
            .map(|span| span.page_index)
            .collect::<std::collections::HashSet<_>>();
        for page_index in 0..self.pages.len() {
            if !referenced.contains(&page_index) && self.pages[page_index].used_rows != 0 {
                self.release_page(page_index)?;
            }
        }
        Ok(())
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
        let cache_updates = self.prepare_append_cache_updates(new_keys, new_values)?;
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
            let tail_span = self.block_table.last_mut().ok_or_else(|| {
                AgentError::Execution("Device KV block table unexpectedly empty".to_string())
            })?;
            if tail_span.page_index != page_index {
                return Err(AgentError::Execution(format!(
                    "Device KV tail span/page mismatch: span page {} vs active page {}",
                    tail_span.page_index, page_index
                )));
            }
            tail_span.row_count += copy_rows;
            appended_rows += copy_rows;
            self.seq_len += copy_rows;
        }
        self.apply_append_cache_updates(cache_updates);
        Ok(())
    }

    fn prepare_append_cache_updates(
        &self,
        new_keys: &CandleTensor,
        new_values: &CandleTensor,
    ) -> Result<AppendCacheUpdates> {
        let active = self.extend_cached_active(new_keys, new_values)?;
        let heads = self.extend_cached_heads(new_keys, new_values)?;
        let selected = self.extend_cached_selected_heads(new_keys, new_values)?;
        Ok(AppendCacheUpdates {
            active,
            heads,
            selected,
        })
    }

    fn apply_append_cache_updates(&mut self, updates: AppendCacheUpdates) {
        self.cached_active = updates.active;
        self.cached_heads = updates.heads;
        self.cached_selected_heads = updates.selected;
    }

    fn extend_cached_active(
        &self,
        new_keys: &CandleTensor,
        new_values: &CandleTensor,
    ) -> Result<Option<(CandleTensor, CandleTensor, usize)>> {
        let Some((cached_keys, cached_values, cached_seq_len)) = &self.cached_active else {
            return Ok(None);
        };
        let keys = concat_rows(cached_keys, new_keys)?;
        let values = concat_rows(cached_values, new_values)?;
        Ok(Some((keys, values, cached_seq_len + new_keys.dims()[0])))
    }

    fn extend_cached_heads(
        &self,
        new_keys: &CandleTensor,
        new_values: &CandleTensor,
    ) -> Result<Option<(CandleTensor, CandleTensor, usize, usize)>> {
        let Some((cached_k_heads, cached_v_heads, cached_seq_len, local_kv_heads)) =
            &self.cached_heads
        else {
            return Ok(None);
        };
        let head_dim = self.resolve_head_dim(*local_kv_heads)?;
        let new_k_heads = reshape_device_kv_heads(new_keys, *local_kv_heads, head_dim)?;
        let new_v_heads = reshape_device_kv_heads(new_values, *local_kv_heads, head_dim)?;
        let k_heads = concat_head_sequence(cached_k_heads, &new_k_heads)?;
        let v_heads = concat_head_sequence(cached_v_heads, &new_v_heads)?;
        Ok(Some((
            k_heads,
            v_heads,
            cached_seq_len + new_keys.dims()[0],
            *local_kv_heads,
        )))
    }

    fn extend_cached_selected_heads(
        &self,
        new_keys: &CandleTensor,
        new_values: &CandleTensor,
    ) -> Result<Option<CachedSelectedDeviceKvHeads>> {
        let Some(cached) = &self.cached_selected_heads else {
            return Ok(None);
        };
        let new_k_heads =
            reshape_device_kv_heads(new_keys, cached.local_kv_heads, cached.head_dim)?;
        let new_v_heads =
            reshape_device_kv_heads(new_values, cached.local_kv_heads, cached.head_dim)?;
        let selected_k_t = new_k_heads
            .index_select(&cached.local_kv_head_indices, 0)
            .map_err(device_error)?
            .transpose(1, 2)
            .map_err(device_error)?
            .contiguous()
            .map_err(device_error)?;
        let selected_v = new_v_heads
            .index_select(&cached.local_kv_head_indices, 0)
            .map_err(device_error)?
            .contiguous()
            .map_err(device_error)?;
        Ok(Some(CachedSelectedDeviceKvHeads {
            selection_signature: cached.selection_signature,
            local_kv_head_indices: cached.local_kv_head_indices.clone(),
            local_kv_heads: cached.local_kv_heads,
            head_dim: cached.head_dim,
            keys_for_scores: concat_selected_keys_for_scores(
                &cached.keys_for_scores,
                &selected_k_t,
            )?,
            values: concat_selected_values(&cached.values, &selected_v)?,
            seq_len: cached.seq_len + new_keys.dims()[0],
        }))
    }

    fn resolve_head_dim(&self, local_kv_heads: usize) -> Result<usize> {
        if local_kv_heads == 0 {
            return Err(AgentError::Execution(
                "Device KV head cache cannot derive a head_dim from zero heads".to_string(),
            ));
        }
        if self.cols == 0 || self.cols % local_kv_heads != 0 {
            return Err(AgentError::Execution(format!(
                "Device KV width {} is not divisible by local_kv_heads {}",
                self.cols, local_kv_heads
            )));
        }
        Ok(self.cols / local_kv_heads)
    }

    fn build_active_tensors(&self) -> Result<(CandleTensor, CandleTensor, usize)> {
        if self.seq_len == 0 || self.block_table.is_empty() {
            return Err(AgentError::Execution(
                "Requested device KV state before cache initialization".to_string(),
            ));
        }

        let mut key_chunks = Vec::with_capacity(self.block_table.len());
        let mut value_chunks = Vec::with_capacity(self.block_table.len());
        for span in &self.block_table {
            let page = self.pages.get(span.page_index).ok_or_else(|| {
                AgentError::Execution(format!("Invalid device KV page index {}", span.page_index))
            })?;
            if span.row_count == 0 {
                continue;
            }
            key_chunks.push(
                page.keys
                    .narrow(0, span.start_row, span.row_count)
                    .map_err(device_error)?,
            );
            value_chunks.push(
                page.values
                    .narrow(0, span.start_row, span.row_count)
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

    fn active_tensors(&mut self) -> Result<(CandleTensor, CandleTensor, usize)> {
        if let Some((keys, values, seq_len)) = &self.cached_active {
            record_runtime_device_kv_active_view_cache_hit();
            return Ok((keys.clone(), values.clone(), *seq_len));
        }
        record_runtime_device_kv_active_view_cache_miss();
        let active = self.build_active_tensors()?;
        self.cached_active = Some((active.0.clone(), active.1.clone(), active.2));
        Ok(active)
    }

    fn active_heads(
        &mut self,
        head_dim: usize,
    ) -> Result<(CandleTensor, CandleTensor, usize, usize)> {
        if let Some((keys, values, seq_len, local_kv_heads)) = &self.cached_heads {
            record_runtime_device_kv_head_view_cache_hit();
            return Ok((keys.clone(), values.clone(), *seq_len, *local_kv_heads));
        }
        record_runtime_device_kv_head_view_cache_miss();

        let (cached_k, cached_v, cached_seq_len) = self.active_tensors()?;
        let k_dims = cached_k.dims();
        if k_dims.len() != 2 {
            return Err(AgentError::Execution(format!(
                "Device KV active tensor expected rank-2 keys, got {:?}",
                k_dims
            )));
        }
        let kv_cols = k_dims[1];
        if kv_cols % head_dim != 0 {
            return Err(AgentError::Execution(format!(
                "Device KV active width {} is not a multiple of head_dim {}",
                kv_cols, head_dim
            )));
        }
        let local_kv_heads = kv_cols / head_dim;
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
        self.cached_heads = Some((
            k_heads.clone(),
            v_heads.clone(),
            cached_seq_len,
            local_kv_heads,
        ));
        Ok((k_heads, v_heads, cached_seq_len, local_kv_heads))
    }

    fn selected_heads(
        &mut self,
        head_dim: usize,
        selection_signature: u64,
        local_kv_head_indices: &CandleTensor,
    ) -> Result<SelectedDeviceKvHeads> {
        if let Some(cached) = &self.cached_selected_heads {
            if cached.selection_signature == selection_signature {
                record_runtime_device_kv_selected_head_view_cache_hit();
                return Ok(SelectedDeviceKvHeads {
                    keys_for_scores: cached.keys_for_scores.clone(),
                    values: cached.values.clone(),
                    seq_len: cached.seq_len,
                });
            }
        }
        record_runtime_device_kv_selected_head_view_cache_miss();

        let (k_heads, v_heads, cached_seq_len, _local_kv_heads) = self.active_heads(head_dim)?;
        let selected_k_t = k_heads
            .index_select(local_kv_head_indices, 0)
            .map_err(device_error)?
            .transpose(1, 2)
            .map_err(device_error)?
            .contiguous()
            .map_err(device_error)?;
        let selected_v = v_heads
            .index_select(local_kv_head_indices, 0)
            .map_err(device_error)?
            .contiguous()
            .map_err(device_error)?;
        self.cached_selected_heads = Some(CachedSelectedDeviceKvHeads {
            selection_signature,
            local_kv_head_indices: local_kv_head_indices.clone(),
            local_kv_heads: self.cols / head_dim,
            head_dim,
            keys_for_scores: selected_k_t.clone(),
            values: selected_v.clone(),
            seq_len: cached_seq_len,
        });
        Ok(SelectedDeviceKvHeads {
            keys_for_scores: selected_k_t,
            values: selected_v,
            seq_len: cached_seq_len,
        })
    }

    #[cfg(test)]
    fn block_table_len(&self) -> usize {
        self.block_table.len()
    }

    #[cfg(test)]
    fn allocated_page_count(&self) -> usize {
        self.pages.len()
    }

    fn clear(&mut self) {
        self.invalidate_active_cache();
        for page in &mut self.pages {
            page.used_rows = 0;
        }
        self.free_pages.clear();
        self.free_pages.extend(0..self.pages.len());
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
        self.invalidate_active_cache();
        let mut drop_rows = self.seq_len - keep_rows;
        while drop_rows > 0 {
            let front_span = self.block_table.first_mut().ok_or_else(|| {
                AgentError::Execution("Device KV block table unexpectedly empty".to_string())
            })?;
            if drop_rows >= front_span.row_count {
                drop_rows -= front_span.row_count;
                let page_index = front_span.page_index;
                self.block_table.remove(0);
                self.release_page(page_index)?;
                continue;
            }

            front_span.start_row += drop_rows;
            front_span.row_count -= drop_rows;
            drop_rows = 0;
        }
        self.seq_len = keep_rows;
        self.reclaim_unused_pages()?;
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

struct AppendCacheUpdates {
    active: Option<(CandleTensor, CandleTensor, usize)>,
    heads: Option<(CandleTensor, CandleTensor, usize, usize)>,
    selected: Option<CachedSelectedDeviceKvHeads>,
}

#[derive(Clone)]
struct DeviceKVCache {
    layers: Vec<DeviceLayerKVCache>,
    base_position: usize,
    config: KVCacheConfig,
    causal_mask_cache: HashMap<(usize, usize, usize), CandleTensor>,
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
            causal_mask_cache: HashMap::new(),
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

    #[cfg(test)]
    fn get_layer_active_kv(
        &mut self,
        layer_idx: usize,
    ) -> Result<(CandleTensor, CandleTensor, usize)> {
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.active_tensors()
    }

    fn get_layer_active_heads(
        &mut self,
        layer_idx: usize,
        head_dim: usize,
    ) -> Result<(CandleTensor, CandleTensor, usize, usize)> {
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.active_heads(head_dim)
    }

    fn get_layer_selected_heads(
        &mut self,
        layer_idx: usize,
        head_dim: usize,
        selection_signature: u64,
        local_kv_head_indices: &CandleTensor,
    ) -> Result<SelectedDeviceKvHeads> {
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.selected_heads(head_dim, selection_signature, local_kv_head_indices)
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
        self.causal_mask_cache.clear();
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

    fn causal_attention_mask(
        &mut self,
        q_rows: usize,
        cached_seq_len: usize,
        cache_prefix_len: usize,
        device: &RuntimeDevice,
    ) -> Result<CandleTensor> {
        let key = (q_rows, cached_seq_len, cache_prefix_len);
        if let Some(mask) = self.causal_mask_cache.get(&key) {
            return Ok(mask.clone());
        }

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
        let tensor = CandleTensor::from_vec(mask, (1, q_rows, cached_seq_len), device)
            .map_err(device_error)?;
        self.causal_mask_cache.insert(key, tensor.clone());
        Ok(tensor)
    }

    #[cfg(test)]
    fn live_page_tokens(&self) -> usize {
        self.config.live_page_tokens()
    }

    #[cfg(test)]
    fn live_block_table_len(&self, layer_idx: usize) -> Result<usize> {
        self.layers
            .get(layer_idx)
            .map(DeviceLayerKVCache::block_table_len)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))
    }

    #[cfg(test)]
    fn allocated_page_count(&self, layer_idx: usize) -> Result<usize> {
        self.layers
            .get(layer_idx)
            .map(DeviceLayerKVCache::allocated_page_count)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))
    }

    #[cfg(test)]
    fn causal_mask_cache_len(&self) -> usize {
        self.causal_mask_cache.len()
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
            let (keys, values, _) = self
                .layers
                .get(layer_idx)
                .ok_or_else(|| {
                    AgentError::Execution(format!("Invalid layer index: {}", layer_idx))
                })?
                .build_active_tensors()?;
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

#[derive(Debug, Clone, Copy)]
struct AttentionShardLayout {
    head_dim: usize,
    q_heads_per_kv_head: usize,
    q_cols: usize,
    q_head_start: usize,
    local_q_heads: usize,
    kv_cols: usize,
    kv_head_start: usize,
    local_kv_heads: usize,
}

fn resolve_attention_shard_layout(
    config: &ModelConfig,
    shard_start: usize,
    shard_end: usize,
) -> Result<AttentionShardLayout> {
    if config.num_heads == 0 || config.hidden_dim % config.num_heads != 0 {
        return Err(AgentError::Execution(format!(
            "Unsupported attention geometry: hidden_dim {} num_heads {}",
            config.hidden_dim, config.num_heads
        )));
    }
    if config.num_kv_heads == 0 || config.num_heads % config.num_kv_heads != 0 {
        return Err(AgentError::Execution(format!(
            "Unsupported grouped-query attention geometry: num_heads {} num_kv_heads {}",
            config.num_heads, config.num_kv_heads
        )));
    }
    if shard_start >= shard_end || shard_end > config.hidden_dim {
        return Err(AgentError::Execution(format!(
            "Invalid shard range {}..{} for hidden_dim {}",
            shard_start, shard_end, config.hidden_dim
        )));
    }

    let head_dim = config.hidden_dim / config.num_heads;
    let q_heads_per_kv_head = config.num_heads / config.num_kv_heads;
    let q_group_width = q_heads_per_kv_head * head_dim;
    if config.hidden_dim % q_group_width != 0 {
        return Err(AgentError::Execution(format!(
            "Hidden dim {} is not divisible by grouped-query shard width {}",
            config.hidden_dim, q_group_width
        )));
    }
    if shard_start % q_group_width != 0 || shard_end % q_group_width != 0 {
        return Err(AgentError::Execution(format!(
            "Shard range {}..{} is not aligned to grouped-query shard width {}",
            shard_start, shard_end, q_group_width
        )));
    }

    let q_cols = shard_end - shard_start;
    let local_group_count = q_cols / q_group_width;
    let local_q_heads = local_group_count * q_heads_per_kv_head;
    let q_head_start = shard_start / head_dim;
    let kv_head_start = shard_start / q_group_width;
    let local_kv_heads = local_group_count;
    let kv_cols = local_kv_heads * head_dim;

    Ok(AttentionShardLayout {
        head_dim,
        q_heads_per_kv_head,
        q_cols,
        q_head_start,
        local_q_heads,
        kv_cols,
        kv_head_start,
        local_kv_heads,
    })
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

fn validate_single_position(position: u32, expected_start: usize) -> Result<()> {
    if position as usize != expected_start {
        return Err(crate::errors::AgentError::Execution(format!(
            "Non-contiguous attention positions: expected {}, got {}",
            expected_start, position
        )));
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
    shard_start: usize,
    shard_end: usize,
    layer_idx: usize,
    absolute_positions: &[u32],
    q_local: Tensor2D,
    k_local: Tensor2D,
    v_local: Tensor2D,
) -> Result<Tensor2D> {
    let attention_layout = resolve_attention_shard_layout(config, shard_start, shard_end)?;
    let head_dim = attention_layout.head_dim;
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
    if q_local.cols != attention_layout.q_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width mismatch: expected {}, got {}",
            attention_layout.q_cols, q_local.cols
        )));
    }
    if k_local.cols != attention_layout.kv_cols || v_local.cols != attention_layout.kv_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection width mismatch: expected {}, got k={} v={}",
            attention_layout.kv_cols, k_local.cols, v_local.cols
        )));
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
        let global_q_head = attention_layout.q_head_start + local_q_idx;
        let global_kv_head = global_q_head / attention_layout.q_heads_per_kv_head;
        if global_kv_head < attention_layout.kv_head_start
            || global_kv_head >= attention_layout.kv_head_start + local_kv_heads
        {
            return Err(crate::errors::AgentError::Execution(format!(
                "Local KV head ownership mismatch: q_head {} maps to kv_head {}, local kv range {}..{}",
                global_q_head,
                global_kv_head,
                attention_layout.kv_head_start,
                attention_layout.kv_head_start + local_kv_heads
            )));
        }
        let local_kv_idx = global_kv_head - attention_layout.kv_head_start;
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
    q_width: usize,
    k_width: usize,
    v_width: usize,
    w_qkv: CandleTensor,
    w_o: CandleTensor,
    gate_width: usize,
    up_width: usize,
    w_gate_up: CandleTensor,
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

struct AttentionPhaseOutput {
    o_partial: CandleTensor,
}

struct MlpPhaseOutput {
    post_attn: CandleTensor,
    down_partial: CandleTensor,
}

struct StagedDeviceCollective {
    template: CandleTensor,
    buffer: DeviceCollectiveBuffer,
}

fn device_tensor_memory_usage_bytes(tensor: &CandleTensor) -> usize {
    tensor
        .dims()
        .iter()
        .copied()
        .product::<usize>()
        .saturating_mul(std::mem::size_of::<f32>())
}

impl DeviceLayerWeights {
    fn memory_usage_bytes(&self) -> usize {
        device_tensor_memory_usage_bytes(&self.w_qkv)
            .saturating_add(device_tensor_memory_usage_bytes(&self.w_o))
            .saturating_add(device_tensor_memory_usage_bytes(&self.w_gate_up))
            .saturating_add(device_tensor_memory_usage_bytes(&self.w_down))
            .saturating_add(device_tensor_memory_usage_bytes(&self.attn_norm))
            .saturating_add(device_tensor_memory_usage_bytes(&self.mlp_norm))
    }
}

impl DeviceModelWeights {
    fn from_host(weights: &ModelWeights) -> Result<Self> {
        let embedding = to_candle_2d(&weights.embedding)?;
        let final_norm = to_candle_1d(&weights.final_norm)?;
        let lm_head = to_candle_2d(&weights.lm_head)?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let w_q = to_candle_2d(&layer.w_q)?;
            let w_k = to_candle_2d(&layer.w_k)?;
            let w_v = to_candle_2d(&layer.w_v)?;
            let w_up = to_candle_2d(&layer.w_up)?;
            let w_gate = to_candle_2d(&layer.w_gate)?;
            layers.push(DeviceLayerWeights {
                q_width: layer.w_q.cols,
                k_width: layer.w_k.cols,
                v_width: layer.w_v.cols,
                w_qkv: concat_projection_tensors(&[&w_q, &w_k, &w_v])?,
                gate_width: layer.w_gate.cols,
                up_width: layer.w_up.cols,
                w_gate_up: concat_projection_tensors(&[&w_gate, &w_up])?,
                w_o: to_candle_2d(&layer.w_o)?,
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
        device_tensor_memory_usage_bytes(&self.embedding)
            .saturating_add(
                self.layers
                    .iter()
                    .map(DeviceLayerWeights::memory_usage_bytes)
                    .sum(),
            )
            .saturating_add(device_tensor_memory_usage_bytes(&self.final_norm))
            .saturating_add(device_tensor_memory_usage_bytes(&self.lm_head))
    }
}

fn concat_projection_tensors(tensors: &[&CandleTensor]) -> Result<CandleTensor> {
    let first = tensors.first().ok_or_else(|| {
        AgentError::Execution("cannot concatenate an empty projection set".to_string())
    })?;
    let dims = first.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "projection concatenation expects rank-2 tensors, got {:?}",
            dims
        )));
    }
    let rows = dims[0];
    for tensor in tensors {
        let tensor_dims = tensor.dims();
        if tensor_dims.len() != 2 || tensor_dims[0] != rows {
            return Err(AgentError::Execution(format!(
                "projection concatenation shape mismatch: expected [{} x _], got {:?}",
                rows, tensor_dims
            )));
        }
    }

    CandleTensor::cat(tensors, 1).map_err(|e| {
        AgentError::Execution(format!(
            "GPU tensor backend error while concatenating projections: {}",
            e
        ))
    })
}

fn narrow_projection(tensor: &CandleTensor, start: usize, width: usize) -> Result<CandleTensor> {
    tensor
        .narrow(1, start, width)
        .map_err(|e| AgentError::Execution(format!("GPU tensor backend error: {}", e)))
}

fn split_qkv_projection(
    tensor: &CandleTensor,
    q_width: usize,
    k_width: usize,
    v_width: usize,
) -> Result<(CandleTensor, CandleTensor, CandleTensor)> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "projection split expects rank-2 tensor, got {:?}",
            dims
        )));
    }
    let total_width = q_width.saturating_add(k_width).saturating_add(v_width);
    if dims[1] != total_width {
        return Err(AgentError::Execution(format!(
            "projection split width mismatch: tensor cols {} vs expected {}",
            dims[1], total_width
        )));
    }

    Ok((
        narrow_projection(tensor, 0, q_width)?,
        narrow_projection(tensor, q_width, k_width)?,
        narrow_projection(tensor, q_width.saturating_add(k_width), v_width)?,
    ))
}

fn split_gate_up_projection(
    tensor: &CandleTensor,
    gate_width: usize,
    up_width: usize,
) -> Result<(CandleTensor, CandleTensor)> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "projection split expects rank-2 tensor, got {:?}",
            dims
        )));
    }
    let total_width = gate_width.saturating_add(up_width);
    if dims[1] != total_width {
        return Err(AgentError::Execution(format!(
            "projection split width mismatch: tensor cols {} vs expected {}",
            dims[1], total_width
        )));
    }

    Ok((
        narrow_projection(tensor, 0, gate_width)?,
        narrow_projection(tensor, gate_width, up_width)?,
    ))
}

fn qkv_partials_from_combined(
    normed: &CandleTensor,
    layer: &DeviceLayerWeights,
) -> Result<(CandleTensor, CandleTensor, CandleTensor)> {
    let combined = normed
        .matmul(&layer.w_qkv)
        .map_err(|e| AgentError::Execution(format!("GPU tensor backend error: {}", e)))?;
    split_qkv_projection(&combined, layer.q_width, layer.k_width, layer.v_width)
}

fn gate_up_partials_from_combined(
    mlp_normed: &CandleTensor,
    layer: &DeviceLayerWeights,
) -> Result<(CandleTensor, CandleTensor)> {
    let combined = mlp_normed
        .matmul(&layer.w_gate_up)
        .map_err(|e| AgentError::Execution(format!("GPU tensor backend error: {}", e)))?;
    split_gate_up_projection(&combined, layer.gate_width, layer.up_width)
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
    attention_layout: AttentionShardLayout,
    local_kv_head_indices: CandleTensor,
    local_kv_head_indices_signature: u64,

    /// KV cache for attention
    device_kv_cache: DeviceKVCache,
    collective_residency: CollectiveResidency,

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
        collective_residency: CollectiveResidency,
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
        let attention_layout = resolve_attention_shard_layout(&config, shard_start, shard_end)?;
        let local_kv_head_indices_host = build_local_kv_head_indices(attention_layout)?;
        let local_kv_head_indices_signature =
            kv_head_selection_signature(&local_kv_head_indices_host);
        let local_kv_head_indices = CandleTensor::from_vec(
            local_kv_head_indices_host,
            attention_layout.local_q_heads,
            residency.device_weights.embedding.device(),
        )
        .map_err(device_error)?;

        Ok(Self {
            device_weights: Arc::clone(&residency.device_weights),
            config,
            allreduce_timeout,
            attention_layout,
            local_kv_head_indices,
            local_kv_head_indices_signature,
            device_kv_cache: DeviceKVCache::new(kv_config),
            collective_residency,
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
            CollectiveResidency::StagedRuntime,
            worker_position,
            shard_start,
            shard_end,
            total_workers,
            allreduce_timeout,
        )
    }

    /// Embed input tokens
    fn embed(&self, tokens: &[u32]) -> Result<CandleTensor> {
        let ids =
            CandleTensor::from_slice(tokens, tokens.len(), self.device_weights.embedding.device())
                .map_err(|e| {
                    crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
                })?;
        self.device_weights.embedding.embedding(&ids).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })
    }

    fn prepare_attention_phase(
        &mut self,
        hidden: &CandleTensor,
        layer_idx: usize,
        absolute_positions: &[u32],
    ) -> Result<AttentionPhaseOutput> {
        let layer = &self.device_weights.layers[layer_idx];
        let normed = rms_norm_candle(hidden, &layer.attn_norm, self.config.rms_norm_eps)?;
        let (q_partial, k_partial, v_partial) = qkv_partials_from_combined(&normed, layer)?;
        let hidden_dims = hidden.dims();

        debug!(
            "Layer {} QKV partial computed: {}x{} -> {}x{}",
            layer_idx,
            hidden_dims[0],
            hidden_dims[1],
            q_partial.dims()[0],
            q_partial.dims()[1]
        );

        info!(
            layer_idx,
            "Layer attention partials prepared; entering KV/attention path"
        );

        let attn_output = attention_output_device(
            &mut self.device_kv_cache,
            &self.config,
            self.attention_layout,
            layer_idx,
            absolute_positions,
            self.local_kv_head_indices_signature,
            &self.local_kv_head_indices,
            q_partial,
            k_partial,
            v_partial,
        )?;

        info!(
            layer_idx,
            "Layer attention output materialized; projecting local O partial"
        );

        let o_partial = attn_output.matmul(&layer.w_o).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        Ok(AttentionPhaseOutput { o_partial })
    }

    fn prepare_mlp_phase(
        &self,
        hidden: &CandleTensor,
        o_full: &CandleTensor,
        layer_idx: usize,
    ) -> Result<MlpPhaseOutput> {
        let layer = &self.device_weights.layers[layer_idx];
        let post_attn = hidden.broadcast_add(o_full).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let mlp_normed = rms_norm_candle(&post_attn, &layer.mlp_norm, self.config.rms_norm_eps)?;
        let (gate_partial, up_partial) = gate_up_partials_from_combined(&mlp_normed, layer)?;
        let gate_activated = silu_candle(&gate_partial)?;
        let mlp_hidden = gate_activated.broadcast_mul(&up_partial).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let down_partial = mlp_hidden.matmul(&layer.w_down).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;

        Ok(MlpPhaseOutput {
            post_attn,
            down_partial,
        })
    }

    fn finalize_layer_output(
        &self,
        post_attn: &CandleTensor,
        down_full: &CandleTensor,
    ) -> Result<CandleTensor> {
        post_attn.broadcast_add(down_full).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })
    }

    /// Execute a single layer of the canonical forward pass implementation.
    async fn forward_layer(
        &mut self,
        hidden: &CandleTensor,
        layer_idx: usize,
        absolute_positions: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<CandleTensor> {
        let start = Instant::now();
        let hidden_dims = hidden.dims();
        let hidden_rows = hidden_dims[0];
        let hidden_cols = hidden_dims[1];

        info!(
            layer_idx,
            seq_len = hidden_rows,
            hidden_dim = hidden_cols,
            position_start = absolute_positions.first().copied().unwrap_or_default(),
            position_end = absolute_positions.last().copied().unwrap_or_default(),
            "Starting forward layer"
        );

        let AttentionPhaseOutput { o_partial } =
            self.prepare_attention_phase(hidden, layer_idx, absolute_positions)?;
        info!(
            layer_idx,
            "Layer O partial ready; staging first ring all-reduce"
        );
        let staged_o = Self::stage_device_collective(
            self.collective_residency,
            &o_partial,
            layer_idx as u32,
            0,
        )?;
        let o_full = self
            .complete_staged_device_collective(staged_o, worker_ring, job_id, layer_idx as u32, 0)
            .await?;
        info!(
            layer_idx,
            "Layer first ring all-reduce complete; entering MLP preparation"
        );

        let MlpPhaseOutput {
            post_attn,
            down_partial,
        } = self.prepare_mlp_phase(hidden, &o_full, layer_idx)?;
        info!(
            layer_idx,
            "Layer MLP down partial ready; staging second ring all-reduce"
        );
        let staged_down = Self::stage_device_collective(
            self.collective_residency,
            &down_partial,
            layer_idx as u32,
            1,
        )?;
        let down_full = self
            .complete_staged_device_collective(
                staged_down,
                worker_ring,
                job_id,
                layer_idx as u32,
                1,
            )
            .await?;
        info!(
            layer_idx,
            "Layer second ring all-reduce complete; finalizing layer output"
        );

        let output = self.finalize_layer_output(&post_attn, &down_full)?;

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
            BackendLogits::Host(logits) => {
                let started = Instant::now();
                let token = if temperature <= 0.0 || temperature == 1.0 && top_p >= 1.0 {
                    sample_greedy(logits)
                } else {
                    sample_token(logits, temperature, top_p, seed)
                };
                record_runtime_host_sampling(1, started.elapsed().as_millis() as u64);
                Ok(token)
            }
            BackendLogits::Device(logits) => {
                let started = Instant::now();
                if temperature <= 0.0 || temperature == 1.0 && top_p >= 1.0 {
                    let token_ids = logits
                        .argmax(1)
                        .and_then(|idx| idx.to_vec1::<u32>())
                        .map_err(device_error)?;
                    let token = token_ids.into_iter().next().ok_or_else(|| {
                        AgentError::Execution("Device argmax produced no token ids".to_string())
                    })?;
                    record_runtime_device_sampling(1, started.elapsed().as_millis() as u64);
                    Ok(token)
                } else {
                    sample_token_device(logits, temperature, top_p, seed).map_err(|device_err| {
                        AgentError::Execution(format!(
                            "Device stochastic sampling failed: {}",
                            device_err
                        ))
                    })
                }
            }
        }
    }

    /// Canonical single-session prefill path.
    pub async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        workspace: Option<&mut PrefillWorkspaceLease>,
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
            .forward_tokens(&tokens[start..], start, worker_ring, job_id, workspace)
            .await?;
        self.position = tokens.len();
        debug_assert_eq!(self.device_kv_cache.next_position(), self.position);
        self.compute_logits(&hidden)
    }

    /// Canonical single-session decode step.
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
            .forward_tokens(&[token], self.position, worker_ring, job_id, None)
            .await?;
        self.position += 1;
        self.compute_logits(&hidden)
    }

    /// Provider-accelerated decode microbatch path used by the serving fast
    /// path once bucket and metadata invariants have been validated.
    pub(crate) async fn fast_path_decode_microbatch(
        backends: &mut [&mut crate::inference::backend::ProviderRuntimeCore],
        tokens: &[u32],
        job_ids: &[Uuid],
        worker_ring: &mut WorkerRing<'_>,
        mut workspace: Option<&mut DecodeWorkspaceLease>,
    ) -> Result<CandleTensor> {
        if backends.is_empty() || tokens.is_empty() {
            return Err(crate::errors::AgentError::Execution(
                "Decode microbatch requires at least one backend and token".to_string(),
            ));
        }
        if backends.len() != tokens.len() || backends.len() != job_ids.len() {
            return Err(crate::errors::AgentError::Execution(format!(
                "Decode microbatch shape mismatch: backends={} tokens={} job_ids={}",
                backends.len(),
                tokens.len(),
                job_ids.len()
            )));
        }

        let (
            config,
            device_weights,
            local_kv_head_indices,
            allreduce_timeout,
            collective_residency,
        ) = {
            let template = &backends[0].forward_pass;
            (
                template.config.clone(),
                template.device_weights.clone(),
                template.local_kv_head_indices.clone(),
                template.allreduce_timeout,
                template.collective_residency,
            )
        };
        let batch_job_id = collective_batch_job_id(job_ids);

        let owned_positions;
        let (positions, token_ids) = if let Some(workspace) = workspace.as_deref_mut() {
            workspace.stage_from_slot_reader(tokens, |slot_idx| {
                let forward_pass = &backends[slot_idx].forward_pass;
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
                Ok((
                    forward_pass.position as u32,
                    forward_pass.device_kv_cache.seq_len() as u32,
                ))
            })?;
            (
                workspace.positions(tokens.len()),
                workspace.token_ids(tokens.len()),
            )
        } else {
            owned_positions = {
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
                positions
            };
            (owned_positions.as_slice(), tokens)
        };

        let ids =
            CandleTensor::from_slice(token_ids, tokens.len(), device_weights.embedding.device())
                .map_err(|e| {
                    crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
                })?;
        let mut hidden = device_weights.embedding.embedding(&ids).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        let mut metrics = RingAllReduceMetrics::default();
        let attention_layout = backends
            .first()
            .map(|backend| backend.forward_pass.attention_layout)
            .ok_or_else(|| {
                AgentError::Execution(
                    "Device batch attention requires at least one backend".to_string(),
                )
            })?;

        for layer_idx in 0..config.num_layers {
            let layer = &device_weights.layers[layer_idx];
            let normed = rms_norm_candle(&hidden, &layer.attn_norm, config.rms_norm_eps)?;
            let (q_partial, k_partial, v_partial) = qkv_partials_from_combined(&normed, &layer)?;

            let attn_output = attention_output_device_batch_from_backends(
                backends,
                &config,
                attention_layout,
                layer_idx,
                positions,
                &local_kv_head_indices,
                q_partial,
                k_partial,
                v_partial,
            )?;

            let o_partial = attn_output.matmul(&layer.w_o).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let o_full = Self::ring_allreduce_device_batch(
                &o_partial,
                worker_ring,
                batch_job_id,
                layer_idx as u32,
                0,
                collective_residency,
                allreduce_timeout,
                &mut metrics,
            )
            .await?;
            let post_attn = hidden.broadcast_add(&o_full).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            drop(o_full);

            let mlp_normed = rms_norm_candle(&post_attn, &layer.mlp_norm, config.rms_norm_eps)?;
            let (gate_partial, up_partial) = gate_up_partials_from_combined(&mlp_normed, &layer)?;
            let gate_activated = silu_candle(&gate_partial)?;
            let mlp_hidden = gate_activated.broadcast_mul(&up_partial).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let down_partial = mlp_hidden.matmul(&layer.w_down).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            let down_full = Self::ring_allreduce_device_batch(
                &down_partial,
                worker_ring,
                batch_job_id,
                layer_idx as u32,
                1,
                collective_residency,
                allreduce_timeout,
                &mut metrics,
            )
            .await?;
            hidden = post_attn.broadcast_add(&down_full).map_err(|e| {
                crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
            })?;
            drop(down_full);
        }

        hidden = rms_norm_candle(&hidden, &device_weights.final_norm, config.rms_norm_eps)?;
        let logits_2d = hidden.matmul(&device_weights.lm_head).map_err(|e| {
            crate::errors::AgentError::Execution(format!("GPU tensor backend error: {}", e))
        })?;
        for backend in backends.iter_mut() {
            backend.forward_pass.position += 1;
            backend.forward_pass.last_allreduce_metrics = metrics;
        }
        Ok(logits_2d)
    }

    /// Stage a device tensor for collective submission without forcing the surrounding
    /// decode phase into a broader blocking region than required.
    fn stage_device_collective(
        collective_residency: CollectiveResidency,
        tensor: &CandleTensor,
        layer_idx: u32,
        collective_seq: u32,
    ) -> Result<StagedDeviceCollective> {
        if !matches!(collective_residency, CollectiveResidency::StagedRuntime) {
            return Err(AgentError::Execution(format!(
                "Execution contract {:?} does not permit staged runtime collectives",
                collective_residency
            )));
        }
        info!(
            layer_idx,
            collective_seq,
            rows = tensor.dims()[0],
            cols = tensor.dims()[1],
            device = ?tensor.device().location(),
            "Preparing collective submission"
        );
        let buffer = DeviceCollectiveBuffer::from_device_tensor(tensor)?;
        info!(
            layer_idx,
            collective_seq,
            elements = buffer.len(),
            "Collective submission is staying device-resident"
        );
        Ok(StagedDeviceCollective {
            template: tensor.clone(),
            buffer,
        })
    }

    /// Complete a previously staged collective submission and restore the reduced
    /// result directly onto the execution device.
    async fn complete_staged_device_collective(
        &mut self,
        staged: StagedDeviceCollective,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        layer_idx: u32,
        collective_seq: u32,
    ) -> Result<CandleTensor> {
        if worker_ring.total_workers <= 1 {
            return Ok(staged.template);
        }
        let StagedDeviceCollective {
            template,
            mut buffer,
        } = staged;
        worker_ring
            .ring_all_reduce_staged_with_timeout(
                &mut buffer,
                job_id,
                layer_idx,
                collective_seq,
                self.allreduce_timeout,
            )
            .await?;
        let mut metrics = worker_ring.last_run_metrics();
        metrics.device_resident_collective_count += 1;
        self.last_allreduce_metrics.accumulate(metrics);
        buffer.into_device_tensor_like(&template)
    }

    async fn ring_allreduce_device_batch(
        tensor: &CandleTensor,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        layer_idx: u32,
        collective_seq: u32,
        collective_residency: CollectiveResidency,
        allreduce_timeout: std::time::Duration,
        metrics: &mut RingAllReduceMetrics,
    ) -> Result<CandleTensor> {
        if worker_ring.total_workers <= 1 {
            return Ok(tensor.clone());
        }
        let staged =
            Self::stage_device_collective(collective_residency, tensor, layer_idx, collective_seq)?;
        let StagedDeviceCollective {
            template,
            mut buffer,
        } = staged;
        worker_ring
            .ring_all_reduce_staged_with_timeout(
                &mut buffer,
                job_id,
                layer_idx,
                collective_seq,
                allreduce_timeout,
            )
            .await?;
        let mut run_metrics = worker_ring.last_run_metrics();
        run_metrics.device_resident_collective_count += 1;
        metrics.accumulate(run_metrics);
        buffer.into_device_tensor_like(&template)
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
        mut workspace: Option<&mut PrefillWorkspaceLease>,
    ) -> Result<CandleTensor> {
        if tokens.is_empty() {
            return Err(crate::errors::AgentError::Execution(
                "Forward pass requires at least one token".to_string(),
            ));
        }
        let start = Instant::now();
        let owned_positions;
        let absolute_positions = if let Some(workspace) = workspace.as_deref_mut() {
            workspace.stage_positions(absolute_position_start, tokens.len())?
        } else {
            owned_positions = build_positions(absolute_position_start, tokens.len());
            owned_positions.as_slice()
        };
        self.last_allreduce_metrics = RingAllReduceMetrics::default();

        info!(
            token_count = tokens.len(),
            position_start = absolute_position_start,
            position_end = absolute_position_start + tokens.len() - 1,
            "Starting forward token segment"
        );

        let mut hidden = self.embed(tokens)?;
        let hidden_dims = hidden.dims();
        debug!(
            "Embedded {} tokens -> {:?} at positions {}..{}",
            tokens.len(),
            (hidden_dims[0], hidden_dims[1]),
            absolute_position_start,
            absolute_position_start + tokens.len() - 1
        );
        info!(
            rows = hidden_dims[0],
            cols = hidden_dims[1],
            "Token embedding materialized; entering transformer layers"
        );

        for layer_idx in 0..self.config.num_layers {
            if layer_idx == 0 {
                info!("Entering first transformer layer of forward token segment");
            }
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
    attention_layout: AttentionShardLayout,
    layer_idx: usize,
    absolute_positions: &[u32],
    local_kv_head_indices_signature: u64,
    local_kv_head_indices: &CandleTensor,
    q_local: CandleTensor,
    k_local: CandleTensor,
    v_local: CandleTensor,
) -> Result<CandleTensor> {
    let head_dim = attention_layout.head_dim;
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
    if q_cols != attention_layout.q_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width mismatch: expected {}, got {}",
            attention_layout.q_cols, q_cols
        )));
    }
    if k_cols != attention_layout.kv_cols || v_cols != attention_layout.kv_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection width mismatch: expected {}, got k={} v={}",
            attention_layout.kv_cols, k_cols, v_cols
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
        attention_layout,
        layer_idx,
        cache_prefix_len,
        local_kv_head_indices_signature,
        local_kv_head_indices,
        &q_rope,
    )
}

fn attention_output_device_batch_from_backends(
    backends: &mut [&mut crate::inference::backend::ProviderRuntimeCore],
    config: &ModelConfig,
    attention_layout: AttentionShardLayout,
    layer_idx: usize,
    absolute_positions: &[u32],
    local_kv_head_indices: &CandleTensor,
    q_local: CandleTensor,
    k_local: CandleTensor,
    v_local: CandleTensor,
) -> Result<CandleTensor> {
    let batch_size = backends.len();
    if absolute_positions.len() != batch_size {
        return Err(AgentError::Execution(format!(
            "Device batch attention position count {} does not match backend batch size {}",
            absolute_positions.len(),
            batch_size
        )));
    }
    if batch_size == 0 {
        return Err(AgentError::Execution(
            "Device batch attention requires at least one backend".to_string(),
        ));
    }
    let head_dim = attention_layout.head_dim;
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
    if q_cols != attention_layout.q_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local query projection width mismatch: expected {}, got {}",
            attention_layout.q_cols, q_cols
        )));
    }
    if k_cols != attention_layout.kv_cols || v_cols != attention_layout.kv_cols {
        return Err(crate::errors::AgentError::Execution(format!(
            "Local KV projection width mismatch: expected {}, got k={} v={}",
            attention_layout.kv_cols, k_cols, v_cols
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
    let local_q_heads = q_cols / head_dim;
    if local_kv_head_indices.dims() != [local_q_heads] {
        return Err(AgentError::Execution(format!(
            "Local KV head index tensor mismatch: expected [{}], got {:?}",
            local_q_heads,
            local_kv_head_indices.dims()
        )));
    }
    let q_heads_batch = q_rope
        .reshape((batch_size, local_q_heads, head_dim))
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)?;

    let mut outputs = CandleTensor::zeros((batch_size, q_cols), DType::F32, q_local.device())
        .map_err(device_error)?;
    for (row_idx, backend) in backends.iter_mut().enumerate() {
        let kv_cache = &mut backend.forward_pass.device_kv_cache;
        let current_layer_seq_len = kv_cache.layer_seq_len(layer_idx)?;
        let max_seq_len = kv_cache.config.max_seq_len.max(1);
        if current_layer_seq_len.saturating_add(1) > max_seq_len {
            let drop_rows = current_layer_seq_len.saturating_add(1) - max_seq_len;
            kv_cache.retain_suffix(current_layer_seq_len.saturating_sub(drop_rows))?;
        }
        let prefix_len = kv_cache.layer_seq_len(layer_idx)?;
        validate_single_position(
            absolute_positions[row_idx],
            kv_cache.base_position().saturating_add(prefix_len),
        )?;
        let q_heads_row = q_heads_batch
            .narrow(1, row_idx, 1)
            .map_err(device_error)?
            .contiguous()
            .map_err(device_error)?;
        let k_row = k_rope.narrow(0, row_idx, 1).map_err(device_error)?;
        let v_row = v_local.narrow(0, row_idx, 1).map_err(device_error)?;
        kv_cache.append_layer(layer_idx, &k_row, &v_row)?;
        let output_row = attention_output_device_cached_single_query_from_heads(
            kv_cache,
            config,
            layer_idx,
            prefix_len,
            backend.forward_pass.local_kv_head_indices_signature,
            local_kv_head_indices,
            &q_heads_row,
            q_cols,
        )?;
        outputs = outputs
            .slice_assign(&[row_idx..row_idx + 1, 0..q_cols], &output_row)
            .map_err(device_error)?;
    }

    Ok(outputs)
}

fn attention_output_device_cached(
    kv_cache: &mut DeviceKVCache,
    config: &ModelConfig,
    _attention_layout: AttentionShardLayout,
    layer_idx: usize,
    cache_prefix_len: usize,
    local_kv_head_indices_signature: u64,
    local_kv_head_indices: &CandleTensor,
    q_rope: &CandleTensor,
) -> Result<CandleTensor> {
    attention_output_device_cached_with_precomputed_query(
        kv_cache,
        config,
        layer_idx,
        cache_prefix_len,
        local_kv_head_indices_signature,
        local_kv_head_indices,
        q_rope,
        None,
    )
}

fn attention_output_device_cached_with_precomputed_query(
    kv_cache: &mut DeviceKVCache,
    config: &ModelConfig,
    layer_idx: usize,
    cache_prefix_len: usize,
    local_kv_head_indices_signature: u64,
    local_kv_head_indices: &CandleTensor,
    q_rope: &CandleTensor,
    precomputed_q_heads: Option<&CandleTensor>,
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
    if local_kv_head_indices.dims() != [local_q_heads] {
        return Err(AgentError::Execution(format!(
            "Local KV head index tensor mismatch: expected [{}], got {:?}",
            local_q_heads,
            local_kv_head_indices.dims()
        )));
    }

    let selected_heads = kv_cache.get_layer_selected_heads(
        layer_idx,
        head_dim,
        local_kv_head_indices_signature,
        local_kv_head_indices,
    )?;

    let q_heads = if let Some(q_heads) = precomputed_q_heads {
        q_heads.clone()
    } else {
        q_rope
            .reshape((q_rows, local_q_heads, head_dim))
            .map_err(device_error)?
            .transpose(0, 1)
            .map_err(device_error)?
            .contiguous()
            .map_err(device_error)?
    };
    let scores = q_heads
        .matmul(&selected_heads.keys_for_scores)
        .map_err(device_error)?
        .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
        .map_err(device_error)?;
    let full_prefix_visible = q_rows == 1 && cache_prefix_len + 1 == selected_heads.seq_len;
    let probs = if full_prefix_visible {
        softmax_device(&scores, 2)?
    } else {
        let mask = kv_cache.causal_attention_mask(
            q_rows,
            selected_heads.seq_len,
            cache_prefix_len,
            q_rope.device(),
        )?;
        let masked = scores.broadcast_add(&mask).map_err(device_error)?;
        softmax_device(&masked, 2)?
    };
    probs
        .matmul(&selected_heads.values)
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .reshape((q_rows, local_q_heads * head_dim))
        .map_err(device_error)
}

fn attention_output_device_cached_single_query_from_heads(
    kv_cache: &mut DeviceKVCache,
    config: &ModelConfig,
    layer_idx: usize,
    cache_prefix_len: usize,
    local_kv_head_indices_signature: u64,
    local_kv_head_indices: &CandleTensor,
    q_heads: &CandleTensor,
    q_cols: usize,
) -> Result<CandleTensor> {
    let head_dim = config.hidden_dim / config.num_heads;
    let q_head_dims = q_heads.dims();
    if q_head_dims.len() != 3 || q_head_dims[1] != 1 || q_head_dims[2] != head_dim {
        return Err(AgentError::Execution(format!(
            "Single-query attention expects [local_q_heads, 1, head_dim], got {:?}",
            q_head_dims
        )));
    }
    let local_q_heads = q_head_dims[0];
    if local_q_heads * head_dim != q_cols {
        return Err(AgentError::Execution(format!(
            "Single-query attention width mismatch: local_q_heads {} * head_dim {} != q_cols {}",
            local_q_heads, head_dim, q_cols
        )));
    }
    if local_kv_head_indices.dims() != [local_q_heads] {
        return Err(AgentError::Execution(format!(
            "Local KV head index tensor mismatch: expected [{}], got {:?}",
            local_q_heads,
            local_kv_head_indices.dims()
        )));
    }

    let selected_heads = kv_cache.get_layer_selected_heads(
        layer_idx,
        head_dim,
        local_kv_head_indices_signature,
        local_kv_head_indices,
    )?;
    if cache_prefix_len + 1 != selected_heads.seq_len {
        return Err(AgentError::Execution(format!(
            "Single-query attention expected full prefix visibility: prefix {} cached_seq_len {}",
            cache_prefix_len, selected_heads.seq_len
        )));
    }
    let scores = q_heads
        .matmul(&selected_heads.keys_for_scores)
        .map_err(device_error)?
        .affine(1.0 / (head_dim as f64).sqrt(), 0.0)
        .map_err(device_error)?;
    softmax_device(&scores, 2)?
        .matmul(&selected_heads.values)
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .reshape((1, q_cols))
        .map_err(device_error)
}

fn concat_rows(existing: &CandleTensor, appended: &CandleTensor) -> Result<CandleTensor> {
    let tensors = [existing, appended];
    CandleTensor::cat(&tensors, 0).map_err(device_error)
}

fn concat_head_sequence(existing: &CandleTensor, appended: &CandleTensor) -> Result<CandleTensor> {
    let tensors = [existing, appended];
    CandleTensor::cat(&tensors, 1).map_err(device_error)
}

fn concat_selected_keys_for_scores(
    existing: &CandleTensor,
    appended: &CandleTensor,
) -> Result<CandleTensor> {
    let tensors = [existing, appended];
    CandleTensor::cat(&tensors, 2).map_err(device_error)
}

fn concat_selected_values(
    existing: &CandleTensor,
    appended: &CandleTensor,
) -> Result<CandleTensor> {
    let tensors = [existing, appended];
    CandleTensor::cat(&tensors, 1).map_err(device_error)
}

fn reshape_device_kv_heads(
    tensor: &CandleTensor,
    local_kv_heads: usize,
    head_dim: usize,
) -> Result<CandleTensor> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Device KV head reshape expects rank-2 tensors, got {:?}",
            dims
        )));
    }
    if dims[1] != local_kv_heads.saturating_mul(head_dim) {
        return Err(AgentError::Execution(format!(
            "Device KV head reshape width mismatch: expected {}, got {}",
            local_kv_heads.saturating_mul(head_dim),
            dims[1]
        )));
    }
    tensor
        .reshape((dims[0], local_kv_heads, head_dim))
        .map_err(device_error)?
        .transpose(0, 1)
        .map_err(device_error)?
        .contiguous()
        .map_err(device_error)
}

fn build_local_kv_head_indices(attention_layout: AttentionShardLayout) -> Result<Vec<u32>> {
    let mut indices = Vec::with_capacity(attention_layout.local_q_heads);
    for local_q_idx in 0..attention_layout.local_q_heads {
        let global_q_head = attention_layout.q_head_start + local_q_idx;
        let global_kv_head = global_q_head / attention_layout.q_heads_per_kv_head;
        if global_kv_head < attention_layout.kv_head_start
            || global_kv_head >= attention_layout.kv_head_start + attention_layout.local_kv_heads
        {
            return Err(AgentError::Execution(format!(
                "Local KV head ownership mismatch: q_head {} maps to kv_head {}, local kv range {}..{}",
                global_q_head,
                global_kv_head,
                attention_layout.kv_head_start,
                attention_layout.kv_head_start + attention_layout.local_kv_heads
            )));
        }
        indices.push((global_kv_head - attention_layout.kv_head_start) as u32);
    }
    Ok(indices)
}

fn kv_head_selection_signature(indices: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    indices.hash(&mut hasher);
    hasher.finish()
}

fn collective_batch_job_id(job_ids: &[Uuid]) -> Uuid {
    let mut batch_seed = [0u8; 16];
    for job_id in job_ids {
        for (slot, byte) in batch_seed.iter_mut().zip(job_id.as_bytes().iter()) {
            *slot ^= *byte;
        }
    }
    Uuid::from_bytes(batch_seed)
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
                config.hidden_dim,
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
    use crate::inference::backend::ProviderRuntimeCore;
    use crate::inference::engine::InferenceRuntimeMode;
    use crate::inference::stats::InferenceStats;
    use crate::network::{TensorPlane, TensorPlaneConfig};
    use crate::provider::ExecutionProviderKind;
    use libp2p::PeerId;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
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
    fn test_attention_shard_layout_supports_gqa_aligned_uneven_shards() {
        let config = ModelConfig {
            hidden_dim: 576,
            num_heads: 9,
            num_kv_heads: 3,
            num_layers: 2,
            vocab_size: 100,
            intermediate_size: 1536,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };

        let first = resolve_attention_shard_layout(&config, 0, 384).unwrap();
        assert_eq!(first.q_cols, 384);
        assert_eq!(first.local_q_heads, 6);
        assert_eq!(first.kv_cols, 128);
        assert_eq!(first.local_kv_heads, 2);
        assert_eq!(
            build_local_kv_head_indices(first).unwrap(),
            vec![0, 0, 0, 1, 1, 1]
        );

        let second = resolve_attention_shard_layout(&config, 384, 576).unwrap();
        assert_eq!(second.q_cols, 192);
        assert_eq!(second.local_q_heads, 3);
        assert_eq!(second.kv_cols, 64);
        assert_eq!(second.local_kv_heads, 1);
        assert_eq!(build_local_kv_head_indices(second).unwrap(), vec![0, 0, 0]);
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

    #[test]
    fn test_device_kv_active_view_cache_invalidates_on_append() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            max_seq_len: 8,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let first_keys = to_candle_2d(&Tensor2D::filled(2, 4, 1.0)).unwrap();
        let first_values = to_candle_2d(&Tensor2D::filled(2, 4, 2.0)).unwrap();
        kv_cache
            .append_layer(0, &first_keys, &first_values)
            .unwrap();

        let (cached_keys, cached_values, cached_rows) = kv_cache.get_layer_active_kv(0).unwrap();
        assert_eq!(cached_rows, 2);
        assert_eq!(cached_keys.dims(), &[2, 4]);
        assert_eq!(cached_values.dims(), &[2, 4]);

        let second_keys = to_candle_2d(&Tensor2D::filled(1, 4, 3.0)).unwrap();
        let second_values = to_candle_2d(&Tensor2D::filled(1, 4, 4.0)).unwrap();
        kv_cache
            .append_layer(0, &second_keys, &second_values)
            .unwrap();

        let (updated_keys, updated_values, updated_rows) = kv_cache.get_layer_active_kv(0).unwrap();
        assert_eq!(updated_rows, 3);
        assert_eq!(updated_keys.dims(), &[3, 4]);
        assert_eq!(updated_values.dims(), &[3, 4]);
        let host_keys = from_candle_2d(&updated_keys).unwrap();
        let host_values = from_candle_2d(&updated_values).unwrap();
        assert_eq!(&host_keys.data[8..12], &[3.0; 4]);
        assert_eq!(&host_values.data[8..12], &[4.0; 4]);
    }

    #[test]
    fn test_device_kv_causal_mask_cache_reuses_by_shape() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 4,
            max_seq_len: 8,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let device = RuntimeDevice::Cpu;

        let first = kv_cache.causal_attention_mask(3, 5, 2, &device).unwrap();
        let second = kv_cache.causal_attention_mask(3, 5, 2, &device).unwrap();
        let third = kv_cache.causal_attention_mask(2, 5, 1, &device).unwrap();

        assert_eq!(first.dims(), &[1, 3, 5]);
        assert_eq!(second.dims(), &[1, 3, 5]);
        assert_eq!(third.dims(), &[1, 2, 5]);
        assert_eq!(kv_cache.causal_mask_cache_len(), 2);
    }

    #[test]
    fn test_device_kv_active_head_cache_invalidates_on_append() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 2,
            max_seq_len: 8,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let first_keys = to_candle_2d(&Tensor2D::filled(2, 4, 1.0)).unwrap();
        let first_values = to_candle_2d(&Tensor2D::filled(2, 4, 2.0)).unwrap();
        kv_cache
            .append_layer(0, &first_keys, &first_values)
            .unwrap();

        let (cached_k_heads, cached_v_heads, cached_rows, cached_heads) =
            kv_cache.get_layer_active_heads(0, 2).unwrap();
        assert_eq!(cached_rows, 2);
        assert_eq!(cached_heads, 2);
        assert_eq!(cached_k_heads.dims(), &[2, 2, 2]);
        assert_eq!(cached_v_heads.dims(), &[2, 2, 2]);

        let second_keys = to_candle_2d(&Tensor2D::filled(1, 4, 3.0)).unwrap();
        let second_values = to_candle_2d(&Tensor2D::filled(1, 4, 4.0)).unwrap();
        kv_cache
            .append_layer(0, &second_keys, &second_values)
            .unwrap();

        let (updated_k_heads, updated_v_heads, updated_rows, updated_heads) =
            kv_cache.get_layer_active_heads(0, 2).unwrap();
        assert_eq!(updated_rows, 3);
        assert_eq!(updated_heads, 2);
        assert_eq!(updated_k_heads.dims(), &[2, 3, 2]);
        assert_eq!(updated_v_heads.dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_device_kv_selected_head_cache_invalidates_on_append() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 2,
            max_seq_len: 8,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let selection = CandleTensor::new(&[1u32, 0u32], &RuntimeDevice::Cpu).unwrap();
        let first_keys = to_candle_2d(&Tensor2D::filled(2, 4, 1.0)).unwrap();
        let first_values = to_candle_2d(&Tensor2D::filled(2, 4, 2.0)).unwrap();
        kv_cache
            .append_layer(0, &first_keys, &first_values)
            .unwrap();

        let selected_heads = kv_cache
            .get_layer_selected_heads(0, 2, 7, &selection)
            .unwrap();
        assert_eq!(selected_heads.seq_len, 2);
        assert_eq!(selected_heads.keys_for_scores.dims(), &[2, 2, 2]);
        assert_eq!(selected_heads.values.dims(), &[2, 2, 2]);

        let second_keys = to_candle_2d(&Tensor2D::filled(1, 4, 3.0)).unwrap();
        let second_values = to_candle_2d(&Tensor2D::filled(1, 4, 4.0)).unwrap();
        kv_cache
            .append_layer(0, &second_keys, &second_values)
            .unwrap();

        let updated_heads = kv_cache
            .get_layer_selected_heads(0, 2, 7, &selection)
            .unwrap();
        assert_eq!(updated_heads.seq_len, 3);
        assert_eq!(updated_heads.keys_for_scores.dims(), &[2, 2, 3]);
        assert_eq!(updated_heads.values.dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_device_kv_cache_metrics_track_runtime_view_reuse() {
        let stats = Arc::new(InferenceStats::new());
        InferenceStats::install_as_runtime_collector(&stats);

        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 2,
            max_seq_len: 8,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let selection = CandleTensor::new(&[1u32, 0u32], &RuntimeDevice::Cpu).unwrap();
        let first_keys = to_candle_2d(&Tensor2D::filled(2, 4, 1.0)).unwrap();
        let first_values = to_candle_2d(&Tensor2D::filled(2, 4, 2.0)).unwrap();
        kv_cache
            .append_layer(0, &first_keys, &first_values)
            .unwrap();

        let _ = kv_cache.get_layer_active_heads(0, 2).unwrap();
        let _ = kv_cache.get_layer_active_heads(0, 2).unwrap();
        let _ = kv_cache
            .get_layer_selected_heads(0, 2, 7, &selection)
            .unwrap();
        let _ = kv_cache
            .get_layer_selected_heads(0, 2, 7, &selection)
            .unwrap();

        assert_eq!(
            stats
                .device_kv_active_view_cache_misses
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .device_kv_active_view_cache_hits
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .device_kv_head_view_cache_misses
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats.device_kv_head_view_cache_hits.load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .device_kv_selected_head_view_cache_misses
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .device_kv_selected_head_view_cache_hits
                .load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_device_kv_retain_suffix_keeps_partial_page_without_rebuild() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            max_seq_len: 32,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let first_keys =
            to_candle_2d(&Tensor2D::new((0..36).map(|v| v as f32).collect(), 18, 2).unwrap())
                .unwrap();
        let first_values =
            to_candle_2d(&Tensor2D::new((100..136).map(|v| v as f32).collect(), 18, 2).unwrap())
                .unwrap();
        kv_cache
            .append_layer(0, &first_keys, &first_values)
            .unwrap();

        kv_cache.retain_suffix(17).unwrap();

        let second_keys = to_candle_2d(&Tensor2D::new(vec![36.0, 37.0], 1, 2).unwrap()).unwrap();
        let second_values =
            to_candle_2d(&Tensor2D::new(vec![136.0, 137.0], 1, 2).unwrap()).unwrap();
        kv_cache
            .append_layer(0, &second_keys, &second_values)
            .unwrap();

        let (active_keys, active_values, active_rows) = kv_cache.get_layer_active_kv(0).unwrap();
        assert_eq!(active_rows, 18);
        let host_keys = from_candle_2d(&active_keys).unwrap();
        let host_values = from_candle_2d(&active_values).unwrap();
        let expected_keys = (2..38).map(|v| v as f32).collect::<Vec<_>>();
        let expected_values = (102..138).map(|v| v as f32).collect::<Vec<_>>();
        assert_eq!(host_keys.data, expected_keys);
        assert_eq!(host_values.data, expected_values);
    }

    #[test]
    fn test_device_kv_reuses_released_pages_after_suffix_trim() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 2,
            max_seq_len: 32,
        };
        let mut kv_cache = DeviceKVCache::new(config);
        let first_keys = to_candle_2d(&Tensor2D::filled(32, 2, 1.0)).unwrap();
        let first_values = to_candle_2d(&Tensor2D::filled(32, 2, 2.0)).unwrap();
        kv_cache
            .append_layer(0, &first_keys, &first_values)
            .unwrap();
        assert_eq!(kv_cache.allocated_page_count(0).unwrap(), 2);

        kv_cache.retain_suffix(16).unwrap();

        let second_keys = to_candle_2d(&Tensor2D::filled(16, 2, 3.0)).unwrap();
        let second_values = to_candle_2d(&Tensor2D::filled(16, 2, 4.0)).unwrap();
        kv_cache
            .append_layer(0, &second_keys, &second_values)
            .unwrap();

        assert_eq!(kv_cache.allocated_page_count(0).unwrap(), 2);
        assert_eq!(kv_cache.live_block_table_len(0).unwrap(), 2);
    }

    #[tokio::test]
    async fn test_decode_microbatch_respects_sliding_device_kv_window() {
        let config = create_test_config();
        let residency = Arc::new(
            SharedModelResidency::from_host(create_test_weights(&config, config.hidden_dim))
                .unwrap(),
        );
        let mut backend_a = ProviderRuntimeCore::new(
            Arc::clone(&residency),
            0,
            0,
            config.hidden_dim,
            1,
            std::time::Duration::from_secs(30),
        )
        .unwrap();
        let mut backend_b = ProviderRuntimeCore::new(
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
            crate::inference::LocalExecutorContract::for_provider(ExecutionProviderKind::Cpu),
            None,
            &mut tensor_plane,
        );

        let prompt = vec![1, 2, 3];
        backend_a
            .forward_pass
            .prefill(&prompt, &mut worker_ring, Uuid::new_v4(), None)
            .await
            .unwrap();
        backend_b
            .forward_pass
            .prefill(&prompt, &mut worker_ring, Uuid::new_v4(), None)
            .await
            .unwrap();

        assert_eq!(backend_a.forward_pass.position, 3);
        assert_eq!(backend_a.forward_pass.device_kv_cache.seq_len(), 2);
        assert_eq!(backend_a.forward_pass.device_kv_cache.next_position(), 3);

        let mut batch = vec![&mut backend_a, &mut backend_b];
        let logits = ForwardPass::fast_path_decode_microbatch(
            &mut batch,
            &[7, 8],
            &[Uuid::new_v4(), Uuid::new_v4()],
            &mut worker_ring,
            None,
        )
        .await
        .unwrap();

        assert_eq!(logits.dims(), &[2, config.vocab_size]);
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
