//! KV Cache Management for Transformer Inference
//!
//! This module manages the key-value cache used in transformer attention layers.
//! The KV cache stores computed key and value tensors from previous positions,
//! allowing efficient autoregressive generation without recomputing past states.

use super::tensor_ops::Tensor2D;
use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};

/// How a KV cache snapshot is encoded for transfer or persistence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KVCacheEncoding {
    /// Full cache snapshot serialized as CBOR.
    FullSnapshotCbor,
}

/// Logical sequence state associated with a KV cache materialization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct KVSequenceState {
    /// Absolute sequence position of the next token to process.
    pub next_position: u32,
    /// Number of tokens currently represented in the cache materialization.
    pub cached_tokens: u32,
}

impl KVSequenceState {
    pub fn new(next_position: u32, cached_tokens: u32) -> Result<Self> {
        let state = Self {
            next_position,
            cached_tokens,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn from_cache(cache: &KVCache, next_position: u32) -> Result<Self> {
        if cache.next_position() as u32 != next_position {
            return Err(AgentError::Execution(format!(
                "KV sequence state next_position {} does not match cache next position {}",
                next_position,
                cache.next_position()
            )));
        }
        Self::new(next_position, cache.seq_len() as u32)
    }

    pub fn validate(&self) -> Result<()> {
        if self.cached_tokens > self.next_position {
            return Err(AgentError::Execution(format!(
                "KV sequence state is invalid: cached_tokens {} exceeds next_position {}",
                self.cached_tokens, self.next_position
            )));
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.cached_tokens == 0
    }

    pub fn first_cached_position(&self) -> u32 {
        self.next_position.saturating_sub(self.cached_tokens)
    }
}

/// Serialized KV payload that can later be replaced by remote or partial access.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KVCacheBlob {
    /// Encoding of the payload bytes.
    pub encoding: KVCacheEncoding,
    /// Serialized payload bytes.
    pub bytes: Vec<u8>,
}

impl KVCacheBlob {
    pub fn from_cache(cache: &KVCache) -> Result<Self> {
        Ok(Self {
            encoding: KVCacheEncoding::FullSnapshotCbor,
            bytes: cache.to_bytes()?,
        })
    }

    pub fn decode(&self) -> Result<KVCache> {
        match self.encoding {
            KVCacheEncoding::FullSnapshotCbor => KVCache::from_bytes(&self.bytes),
        }
    }

    pub fn size_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
}

/// Materialized KV cache snapshot paired with its logical sequence state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KVCacheSnapshot {
    /// Sequence state represented by this cache view.
    pub sequence: KVSequenceState,
    /// Serialized cache payload.
    pub blob: KVCacheBlob,
}

impl KVCacheSnapshot {
    pub fn from_cache(cache: &KVCache, next_position: u32) -> Result<Self> {
        let sequence = KVSequenceState::from_cache(cache, next_position)?;
        let blob = KVCacheBlob::from_cache(cache)?;
        Ok(Self { sequence, blob })
    }

    pub fn decode_cache(&self) -> Result<KVCache> {
        let cache = self.blob.decode()?;
        self.validate_with_cache(&cache)?;
        Ok(cache)
    }

    pub fn validate(&self) -> Result<()> {
        self.sequence.validate()?;
        let cache = self.blob.decode()?;
        self.validate_with_cache(&cache)
    }

    pub fn validate_with_cache(&self, cache: &KVCache) -> Result<()> {
        self.sequence.validate()?;
        if cache.seq_len() as u32 != self.sequence.cached_tokens {
            return Err(AgentError::Execution(format!(
                "KV snapshot cached token mismatch: payload has {} tokens, sequence metadata says {}",
                cache.seq_len(),
                self.sequence.cached_tokens
            )));
        }
        if cache.base_position() as u32 != self.sequence.first_cached_position() {
            return Err(AgentError::Execution(format!(
                "KV snapshot base position mismatch: payload starts at {}, sequence metadata says {}",
                cache.base_position(),
                self.sequence.first_cached_position()
            )));
        }
        Ok(())
    }

    pub fn live_metadata(
        &self,
        config: &KVCacheConfig,
        residency: LiveKVResidency,
        owner_session_id: Option<String>,
        owner_worker_id: Option<String>,
    ) -> LiveKVSequenceMetadata {
        let page_tokens = config.live_page_tokens();
        let cached_tokens = self.sequence.cached_tokens as usize;
        LiveKVSequenceMetadata {
            next_position: self.sequence.next_position,
            first_cached_position: self.sequence.first_cached_position(),
            cached_tokens: self.sequence.cached_tokens,
            page_tokens,
            block_table_len: cached_tokens.div_ceil(page_tokens),
            tail_tokens: cached_tokens % page_tokens,
            residency,
            owner_session_id,
            owner_worker_id,
        }
    }

    pub fn transfer_hooks(
        &self,
        config: &KVCacheConfig,
        residency: LiveKVResidency,
        owner_session_id: Option<String>,
        owner_worker_id: Option<String>,
    ) -> KVTransferHooks {
        KVTransferHooks {
            live_layout: LiveKVLayout::from_config(config),
            sequence: self.live_metadata(config, residency, owner_session_id, owner_worker_id),
            snapshot_encoding: self.blob.encoding,
        }
    }
}

/// Configuration for KV cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheConfig {
    /// Number of layers in the model
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Maximum sequence length to cache
    pub max_seq_len: usize,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            num_layers: 80,    // LLaMA 70B has 80 layers
            num_heads: 64,     // 64 heads
            head_dim: 128,     // 8192 / 64 = 128
            max_seq_len: 4096, // 4K context
        }
    }
}

/// Canonical token capacity of one live KV page.
pub const DEFAULT_LIVE_KV_PAGE_TOKENS: usize = 16;

impl KVCacheConfig {
    /// Fixed live page size used by the paged execution cache.
    pub fn live_page_tokens(&self) -> usize {
        self.max_seq_len.clamp(1, DEFAULT_LIVE_KV_PAGE_TOKENS)
    }
}

/// Where the active live KV currently resides.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LiveKVResidency {
    LocalOnly,
    ImportedFromSnapshot,
    RemoteOwned,
}

/// Canonical live-KV layout contract for decode/prefill execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiveKVLayout {
    /// Fixed token capacity per page.
    pub page_tokens: usize,
    /// Number of transformer layers covered by the cache.
    pub num_layers: usize,
    /// Number of KV heads stored per layer.
    pub num_heads: usize,
    /// Width of one attention head.
    pub head_dim: usize,
    /// Flattened scalar width of one token row.
    pub row_width: usize,
    /// Number of scalars reserved for one key or value page.
    pub page_stride_scalars: usize,
}

impl LiveKVLayout {
    pub fn from_config(config: &KVCacheConfig) -> Self {
        let page_tokens = config.live_page_tokens();
        let row_width = config.num_heads.saturating_mul(config.head_dim);
        Self {
            page_tokens,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            row_width,
            page_stride_scalars: page_tokens.saturating_mul(row_width),
        }
    }
}

/// Minimal session metadata the executor needs to reason about live KV residency.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LiveKVSequenceMetadata {
    /// Absolute sequence position of the next token to execute.
    pub next_position: u32,
    /// First absolute position still represented by the live cache.
    pub first_cached_position: u32,
    /// Number of cached tokens currently resident.
    pub cached_tokens: u32,
    /// Fixed tokens per page in the live layout.
    pub page_tokens: usize,
    /// Number of pages in the block table for each layer.
    pub block_table_len: usize,
    /// Number of valid rows in the tail page.
    pub tail_tokens: usize,
    /// Where this live cache currently resides.
    pub residency: LiveKVResidency,
    /// Optional distributed session owner identifier.
    pub owner_session_id: Option<String>,
    /// Optional distributed worker owner identifier.
    pub owner_worker_id: Option<String>,
}

/// Export/import hooks for converting the live layout into transfer-friendly blobs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KVTransferHooks {
    /// The paged live layout that the exporter must read from.
    pub live_layout: LiveKVLayout,
    /// Sequence metadata that describes the live residency window.
    pub sequence: LiveKVSequenceMetadata,
    /// The current transfer encoding used by snapshots.
    pub snapshot_encoding: KVCacheEncoding,
}

/// Key-Value cache for a single attention layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerKVCache {
    /// Cached keys: [seq_len, num_heads * head_dim]
    pub keys: Option<Tensor2D>,
    /// Cached values: [seq_len, num_heads * head_dim]
    pub values: Option<Tensor2D>,
    /// Current sequence length in cache
    pub seq_len: usize,
    /// Layer index
    pub layer_idx: usize,
}

impl LayerKVCache {
    /// Create a new empty layer cache
    pub fn new(layer_idx: usize) -> Self {
        Self {
            keys: None,
            values: None,
            seq_len: 0,
            layer_idx,
        }
    }

    /// Update the cache with new keys and values
    ///
    /// # Arguments
    /// * `new_keys` - New key tensor [new_tokens, kv_dim]
    /// * `new_values` - New value tensor [new_tokens, kv_dim]
    pub fn append(&mut self, new_keys: Tensor2D, new_values: Tensor2D) -> Result<()> {
        if new_keys.rows != new_values.rows || new_keys.cols != new_values.cols {
            return Err(AgentError::Execution(format!(
                "KV cache update shape mismatch: keys {}x{} vs values {}x{}",
                new_keys.rows, new_keys.cols, new_values.rows, new_values.cols
            )));
        }
        self.validate_new_tensors(&new_keys, &new_values)?;

        match (&mut self.keys, &mut self.values) {
            (Some(existing_keys), Some(existing_values)) => {
                if existing_keys.cols != new_keys.cols || existing_values.cols != new_values.cols {
                    return Err(AgentError::Execution(format!(
                        "KV cache append width mismatch: existing {}x{}, new {}x{}",
                        existing_keys.rows, existing_keys.cols, new_keys.rows, new_keys.cols
                    )));
                }
                existing_keys.data.reserve(new_keys.data.len());
                existing_keys.data.extend_from_slice(&new_keys.data);
                existing_values.data.reserve(new_values.data.len());
                existing_values.data.extend_from_slice(&new_values.data);

                existing_keys.rows += new_keys.rows;
                existing_values.rows += new_values.rows;
                self.seq_len = existing_keys.rows;
            }
            _ => {
                // First update - just store
                self.seq_len = new_keys.rows;
                self.keys = Some(new_keys);
                self.values = Some(new_values);
            }
        }

        self.validate()?;
        Ok(())
    }

    fn validate_new_tensors(&self, keys: &Tensor2D, values: &Tensor2D) -> Result<()> {
        if keys.rows != values.rows || keys.cols != values.cols {
            return Err(AgentError::Execution(format!(
                "KV cache tensor mismatch: keys {}x{} vs values {}x{}",
                keys.rows, keys.cols, values.rows, values.cols
            )));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        match (&self.keys, &self.values) {
            (Some(keys), Some(values)) => {
                if keys.rows != values.rows || keys.cols != values.cols {
                    return Err(AgentError::Execution(format!(
                        "KV cache invariant violated for layer {}: keys {}x{} vs values {}x{}",
                        self.layer_idx, keys.rows, keys.cols, values.rows, values.cols
                    )));
                }
                if self.seq_len != keys.rows {
                    return Err(AgentError::Execution(format!(
                        "KV cache invariant violated for layer {}: seq_len {} vs tensor rows {}",
                        self.layer_idx, self.seq_len, keys.rows
                    )));
                }
            }
            (None, None) => {
                if self.seq_len != 0 {
                    return Err(AgentError::Execution(format!(
                        "KV cache invariant violated for layer {}: empty tensors with seq_len {}",
                        self.layer_idx, self.seq_len
                    )));
                }
            }
            _ => {
                return Err(AgentError::Execution(format!(
                    "KV cache invariant violated for layer {}: keys/values presence diverged",
                    self.layer_idx
                )));
            }
        }

        Ok(())
    }

    /// Get the current keys tensor
    pub fn get_keys(&self) -> Option<&Tensor2D> {
        self.keys.as_ref()
    }

    /// Get the current values tensor
    pub fn get_values(&self) -> Option<&Tensor2D> {
        self.values.as_ref()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys = None;
        self.values = None;
        self.seq_len = 0;
    }

    /// Truncate cache to a specific length
    pub fn truncate(&mut self, max_len: usize) {
        if self.seq_len <= max_len {
            return;
        }

        if let (Some(keys), Some(values)) = (&mut self.keys, &mut self.values) {
            let kv_dim = keys.cols;
            let new_data_len = max_len * kv_dim;

            keys.data.truncate(new_data_len);
            keys.rows = max_len;
            values.data.truncate(new_data_len);
            values.rows = max_len;
            self.seq_len = max_len;
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let key_size = self.keys.as_ref().map(|k| k.data.len() * 4).unwrap_or(0);
        let val_size = self.values.as_ref().map(|v| v.data.len() * 4).unwrap_or(0);
        key_size + val_size
    }
}

/// Full KV cache for all layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCache {
    /// Per-layer caches
    pub layers: Vec<LayerKVCache>,
    /// Absolute token position represented by row 0 of the cache.
    #[serde(default)]
    pub base_position: usize,
    /// Configuration
    pub config: KVCacheConfig,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(config: KVCacheConfig) -> Self {
        let layers = (0..config.num_layers).map(LayerKVCache::new).collect();

        Self {
            layers,
            base_position: 0,
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(KVCacheConfig::default())
    }

    /// Get cache for a specific layer
    pub fn get_layer(&self, layer_idx: usize) -> Option<&LayerKVCache> {
        self.layers.get(layer_idx)
    }

    /// Get mutable cache for a specific layer
    pub fn get_layer_mut(&mut self, layer_idx: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(layer_idx)
    }

    /// Update cache for a specific layer
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        keys: Tensor2D,
        values: Tensor2D,
    ) -> Result<()> {
        self.append_layer(layer_idx, keys, values)
    }

    pub fn append_layer(
        &mut self,
        layer_idx: usize,
        keys: Tensor2D,
        values: Tensor2D,
    ) -> Result<()> {
        let max_seq_len = self.config.max_seq_len.max(1);
        let incoming_rows = keys.rows;
        let (keys, values, incoming_rows) = if incoming_rows > max_seq_len {
            let keep_rows = max_seq_len;
            let start_row = incoming_rows - keep_rows;
            let start = start_row * keys.cols;
            let end = incoming_rows * keys.cols;
            (
                Tensor2D::new(keys.data[start..end].to_vec(), keep_rows, keys.cols)?,
                Tensor2D::new(values.data[start..end].to_vec(), keep_rows, values.cols)?,
                keep_rows,
            )
        } else {
            (keys, values, incoming_rows)
        };

        let target_layer_seq_len = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?
            .seq_len;
        if target_layer_seq_len.saturating_add(incoming_rows) > max_seq_len {
            let drop_rows = target_layer_seq_len.saturating_add(incoming_rows) - max_seq_len;
            self.retain_suffix(target_layer_seq_len.saturating_sub(drop_rows));
        }

        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.append(keys, values)
    }

    /// Get the current sequence length
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    pub fn base_position(&self) -> usize {
        self.base_position
    }

    pub fn next_position(&self) -> usize {
        self.base_position.saturating_add(self.seq_len())
    }

    pub fn live_layout(&self) -> LiveKVLayout {
        LiveKVLayout::from_config(&self.config)
    }

    pub fn live_sequence_metadata(
        &self,
        residency: LiveKVResidency,
        owner_session_id: Option<String>,
        owner_worker_id: Option<String>,
    ) -> LiveKVSequenceMetadata {
        let page_tokens = self.config.live_page_tokens();
        let seq_len = self.seq_len();
        LiveKVSequenceMetadata {
            next_position: self.next_position() as u32,
            first_cached_position: self.base_position as u32,
            cached_tokens: seq_len as u32,
            page_tokens,
            block_table_len: seq_len.div_ceil(page_tokens),
            tail_tokens: seq_len % page_tokens,
            residency,
            owner_session_id,
            owner_worker_id,
        }
    }

    pub fn transfer_hooks(
        &self,
        residency: LiveKVResidency,
        owner_session_id: Option<String>,
        owner_worker_id: Option<String>,
    ) -> KVTransferHooks {
        KVTransferHooks {
            live_layout: self.live_layout(),
            sequence: self.live_sequence_metadata(residency, owner_session_id, owner_worker_id),
            snapshot_encoding: KVCacheEncoding::FullSnapshotCbor,
        }
    }

    pub fn layer_seq_len(&self, layer_idx: usize) -> Result<usize> {
        Ok(self
            .layers
            .get(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?
            .seq_len)
    }

    pub fn get_layer_kv(&self, layer_idx: usize) -> Result<(&Tensor2D, &Tensor2D)> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid layer index: {}", layer_idx)))?;
        layer.validate()?;
        match (layer.get_keys(), layer.get_values()) {
            (Some(keys), Some(values)) => Ok((keys, values)),
            _ => Err(AgentError::Execution(format!(
                "Layer {} has no cached KV state",
                layer_idx
            ))),
        }
    }

    /// Check if cache has reached maximum length
    pub fn is_full(&self) -> bool {
        self.seq_len() >= self.config.max_seq_len
    }

    /// Clear all layer caches
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.base_position = 0;
    }

    /// Keep only the newest `max_len` tokens across all layers.
    pub fn retain_suffix(&mut self, max_len: usize) {
        let seq_len = self.seq_len();
        if seq_len <= max_len {
            return;
        }

        let drop_rows = seq_len - max_len;
        for layer in &mut self.layers {
            if let (Some(keys), Some(values)) = (&mut layer.keys, &mut layer.values) {
                let kv_dim = keys.cols;
                let drop_values = drop_rows * kv_dim;
                keys.data.drain(0..drop_values);
                values.data.drain(0..drop_values);
                keys.rows = max_len;
                values.rows = max_len;
                layer.seq_len = max_len;
            }
        }
        self.base_position = self.base_position.saturating_add(drop_rows);
    }

    pub fn set_base_position_for_restore(&mut self, base_position: usize) {
        self.base_position = base_position;
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.layers.iter().map(|l| l.memory_usage()).sum()
    }

    /// Serialize to bytes for checkpointing
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut buffer = Vec::new();
        ciborium::ser::into_writer(self, &mut buffer)
            .map_err(|e| AgentError::Config(format!("Failed to serialize KV cache: {}", e)))?;
        Ok(buffer)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let cache: Self = ciborium::de::from_reader(data)
            .map_err(|e| AgentError::Config(format!("Failed to deserialize KV cache: {}", e)))?;
        cache.validate()?;
        Ok(cache)
    }

    pub fn validate(&self) -> Result<()> {
        let mut expected_seq_len = None;
        for layer in &self.layers {
            layer.validate()?;
            match expected_seq_len {
                Some(expected) if expected != layer.seq_len => {
                    return Err(AgentError::Execution(format!(
                        "KV cache invariant violated across layers: expected seq_len {}, layer {} has {}",
                        expected, layer.layer_idx, layer.seq_len
                    )));
                }
                None => expected_seq_len = Some(layer.seq_len),
                _ => {}
            }
        }
        Ok(())
    }
}

/// Sliding window cache for memory efficiency
///
/// Keeps only the last N tokens in the cache, useful for long sequences.
#[derive(Debug, Clone)]
pub struct SlidingWindowCache {
    /// Inner KV cache
    inner: KVCache,
    /// Window size
    window_size: usize,
}

impl SlidingWindowCache {
    /// Create a new sliding window cache
    pub fn new(config: KVCacheConfig, window_size: usize) -> Self {
        Self {
            inner: KVCache::new(config),
            window_size,
        }
    }

    /// Update with automatic sliding
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        keys: Tensor2D,
        values: Tensor2D,
    ) -> Result<()> {
        self.inner.update_layer(layer_idx, keys, values)?;

        // Slide window if needed
        if self.inner.seq_len() > self.window_size {
            self.inner.retain_suffix(self.window_size);
        }

        Ok(())
    }

    /// Get the inner cache
    pub fn cache(&self) -> &KVCache {
        &self.inner
    }

    /// Get mutable inner cache
    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_cache_update() {
        let mut cache = LayerKVCache::new(0);

        // First update
        let keys1 = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let values1 = Tensor2D::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        cache.append(keys1, values1).unwrap();

        assert_eq!(cache.seq_len, 2);

        // Second update
        let keys2 = Tensor2D::new(vec![9.0, 10.0], 1, 2).unwrap();
        let values2 = Tensor2D::new(vec![11.0, 12.0], 1, 2).unwrap();
        cache.append(keys2, values2).unwrap();

        assert_eq!(cache.seq_len, 3);

        let keys = cache.get_keys().unwrap();
        assert_eq!(keys.rows, 3);
        assert_eq!(keys.get(2, 0), 9.0);
    }

    #[test]
    fn test_kv_cache_memory() {
        let config = KVCacheConfig {
            num_layers: 2,
            num_heads: 4,
            head_dim: 8,
            max_seq_len: 100,
        };
        let mut cache = KVCache::new(config);

        // Update layer 0
        let keys = Tensor2D::new(vec![1.0; 10 * 32], 10, 32).unwrap();
        let values = Tensor2D::new(vec![1.0; 10 * 32], 10, 32).unwrap();
        cache.update_layer(0, keys, values).unwrap();

        // 10 * 32 * 4 bytes * 2 (k+v) = 2560 bytes for layer 0
        assert!(cache.memory_usage() > 0);
    }

    #[test]
    fn test_kv_cache_retain_suffix() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 100,
        };
        let mut cache = KVCache::new(config);

        let keys = Tensor2D::new(vec![1.0; 20 * 8], 20, 8).unwrap();
        let values = Tensor2D::new(vec![1.0; 20 * 8], 20, 8).unwrap();
        cache.update_layer(0, keys, values).unwrap();

        assert_eq!(cache.seq_len(), 20);

        cache.retain_suffix(10);
        assert_eq!(cache.seq_len(), 10);
        assert_eq!(cache.base_position(), 10);
    }

    #[test]
    fn test_sliding_window() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 100,
        };
        let mut cache = SlidingWindowCache::new(config, 5);

        // Add 3 tokens
        let keys = Tensor2D::new(vec![1.0; 3 * 8], 3, 8).unwrap();
        let values = Tensor2D::new(vec![1.0; 3 * 8], 3, 8).unwrap();
        cache.update_layer(0, keys, values).unwrap();
        assert_eq!(cache.cache().seq_len(), 3);

        // Add 4 more (total 7, should slide to 5)
        let keys = Tensor2D::new(vec![2.0; 4 * 8], 4, 8).unwrap();
        let values = Tensor2D::new(vec![2.0; 4 * 8], 4, 8).unwrap();
        cache.update_layer(0, keys, values).unwrap();
        assert_eq!(cache.cache().seq_len(), 5);
        assert_eq!(cache.cache().base_position(), 2);
    }

    #[test]
    fn test_kv_cache_serialization() {
        let config = KVCacheConfig {
            num_layers: 2,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 10,
        };
        let mut cache = KVCache::new(config);

        for layer_idx in 0..2 {
            let keys = Tensor2D::new(vec![1.0; 2 * 8], 2, 8).unwrap();
            let values = Tensor2D::new(vec![2.0; 2 * 8], 2, 8).unwrap();
            cache.update_layer(layer_idx, keys, values).unwrap();
        }

        // Serialize and deserialize
        let bytes = cache.to_bytes().unwrap();
        let restored = KVCache::from_bytes(&bytes).unwrap();

        assert_eq!(restored.seq_len(), cache.seq_len());
        assert_eq!(restored.layers.len(), cache.layers.len());
        assert_eq!(restored.base_position(), cache.base_position());
    }

    #[test]
    fn test_kv_snapshot_tracks_sequence_state() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 10,
        };
        let mut cache = KVCache::new(config);
        cache
            .update_layer(
                0,
                Tensor2D::new(vec![1.0; 2 * 8], 2, 8).unwrap(),
                Tensor2D::new(vec![2.0; 2 * 8], 2, 8).unwrap(),
            )
            .unwrap();

        let snapshot = KVCacheSnapshot::from_cache(&cache, 2).unwrap();
        assert_eq!(snapshot.sequence.next_position, 2);
        assert_eq!(snapshot.sequence.cached_tokens, 2);
        assert_eq!(snapshot.sequence.first_cached_position(), 0);
        assert_eq!(snapshot.decode_cache().unwrap().seq_len(), 2);
    }

    #[test]
    fn test_kv_snapshot_tracks_nonzero_base_position() {
        let config = KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 10,
        };
        let mut cache = KVCache::new(config);
        cache
            .update_layer(
                0,
                Tensor2D::new(vec![1.0; 4 * 8], 4, 8).unwrap(),
                Tensor2D::new(vec![2.0; 4 * 8], 4, 8).unwrap(),
            )
            .unwrap();
        cache.retain_suffix(2);

        let snapshot = KVCacheSnapshot::from_cache(&cache, cache.next_position() as u32).unwrap();
        assert_eq!(snapshot.sequence.next_position, 4);
        assert_eq!(snapshot.sequence.cached_tokens, 2);
        assert_eq!(snapshot.sequence.first_cached_position(), 2);

        let restored = snapshot.decode_cache().unwrap();
        assert_eq!(restored.base_position(), 2);
        assert_eq!(restored.seq_len(), 2);
    }

    #[test]
    fn test_kv_sequence_rejects_invalid_window() {
        let err = KVSequenceState::new(1, 2).unwrap_err();
        assert!(err
            .to_string()
            .contains("cached_tokens 2 exceeds next_position 1"));
    }

    #[test]
    fn test_kv_cache_enforces_max_seq_len() {
        let mut cache = KVCache::new(KVCacheConfig {
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 2,
        });

        cache
            .append_layer(
                0,
                Tensor2D::new(vec![1.0; 2 * 8], 2, 8).unwrap(),
                Tensor2D::new(vec![2.0; 2 * 8], 2, 8).unwrap(),
            )
            .unwrap();

        cache
            .append_layer(
                0,
                Tensor2D::new(vec![3.0; 8], 1, 8).unwrap(),
                Tensor2D::new(vec![4.0; 8], 1, 8).unwrap(),
            )
            .unwrap();
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.base_position(), 1);
    }

    #[test]
    fn test_live_kv_layout_metadata_tracks_block_table() {
        let mut cache = KVCache::new(KVCacheConfig {
            num_layers: 2,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 64,
        });

        for layer_idx in 0..2 {
            cache
                .append_layer(
                    layer_idx,
                    Tensor2D::filled(17, 8, 1.0 + layer_idx as f32),
                    Tensor2D::filled(17, 8, 2.0 + layer_idx as f32),
                )
                .unwrap();
        }

        let layout = cache.live_layout();
        assert_eq!(layout.page_tokens, DEFAULT_LIVE_KV_PAGE_TOKENS);
        assert_eq!(layout.row_width, 8);
        assert_eq!(layout.page_stride_scalars, DEFAULT_LIVE_KV_PAGE_TOKENS * 8);

        let metadata = cache.live_sequence_metadata(
            LiveKVResidency::LocalOnly,
            Some("session-a".to_string()),
            Some("worker-2".to_string()),
        );
        assert_eq!(metadata.cached_tokens, 17);
        assert_eq!(metadata.block_table_len, 2);
        assert_eq!(metadata.tail_tokens, 1);
        assert_eq!(metadata.owner_session_id.as_deref(), Some("session-a"));
        assert_eq!(metadata.owner_worker_id.as_deref(), Some("worker-2"));

        let hooks = cache.transfer_hooks(
            LiveKVResidency::ImportedFromSnapshot,
            Some("session-a".to_string()),
            None,
        );
        assert_eq!(hooks.live_layout, layout);
        assert_eq!(hooks.snapshot_encoding, KVCacheEncoding::FullSnapshotCbor);
        assert_eq!(
            hooks.sequence.residency,
            LiveKVResidency::ImportedFromSnapshot
        );
    }
}
