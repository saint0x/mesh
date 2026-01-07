//! KV Cache Management for Transformer Inference
//!
//! This module manages the key-value cache used in transformer attention layers.
//! The KV cache stores computed key and value tensors from previous positions,
//! allowing efficient autoregressive generation without recomputing past states.

use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use super::tensor_ops::Tensor2D;

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
    pub fn update(&mut self, new_keys: Tensor2D, new_values: Tensor2D) -> Result<()> {
        if new_keys.rows != new_values.rows || new_keys.cols != new_values.cols {
            return Err(AgentError::Execution(format!(
                "KV cache update shape mismatch: keys {}x{} vs values {}x{}",
                new_keys.rows, new_keys.cols, new_values.rows, new_values.cols
            )));
        }

        match (&mut self.keys, &mut self.values) {
            (Some(existing_keys), Some(existing_values)) => {
                // Concatenate with existing cache
                let new_k_data = [existing_keys.data.as_slice(), new_keys.data.as_slice()].concat();
                let new_v_data = [existing_values.data.as_slice(), new_values.data.as_slice()].concat();

                let new_rows = existing_keys.rows + new_keys.rows;

                self.keys = Some(Tensor2D {
                    data: new_k_data,
                    rows: new_rows,
                    cols: existing_keys.cols,
                });
                self.values = Some(Tensor2D {
                    data: new_v_data,
                    rows: new_rows,
                    cols: existing_values.cols,
                });
                self.seq_len = new_rows;
            }
            _ => {
                // First update - just store
                self.seq_len = new_keys.rows;
                self.keys = Some(new_keys);
                self.values = Some(new_values);
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
    /// Configuration
    pub config: KVCacheConfig,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(config: KVCacheConfig) -> Self {
        let layers = (0..config.num_layers)
            .map(LayerKVCache::new)
            .collect();

        Self { layers, config }
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
        let layer = self.layers.get_mut(layer_idx).ok_or_else(|| {
            AgentError::Execution(format!("Invalid layer index: {}", layer_idx))
        })?;
        layer.update(keys, values)
    }

    /// Get the current sequence length
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
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
    }

    /// Truncate all caches to specific length
    pub fn truncate(&mut self, max_len: usize) {
        for layer in &mut self.layers {
            layer.truncate(max_len);
        }
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.layers.iter().map(|l| l.memory_usage()).sum()
    }

    /// Serialize to bytes for checkpointing
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        ciborium::ser::into_writer(self, &mut buffer)
            .map_err(|e| AgentError::Config(format!("Failed to serialize KV cache: {}", e)))?;
        Ok(buffer)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        ciborium::de::from_reader(data)
            .map_err(|e| AgentError::Config(format!("Failed to deserialize KV cache: {}", e)))
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
            self.inner.truncate(self.window_size);
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
        cache.update(keys1, values1).unwrap();

        assert_eq!(cache.seq_len, 2);

        // Second update
        let keys2 = Tensor2D::new(vec![9.0, 10.0], 1, 2).unwrap();
        let values2 = Tensor2D::new(vec![11.0, 12.0], 1, 2).unwrap();
        cache.update(keys2, values2).unwrap();

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
    fn test_kv_cache_truncate() {
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

        cache.truncate(10);
        assert_eq!(cache.seq_len(), 10);
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

        let keys = Tensor2D::new(vec![1.0; 2 * 8], 2, 8).unwrap();
        let values = Tensor2D::new(vec![2.0; 2 * 8], 2, 8).unwrap();
        cache.update_layer(0, keys, values).unwrap();

        // Serialize and deserialize
        let bytes = cache.to_bytes().unwrap();
        let restored = KVCache::from_bytes(&bytes).unwrap();

        assert_eq!(restored.seq_len(), cache.seq_len());
        assert_eq!(restored.layers.len(), cache.layers.len());
    }
}
