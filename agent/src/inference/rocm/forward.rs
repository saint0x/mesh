use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use candle_core::{Device, Tensor as CandleTensor};
use uuid::Uuid;

use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::{
    RingAllReduceMetrics, StageSendChunkMode, StageSendScratch, StagedCollectiveBuffer, WorkerRing,
};
use crate::inference::engine::{BackendOptimizationProfile, LocalExecutorContract};
use crate::inference::fast_path::FastPathBackendContext;
use crate::inference::forward_pass::{ModelConfig, ModelWeights, SharedModelResidency};
use crate::inference::kv_cache::{KVCache, KVCacheConfig, KVCacheSnapshot};
use crate::inference::runtime::{sample_tokens_device_with_seeds, DeviceTensor};
use crate::inference::tensor_ops::{Tensor1D, Tensor2D};
use crate::provider::ExecutionProviderKind;
use crate::wire_f32::decode_into_f32_scratch;

use super::{RocmTensor, RocmU32Buffer};

#[derive(Debug)]
pub(crate) struct RocmLayerWeights {
    q_width: usize,
    k_width: usize,
    v_width: usize,
    w_qkv: RocmTensor,
    w_o: RocmTensor,
    gate_width: usize,
    up_width: usize,
    w_gate_up: RocmTensor,
    w_down: RocmTensor,
    attn_norm: RocmTensor,
    mlp_norm: RocmTensor,
}

#[derive(Debug)]
pub(crate) struct RocmModelWeights {
    embedding: RocmTensor,
    layers: Vec<RocmLayerWeights>,
    final_norm: RocmTensor,
    lm_head: RocmTensor,
    resident_bytes: usize,
}

impl RocmModelWeights {
    pub(crate) fn from_host(weights: &ModelWeights) -> Result<Self> {
        let embedding = RocmTensor::from_2d(&weights.embedding)?;
        let final_norm = RocmTensor::from_1d(&weights.final_norm)?;
        let lm_head = RocmTensor::from_2d(&weights.lm_head)?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for layer in &weights.layers {
            let w_qkv = RocmTensor::from_2d(&concat_projection_matrices(&[
                &layer.w_q, &layer.w_k, &layer.w_v,
            ])?)?;
            let w_gate_up =
                RocmTensor::from_2d(&concat_projection_matrices(&[&layer.w_gate, &layer.w_up])?)?;
            layers.push(RocmLayerWeights {
                q_width: layer.w_q.cols,
                k_width: layer.w_k.cols,
                v_width: layer.w_v.cols,
                w_qkv,
                w_o: RocmTensor::from_2d(&layer.w_o)?,
                gate_width: layer.w_gate.cols,
                up_width: layer.w_up.cols,
                w_gate_up,
                w_down: RocmTensor::from_2d(&layer.w_down)?,
                attn_norm: RocmTensor::from_1d(&layer.attn_norm)?,
                mlp_norm: RocmTensor::from_1d(&layer.mlp_norm)?,
            });
        }
        let resident_bytes = embedding
            .memory_usage_bytes()
            .saturating_add(final_norm.memory_usage_bytes())
            .saturating_add(lm_head.memory_usage_bytes())
            .saturating_add(
                layers
                    .iter()
                    .map(RocmLayerWeights::memory_usage_bytes)
                    .sum::<usize>(),
            );
        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            resident_bytes,
        })
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.resident_bytes
    }
}

impl RocmLayerWeights {
    fn memory_usage_bytes(&self) -> usize {
        self.w_qkv
            .memory_usage_bytes()
            .saturating_add(self.w_o.memory_usage_bytes())
            .saturating_add(self.w_gate_up.memory_usage_bytes())
            .saturating_add(self.w_down.memory_usage_bytes())
            .saturating_add(self.attn_norm.memory_usage_bytes())
            .saturating_add(self.mlp_norm.memory_usage_bytes())
    }
}

#[derive(Debug)]
struct RocmLayerKvCache {
    keys: RocmTensor,
    values: RocmTensor,
    seq_len: usize,
}

#[derive(Debug)]
struct RocmKvCache {
    layers: Vec<RocmLayerKvCache>,
    config: KVCacheConfig,
    base_position: usize,
}

#[derive(Debug, Clone, Copy)]
struct RocmAttentionLayout {
    head_dim: usize,
    q_heads_per_kv_head: usize,
    q_head_start: usize,
    local_q_heads: usize,
    kv_cols: usize,
    kv_head_start: usize,
    local_kv_heads: usize,
}

pub(crate) struct RocmForwardPass {
    weights: Arc<RocmModelWeights>,
    config: ModelConfig,
    kv_cache: RocmKvCache,
    attention_layout: RocmAttentionLayout,
    local_kv_head_indices: RocmU32Buffer,
    allreduce_timeout: Duration,
    total_workers: u32,
    position: usize,
    last_allreduce_metrics: RingAllReduceMetrics,
}

pub struct RocmExecutionBackend {
    model_id: String,
    executor_contract: LocalExecutorContract,
    forward_pass: RocmForwardPass,
}

impl RocmExecutionBackend {
    pub(crate) fn new(
        model: Arc<SharedModelResidency>,
        _worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: Duration,
    ) -> Result<Self> {
        let executor_contract = LocalExecutorContract::for_provider(ExecutionProviderKind::Rocm);
        Ok(Self {
            model_id: model.model_id().to_string(),
            forward_pass: RocmForwardPass::from_residency(
                model,
                _worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?,
            executor_contract,
        })
    }
}

#[async_trait]
impl crate::inference::backend::ExecutionBackend for RocmExecutionBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn provider_kind(&self) -> ExecutionProviderKind {
        ExecutionProviderKind::Rocm
    }

    fn optimization_profile(&self) -> BackendOptimizationProfile {
        self.executor_contract.optimization_profile
    }

    fn executor_contract(&self) -> &LocalExecutorContract {
        &self.executor_contract
    }

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        _workspace: Option<&mut crate::inference::fast_path::PrefillWorkspaceLease>,
    ) -> Result<DeviceTensor> {
        self.forward_pass
            .prefill(tokens, worker_ring, job_id)
            .await?
            .to_candle_cpu()
    }

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<DeviceTensor> {
        self.forward_pass
            .decode_step(token, worker_ring, job_id)
            .await?
            .to_candle_cpu()
    }

    fn sample(
        &self,
        logits: &DeviceTensor,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<u32> {
        let tokens = sample_tokens_device_with_seeds(logits, temperature, top_p, &[seed])?;
        tokens
            .into_iter()
            .next()
            .ok_or_else(|| AgentError::Execution("ROCm sampling produced no token ids".to_string()))
    }

    fn cache_seq_len(&self) -> usize {
        self.forward_pass.cache_seq_len()
    }

    fn live_kv_cache_bytes(&self) -> usize {
        self.forward_pass.live_kv_cache_bytes()
    }

    fn logical_kv_tokens(&self) -> usize {
        self.forward_pass.logical_kv_tokens()
    }

    fn sequence_position(&self) -> usize {
        self.forward_pass.position
    }

    fn last_allreduce_metrics(&self) -> RingAllReduceMetrics {
        self.forward_pass.last_allreduce_metrics
    }

    fn export_kv_cache(&self, max_cached_tokens: Option<usize>) -> Result<Option<KVCacheSnapshot>> {
        self.forward_pass.export_kv_cache(max_cached_tokens)
    }

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        self.forward_pass.import_kv_cache(snapshot)
    }

    fn clear(&mut self) {
        self.forward_pass.clear_cache();
    }

    fn fast_path_context(&self) -> FastPathBackendContext {
        FastPathBackendContext {
            provider: ExecutionProviderKind::Rocm,
            optimization_profile: self.optimization_profile(),
            model_id: Some(self.model_id.clone()),
            logical_kv_tokens: self.logical_kv_tokens(),
        }
    }
}

struct RocmLogits {
    tensor: RocmTensor,
}

impl RocmLogits {
    fn to_candle_cpu(self) -> Result<CandleTensor> {
        CandleTensor::from_vec(
            self.tensor.to_vec()?,
            (self.tensor.rows(), self.tensor.cols()),
            &Device::Cpu,
        )
        .map_err(|err| AgentError::Execution(format!("ROCm logits materialization failed: {err}")))
    }
}

impl RocmForwardPass {
    fn from_residency(
        residency: Arc<SharedModelResidency>,
        _worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: Duration,
    ) -> Result<Self> {
        let config = residency.config().clone();
        let kv_config = KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 4096,
        };
        let attention_layout = resolve_attention_layout(&config, shard_start, shard_end)?;
        let local_kv_head_indices =
            RocmU32Buffer::from_slice(&build_local_kv_head_indices(attention_layout)?)?;
        Ok(Self {
            weights: residency.rocm_weights()?,
            config,
            kv_cache: RocmKvCache::new(kv_config, attention_layout.kv_cols)?,
            attention_layout,
            local_kv_head_indices,
            allreduce_timeout,
            total_workers,
            position: 0,
            last_allreduce_metrics: RingAllReduceMetrics::default(),
        })
    }

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<RocmLogits> {
        if tokens.is_empty() {
            return Err(AgentError::Execution(
                "Cannot prefill an empty prompt without an explicit BOS policy".to_string(),
            ));
        }
        self.clear_cache();
        let window = self.kv_cache.config.max_seq_len.max(1);
        let start = tokens.len().saturating_sub(window);
        self.kv_cache.base_position = start;
        let hidden = self
            .forward_tokens(&tokens[start..], start, worker_ring, job_id)
            .await?;
        self.position = tokens.len();
        self.compute_logits(hidden)
    }

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<RocmLogits> {
        if self.position == 0 {
            return Err(AgentError::Execution(
                "Decode step requested before prompt prefill".to_string(),
            ));
        }
        if self.kv_cache.next_position() != self.position {
            return Err(AgentError::Execution(format!(
                "ROCm forward pass position {} diverged from KV cache next position {}",
                self.position,
                self.kv_cache.next_position()
            )));
        }
        let hidden = self
            .forward_tokens(&[token], self.position, worker_ring, job_id)
            .await?;
        self.position += 1;
        self.compute_logits(hidden)
    }

    async fn forward_tokens(
        &mut self,
        tokens: &[u32],
        absolute_position_start: usize,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<RocmTensor> {
        let positions = build_positions(absolute_position_start, tokens.len());
        let mut hidden = RocmTensor::embedding(&self.weights.embedding, tokens)?;
        self.last_allreduce_metrics = RingAllReduceMetrics::default();

        for layer_idx in 0..self.config.num_layers {
            let o_partial = {
                let layer = &self.weights.layers[layer_idx];
                let normed = hidden.rms_norm(&layer.attn_norm, self.config.rms_norm_eps)?;
                let qkv = normed.matmul(&layer.w_qkv)?;
                let q_local = qkv.copy_columns(0, layer.q_width)?;
                let k_local = qkv.copy_columns(layer.q_width, layer.k_width)?;
                let v_local = qkv.copy_columns(layer.q_width + layer.k_width, layer.v_width)?;
                let q_rope = q_local.rope(
                    &positions,
                    self.attention_layout.head_dim,
                    self.config.rope_base,
                )?;
                let k_rope = k_local.rope(
                    &positions,
                    self.attention_layout.head_dim,
                    self.config.rope_base,
                )?;
                let cache_prefix_len = self.kv_cache.layer_seq_len(layer_idx)?;
                self.kv_cache.append_layer(layer_idx, &k_rope, &v_local)?;
                let (k_cache, v_cache, cache_seq_len) = self.kv_cache.layer_view(layer_idx)?;
                let attn_output = RocmTensor::attention(
                    &q_rope,
                    k_cache,
                    v_cache,
                    &self.local_kv_head_indices,
                    cache_prefix_len,
                    cache_seq_len,
                    self.attention_layout.head_dim,
                )?;
                attn_output.matmul(&layer.w_o)?
            };
            let o_full = self
                .ring_allreduce(o_partial, worker_ring, job_id, layer_idx as u32, 0)
                .await?;
            let post_attn = hidden.add(&o_full)?;

            let down_partial = {
                let layer = &self.weights.layers[layer_idx];
                let mlp_normed = post_attn.rms_norm(&layer.mlp_norm, self.config.rms_norm_eps)?;
                let gate_up = mlp_normed.matmul(&layer.w_gate_up)?;
                let gate = gate_up.copy_columns(0, layer.gate_width)?;
                let up = gate_up.copy_columns(layer.gate_width, layer.up_width)?;
                let mlp_hidden = gate.silu()?.mul(&up)?;
                mlp_hidden.matmul(&layer.w_down)?
            };
            let down_full = self
                .ring_allreduce(down_partial, worker_ring, job_id, layer_idx as u32, 1)
                .await?;
            hidden = post_attn.add(&down_full)?;
        }

        hidden.rms_norm(&self.weights.final_norm, self.config.rms_norm_eps)
    }

    async fn ring_allreduce(
        &mut self,
        tensor: RocmTensor,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        layer_idx: u32,
        collective_seq: u32,
    ) -> Result<RocmTensor> {
        if self.total_workers <= 1 {
            return Ok(tensor);
        }
        let mut buffer = RocmCollectiveBuffer::from_tensor(tensor);
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
        Ok(buffer.into_tensor())
    }

    fn compute_logits(&self, hidden: RocmTensor) -> Result<RocmLogits> {
        let last_hidden = hidden.copy_rows(hidden.rows() - 1, 1)?;
        Ok(RocmLogits {
            tensor: last_hidden.matmul(&self.weights.lm_head)?,
        })
    }

    fn clear_cache(&mut self) {
        self.kv_cache.clear();
        self.position = 0;
    }

    fn cache_seq_len(&self) -> usize {
        self.kv_cache.seq_len()
    }

    fn logical_kv_tokens(&self) -> usize {
        self.position.saturating_sub(self.kv_cache.base_position)
    }

    fn live_kv_cache_bytes(&self) -> usize {
        self.kv_cache.memory_usage_bytes()
    }

    fn export_kv_cache(&self, max_cached_tokens: Option<usize>) -> Result<Option<KVCacheSnapshot>> {
        if self.kv_cache.seq_len() == 0 {
            return Ok(None);
        }
        let mut host = self.kv_cache.to_host_cache()?;
        if let Some(max_cached_tokens) = max_cached_tokens {
            host.retain_suffix(max_cached_tokens);
        }
        KVCacheSnapshot::from_cache(&host, self.position as u32).map(Some)
    }

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        let host = snapshot.decode_cache()?;
        self.kv_cache.import_host_cache(&host)?;
        self.position = snapshot.sequence.next_position as usize;
        Ok(())
    }
}

impl RocmKvCache {
    fn new(config: KVCacheConfig, kv_cols: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(RocmLayerKvCache {
                keys: RocmTensor::zeros(config.max_seq_len, kv_cols)?,
                values: RocmTensor::zeros(config.max_seq_len, kv_cols)?,
                seq_len: 0,
            });
        }
        Ok(Self {
            layers,
            config,
            base_position: 0,
        })
    }

    fn append_layer(
        &mut self,
        layer_idx: usize,
        keys: &RocmTensor,
        values: &RocmTensor,
    ) -> Result<()> {
        if keys.rows() != values.rows() || keys.cols() != values.cols() {
            return Err(AgentError::Execution(format!(
                "ROCm KV append shape mismatch: keys {}x{} vs values {}x{}",
                keys.rows(),
                keys.cols(),
                values.rows(),
                values.cols()
            )));
        }
        let max_seq_len = self.config.max_seq_len.max(1);
        if keys.rows() > max_seq_len {
            return Err(AgentError::Execution(format!(
                "ROCm KV append rows {} exceed max_seq_len {}",
                keys.rows(),
                max_seq_len
            )));
        }
        let layer = self.layer_mut(layer_idx)?;
        if layer.seq_len.saturating_add(keys.rows()) > max_seq_len {
            let keep = layer.seq_len.saturating_sub(
                layer
                    .seq_len
                    .saturating_add(keys.rows())
                    .saturating_sub(max_seq_len),
            );
            self.retain_suffix(keep)?;
        }
        let layer = self.layer_mut(layer_idx)?;
        keys.copy_rows_to(&layer.keys, layer.seq_len)?;
        values.copy_rows_to(&layer.values, layer.seq_len)?;
        layer.seq_len += keys.rows();
        Ok(())
    }

    fn retain_suffix(&mut self, max_len: usize) -> Result<()> {
        let seq_len = self.seq_len();
        if seq_len <= max_len {
            return Ok(());
        }
        let drop_rows = seq_len - max_len;
        for layer in &mut self.layers {
            let new_keys = RocmTensor::zeros(self.config.max_seq_len, layer.keys.cols())?;
            let new_values = RocmTensor::zeros(self.config.max_seq_len, layer.values.cols())?;
            layer
                .keys
                .copy_rows_range_to(&new_keys, drop_rows, max_len, 0)?;
            layer
                .values
                .copy_rows_range_to(&new_values, drop_rows, max_len, 0)?;
            layer.keys = new_keys;
            layer.values = new_values;
            layer.seq_len = max_len;
        }
        self.base_position = self.base_position.saturating_add(drop_rows);
        Ok(())
    }

    fn import_host_cache(&mut self, cache: &KVCache) -> Result<()> {
        self.clear();
        self.base_position = cache.base_position();
        for layer_idx in 0..self.layers.len() {
            let (keys, values) = cache.get_layer_kv(layer_idx)?;
            self.append_layer(
                layer_idx,
                &RocmTensor::from_2d(keys)?,
                &RocmTensor::from_2d(values)?,
            )?;
        }
        Ok(())
    }

    fn to_host_cache(&self) -> Result<KVCache> {
        let mut cache = KVCache::new(self.config.clone());
        cache.set_base_position_for_restore(self.base_position);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer.seq_len == 0 {
                continue;
            }
            cache.append_layer(
                layer_idx,
                layer.keys.to_2d_prefix_rows(layer.seq_len)?,
                layer.values.to_2d_prefix_rows(layer.seq_len)?,
            )?;
        }
        Ok(cache)
    }

    fn layer_seq_len(&self, layer_idx: usize) -> Result<usize> {
        Ok(self.layer(layer_idx)?.seq_len)
    }

    fn layer_view(&self, layer_idx: usize) -> Result<(&RocmTensor, &RocmTensor, usize)> {
        let layer = self.layer(layer_idx)?;
        Ok((&layer.keys, &layer.values, layer.seq_len))
    }

    fn layer(&self, layer_idx: usize) -> Result<&RocmLayerKvCache> {
        self.layers
            .get(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid ROCm KV layer {layer_idx}")))
    }

    fn layer_mut(&mut self, layer_idx: usize) -> Result<&mut RocmLayerKvCache> {
        self.layers
            .get_mut(layer_idx)
            .ok_or_else(|| AgentError::Execution(format!("Invalid ROCm KV layer {layer_idx}")))
    }

    fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.seq_len = 0;
        }
        self.base_position = 0;
    }

    fn seq_len(&self) -> usize {
        self.layers.first().map(|layer| layer.seq_len).unwrap_or(0)
    }

    fn next_position(&self) -> usize {
        self.base_position.saturating_add(self.seq_len())
    }

    fn memory_usage_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                layer
                    .keys
                    .memory_usage_bytes()
                    .saturating_add(layer.values.memory_usage_bytes())
            })
            .sum()
    }
}

struct RocmCollectiveBuffer {
    tensor: RocmTensor,
    receive_decode_scratch: Vec<f32>,
}

impl RocmCollectiveBuffer {
    fn from_tensor(tensor: RocmTensor) -> Self {
        Self {
            tensor,
            receive_decode_scratch: Vec::new(),
        }
    }

    fn into_tensor(self) -> RocmTensor {
        self.tensor
    }
}

impl StagedCollectiveBuffer for RocmCollectiveBuffer {
    fn len(&self) -> usize {
        self.tensor.len()
    }

    fn stage_send_chunk(
        &mut self,
        range: Range<usize>,
        scratch: &mut StageSendScratch,
    ) -> Result<StageSendChunkMode> {
        let staged = self.tensor.download_range(range)?;
        let scratch = scratch.ensure_vec(staged.len());
        scratch.extend(staged);
        Ok(StageSendChunkMode::HostMaterializedScratch)
    }

    fn accumulate_recv_chunk(&mut self, range: Range<usize>, payload_bytes: &[u8]) -> Result<()> {
        let payload =
            decode_into_f32_scratch(range.len(), payload_bytes, &mut self.receive_decode_scratch)?;
        let update = RocmTensor::from_1d(&Tensor1D::new(payload.to_vec()))?;
        self.tensor.add_range_from(range.start, &update)
    }

    fn copy_recv_chunk(&mut self, range: Range<usize>, payload_bytes: &[u8]) -> Result<()> {
        let payload =
            decode_into_f32_scratch(range.len(), payload_bytes, &mut self.receive_decode_scratch)?;
        self.tensor.upload_range(range.start, payload)
    }
}

fn concat_projection_matrices(tensors: &[&Tensor2D]) -> Result<Tensor2D> {
    let first = tensors.first().ok_or_else(|| {
        AgentError::Execution("cannot concatenate an empty ROCm projection set".to_string())
    })?;
    let rows = first.rows;
    let total_cols = tensors.iter().map(|tensor| tensor.cols).sum::<usize>();
    let mut data = Vec::with_capacity(rows.saturating_mul(total_cols));
    for row in 0..rows {
        for tensor in tensors {
            if tensor.rows != rows {
                return Err(AgentError::Execution(format!(
                    "ROCm projection concatenation row mismatch: expected {}, got {}",
                    rows, tensor.rows
                )));
            }
            data.extend_from_slice(tensor.row(row));
        }
    }
    Tensor2D::new(data, rows, total_cols)
}

fn resolve_attention_layout(
    config: &ModelConfig,
    shard_start: usize,
    shard_end: usize,
) -> Result<RocmAttentionLayout> {
    if config.num_heads == 0 || config.hidden_dim % config.num_heads != 0 {
        return Err(AgentError::Execution(format!(
            "Unsupported ROCm attention geometry: hidden_dim {} num_heads {}",
            config.hidden_dim, config.num_heads
        )));
    }
    if config.num_kv_heads == 0 || config.num_heads % config.num_kv_heads != 0 {
        return Err(AgentError::Execution(format!(
            "Unsupported ROCm grouped-query geometry: num_heads {} num_kv_heads {}",
            config.num_heads, config.num_kv_heads
        )));
    }
    if shard_start >= shard_end || shard_end > config.hidden_dim {
        return Err(AgentError::Execution(format!(
            "Invalid ROCm shard range {}..{} for hidden_dim {}",
            shard_start, shard_end, config.hidden_dim
        )));
    }
    let head_dim = config.hidden_dim / config.num_heads;
    let q_heads_per_kv_head = config.num_heads / config.num_kv_heads;
    let q_group_width = q_heads_per_kv_head * head_dim;
    if shard_start % q_group_width != 0 || shard_end % q_group_width != 0 {
        return Err(AgentError::Execution(format!(
            "ROCm shard range {}..{} is not aligned to grouped-query width {}",
            shard_start, shard_end, q_group_width
        )));
    }
    let local_group_count = (shard_end - shard_start) / q_group_width;
    let local_q_heads = local_group_count * q_heads_per_kv_head;
    let q_head_start = shard_start / head_dim;
    let kv_head_start = shard_start / q_group_width;
    let local_kv_heads = local_group_count;
    let kv_cols = local_kv_heads * head_dim;
    Ok(RocmAttentionLayout {
        head_dim,
        q_heads_per_kv_head,
        q_head_start,
        local_q_heads,
        kv_cols,
        kv_head_start,
        local_kv_heads,
    })
}

fn build_local_kv_head_indices(layout: RocmAttentionLayout) -> Result<Vec<u32>> {
    let mut indices = Vec::with_capacity(layout.local_q_heads);
    for local_q_idx in 0..layout.local_q_heads {
        let global_q_head = layout.q_head_start + local_q_idx;
        let global_kv_head = global_q_head / layout.q_heads_per_kv_head;
        if global_kv_head < layout.kv_head_start
            || global_kv_head >= layout.kv_head_start + layout.local_kv_heads
        {
            return Err(AgentError::Execution(format!(
                "ROCm local KV ownership mismatch: q_head {} maps to kv_head {}, local kv range {}..{}",
                global_q_head,
                global_kv_head,
                layout.kv_head_start,
                layout.kv_head_start + layout.local_kv_heads
            )));
        }
        indices.push((global_kv_head - layout.kv_head_start) as u32);
    }
    Ok(indices)
}

fn build_positions(start: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|offset| start.saturating_add(offset) as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::forward_pass::LayerWeights;
    use crate::network::{TensorPlane, TensorPlaneConfig};
    use libp2p::PeerId;

    #[test]
    fn rocm_weights_concat_projection_widths() {
        let config = ModelConfig {
            hidden_dim: 4,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 1,
            vocab_size: 8,
            intermediate_size: 6,
            local_mlp_start: 0,
            local_mlp_end: 3,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let weights = ModelWeights {
            model_id: "rocm-test".to_string(),
            embedding: Tensor2D::filled(8, 4, 0.1),
            layers: vec![LayerWeights {
                layer_idx: 0,
                w_q: Tensor2D::filled(4, 4, 0.1),
                w_k: Tensor2D::filled(4, 2, 0.2),
                w_v: Tensor2D::filled(4, 2, 0.3),
                w_o: Tensor2D::filled(4, 4, 0.4),
                w_up: Tensor2D::filled(4, 3, 0.5),
                w_gate: Tensor2D::filled(4, 3, 0.6),
                w_down: Tensor2D::filled(3, 4, 0.7),
                attn_norm: Tensor1D::new(vec![1.0; 4]),
                mlp_norm: Tensor1D::new(vec![1.0; 4]),
            }],
            final_norm: Tensor1D::new(vec![1.0; 4]),
            lm_head: Tensor2D::filled(4, 8, 0.8),
            config,
        };
        let rocm = RocmModelWeights::from_host(&weights).unwrap();
        assert_eq!(rocm.layers[0].w_qkv.cols(), 8);
        assert_eq!(rocm.layers[0].w_gate_up.cols(), 6);
    }

    #[tokio::test]
    async fn rocm_forward_prefill_and_decode_produce_logits() {
        let weights = tiny_model_weights();
        let config = weights.config.clone();
        let layout = resolve_attention_layout(&config, 0, config.hidden_dim).unwrap();
        let kv_config = KVCacheConfig {
            num_layers: config.num_layers,
            num_heads: config.num_kv_heads,
            head_dim: config.hidden_dim / config.num_heads,
            max_seq_len: 16,
        };
        let mut forward = RocmForwardPass {
            weights: Arc::new(RocmModelWeights::from_host(&weights).unwrap()),
            config,
            kv_cache: RocmKvCache::new(kv_config, layout.kv_cols).unwrap(),
            attention_layout: layout,
            local_kv_head_indices: RocmU32Buffer::from_slice(
                &build_local_kv_head_indices(layout).unwrap(),
            )
            .unwrap(),
            allreduce_timeout: Duration::from_secs(30),
            total_workers: 1,
            position: 0,
            last_allreduce_metrics: RingAllReduceMetrics::default(),
        };
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
            ExecutionProviderKind::Rocm,
            LocalExecutorContract::for_provider(ExecutionProviderKind::Rocm),
            None,
            &mut tensor_plane,
        );

        let prefill = forward
            .prefill(&[1, 2, 3], &mut worker_ring, Uuid::new_v4())
            .await
            .unwrap()
            .tensor;
        assert_eq!(prefill.rows(), 1);
        assert_eq!(prefill.cols(), weights.config.vocab_size);
        assert_eq!(forward.cache_seq_len(), 3);

        let decode = forward
            .decode_step(4, &mut worker_ring, Uuid::new_v4())
            .await
            .unwrap()
            .tensor;
        assert_eq!(decode.rows(), 1);
        assert_eq!(decode.cols(), weights.config.vocab_size);
        assert_eq!(forward.cache_seq_len(), 4);
        assert!(decode
            .to_vec()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }

    fn tiny_model_weights() -> ModelWeights {
        let config = ModelConfig {
            hidden_dim: 4,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 1,
            vocab_size: 8,
            intermediate_size: 4,
            local_mlp_start: 0,
            local_mlp_end: 4,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        ModelWeights {
            model_id: "rocm-forward-test".to_string(),
            embedding: Tensor2D::new(
                (0..32).map(|idx| ((idx % 7) as f32 - 3.0) * 0.03).collect(),
                8,
                4,
            )
            .unwrap(),
            layers: vec![LayerWeights {
                layer_idx: 0,
                w_q: patterned_matrix(4, 4, 0.02),
                w_k: patterned_matrix(4, 2, 0.03),
                w_v: patterned_matrix(4, 2, 0.04),
                w_o: patterned_matrix(4, 4, 0.02),
                w_up: patterned_matrix(4, 4, 0.03),
                w_gate: patterned_matrix(4, 4, 0.025),
                w_down: patterned_matrix(4, 4, 0.02),
                attn_norm: Tensor1D::new(vec![1.0; 4]),
                mlp_norm: Tensor1D::new(vec![1.0; 4]),
            }],
            final_norm: Tensor1D::new(vec![1.0; 4]),
            lm_head: patterned_matrix(4, 8, 0.015),
            config,
        }
    }

    fn patterned_matrix(rows: usize, cols: usize, scale: f32) -> Tensor2D {
        Tensor2D::new(
            (0..rows * cols)
                .map(|idx| ((idx % 11) as f32 - 5.0) * scale)
                .collect(),
            rows,
            cols,
        )
        .unwrap()
    }
}
