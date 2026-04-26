use async_trait::async_trait;
use uuid::Uuid;

use crate::errors::Result;
use crate::executor::ring_allreduce::{RingAllReduceMetrics, WorkerRing};
use crate::provider::{selected_execution_provider, ExecutionProviderKind};

use super::forward_pass::{ForwardPass, ModelWeights};
use super::kv_cache::KVCache;
use super::tensor_ops::Tensor1D;

#[async_trait]
pub trait ExecutionBackend: Send {
    fn provider_kind(&self) -> ExecutionProviderKind;

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<Tensor1D>;

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<Tensor1D>;

    fn sample(&self, logits: &Tensor1D, temperature: f32, top_p: f32, seed: u64) -> u32;

    fn cache_seq_len(&self) -> usize;

    fn cache_memory_usage(&self) -> usize;

    fn sequence_position(&self) -> usize;

    fn last_allreduce_metrics(&self) -> RingAllReduceMetrics;

    fn export_kv_cache(&self) -> Result<Option<Vec<u8>>>;

    fn import_kv_cache(&mut self, bytes: &[u8], sequence_position: usize) -> Result<()>;

    fn clear(&mut self);
}

pub struct CandleExecutionBackend {
    provider: ExecutionProviderKind,
    forward_pass: ForwardPass,
}

impl CandleExecutionBackend {
    pub fn new(
        weights: ModelWeights,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            provider: selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu),
            forward_pass: ForwardPass::new(
                weights,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?,
        })
    }
}

#[async_trait]
impl ExecutionBackend for CandleExecutionBackend {
    fn provider_kind(&self) -> ExecutionProviderKind {
        self.provider
    }

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<Tensor1D> {
        self.forward_pass.prefill(tokens, worker_ring, job_id).await
    }

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<Tensor1D> {
        self.forward_pass
            .decode_step(token, worker_ring, job_id)
            .await
    }

    fn sample(&self, logits: &Tensor1D, temperature: f32, top_p: f32, seed: u64) -> u32 {
        self.forward_pass.sample(logits, temperature, top_p, seed)
    }

    fn cache_seq_len(&self) -> usize {
        self.forward_pass.cache_seq_len()
    }

    fn cache_memory_usage(&self) -> usize {
        self.forward_pass.cache_memory_usage()
    }

    fn sequence_position(&self) -> usize {
        self.forward_pass.position
    }

    fn last_allreduce_metrics(&self) -> RingAllReduceMetrics {
        self.forward_pass.last_allreduce_metrics
    }

    fn export_kv_cache(&self) -> Result<Option<Vec<u8>>> {
        if self.forward_pass.kv_cache.seq_len() == 0 {
            return Ok(None);
        }
        self.forward_pass.kv_cache.to_bytes().map(Some)
    }

    fn import_kv_cache(&mut self, bytes: &[u8], sequence_position: usize) -> Result<()> {
        let kv_cache = KVCache::from_bytes(bytes)?;
        if kv_cache.seq_len() != sequence_position {
            return Err(crate::errors::AgentError::Execution(format!(
                "Recovered KV cache length {} does not match authoritative sequence position {}",
                kv_cache.seq_len(),
                sequence_position
            )));
        }
        self.forward_pass.kv_cache = kv_cache;
        self.forward_pass.position = sequence_position;
        Ok(())
    }

    fn clear(&mut self) {
        self.forward_pass.clear_cache();
    }
}
