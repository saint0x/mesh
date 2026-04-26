use async_trait::async_trait;
use std::time::Instant;
use uuid::Uuid;

use crate::errors::Result;
use crate::executor::ring_allreduce::{RingAllReduceMetrics, WorkerRing};
use crate::provider::{selected_execution_provider, ExecutionProviderKind};

use super::engine::BackendOptimizationProfile;
use super::forward_pass::{ForwardPass, ModelWeights};
use super::kv_cache::KVCacheSnapshot;
use super::tensor_ops::Tensor1D;

#[async_trait]
pub trait ExecutionBackend: Send {
    fn provider_kind(&self) -> ExecutionProviderKind;

    fn optimization_profile(&self) -> BackendOptimizationProfile;

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

    fn export_kv_cache(&self) -> Result<Option<KVCacheSnapshot>>;

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()>;

    fn clear(&mut self);
}

pub struct DecodeMicrobatchRequest<'a> {
    pub session_id: Uuid,
    pub job_id: Uuid,
    pub token: u32,
    pub backend: &'a mut dyn ExecutionBackend,
}

pub struct DecodeMicrobatchOutput {
    pub session_id: Uuid,
    pub logits: Tensor1D,
    pub execution_time_ms: u64,
}

pub struct BackendMicrobatchExecutor;

impl BackendMicrobatchExecutor {
    pub async fn decode_step_batch(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
    ) -> Result<Vec<DecodeMicrobatchOutput>> {
        let profile = requests
            .first()
            .map(|request| request.backend.optimization_profile())
            .unwrap_or(BackendOptimizationProfile::CpuSerial);

        match profile {
            BackendOptimizationProfile::CpuSerial => {
                Self::decode_step_batch_serial(requests, worker_ring).await
            }
            BackendOptimizationProfile::MetalVectorized => {
                Self::decode_step_batch_accelerated(requests, worker_ring).await
            }
            BackendOptimizationProfile::CudaFused => {
                Self::decode_step_batch_accelerated(requests, worker_ring).await
            }
        }
    }

    async fn decode_step_batch_serial(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
    ) -> Result<Vec<DecodeMicrobatchOutput>> {
        let mut outputs = Vec::with_capacity(requests.len());
        for request in requests.iter_mut() {
            let step_start = Instant::now();
            let logits = request
                .backend
                .decode_step(request.token, worker_ring, request.job_id)
                .await?;
            outputs.push(DecodeMicrobatchOutput {
                session_id: request.session_id,
                logits,
                execution_time_ms: step_start.elapsed().as_millis() as u64,
            });
        }
        Ok(outputs)
    }

    async fn decode_step_batch_accelerated(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
    ) -> Result<Vec<DecodeMicrobatchOutput>> {
        // Provider-specialized fast paths keep the same runtime API while
        // allowing Metal/CUDA backends to swap in fused kernels later.
        Self::decode_step_batch_serial(requests, worker_ring).await
    }
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

    fn optimization_profile(&self) -> BackendOptimizationProfile {
        match self.provider {
            ExecutionProviderKind::Cpu => BackendOptimizationProfile::CpuSerial,
            ExecutionProviderKind::Metal => BackendOptimizationProfile::MetalVectorized,
            ExecutionProviderKind::Cuda => BackendOptimizationProfile::CudaFused,
        }
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

    fn export_kv_cache(&self) -> Result<Option<KVCacheSnapshot>> {
        if self.forward_pass.kv_cache.seq_len() == 0 {
            return Ok(None);
        }
        KVCacheSnapshot::from_cache(
            &self.forward_pass.kv_cache,
            self.forward_pass.position as u32,
        )
        .map(Some)
    }

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        snapshot.validate()?;
        let kv_cache = snapshot.decode_cache()?;
        if kv_cache.seq_len() as u32 != snapshot.sequence.cached_tokens {
            return Err(crate::errors::AgentError::Execution(format!(
                "Recovered KV cache length {} does not match snapshot cached token count {}",
                kv_cache.seq_len(),
                snapshot.sequence.cached_tokens
            )));
        }
        let sequence_position = snapshot.sequence.next_position as usize;
        self.forward_pass.kv_cache = kv_cache;
        self.forward_pass.position = sequence_position;
        Ok(())
    }

    fn clear(&mut self) {
        self.forward_pass.clear_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ProfileBackend {
        provider: ExecutionProviderKind,
    }

    #[async_trait]
    impl ExecutionBackend for ProfileBackend {
        fn provider_kind(&self) -> ExecutionProviderKind {
            self.provider
        }

        fn optimization_profile(&self) -> BackendOptimizationProfile {
            match self.provider {
                ExecutionProviderKind::Cpu => BackendOptimizationProfile::CpuSerial,
                ExecutionProviderKind::Metal => BackendOptimizationProfile::MetalVectorized,
                ExecutionProviderKind::Cuda => BackendOptimizationProfile::CudaFused,
            }
        }

        async fn prefill(
            &mut self,
            _tokens: &[u32],
            _worker_ring: &mut WorkerRing<'_>,
            _job_id: Uuid,
        ) -> Result<Tensor1D> {
            Ok(Tensor1D::zeros(1))
        }

        async fn decode_step(
            &mut self,
            _token: u32,
            _worker_ring: &mut WorkerRing<'_>,
            _job_id: Uuid,
        ) -> Result<Tensor1D> {
            Ok(Tensor1D::zeros(1))
        }

        fn sample(&self, _logits: &Tensor1D, _temperature: f32, _top_p: f32, _seed: u64) -> u32 {
            0
        }

        fn cache_seq_len(&self) -> usize {
            0
        }

        fn cache_memory_usage(&self) -> usize {
            0
        }

        fn sequence_position(&self) -> usize {
            0
        }

        fn last_allreduce_metrics(&self) -> RingAllReduceMetrics {
            RingAllReduceMetrics::default()
        }

        fn export_kv_cache(&self) -> Result<Option<KVCacheSnapshot>> {
            Ok(None)
        }

        fn import_kv_cache(&mut self, _snapshot: &KVCacheSnapshot) -> Result<()> {
            Ok(())
        }

        fn clear(&mut self) {}
    }

    #[test]
    fn backend_optimization_profile_tracks_provider_kind() {
        assert_eq!(
            ProfileBackend {
                provider: ExecutionProviderKind::Cpu
            }
            .optimization_profile(),
            BackendOptimizationProfile::CpuSerial
        );
        assert_eq!(
            ProfileBackend {
                provider: ExecutionProviderKind::Metal
            }
            .optimization_profile(),
            BackendOptimizationProfile::MetalVectorized
        );
        assert_eq!(
            ProfileBackend {
                provider: ExecutionProviderKind::Cuda
            }
            .optimization_profile(),
            BackendOptimizationProfile::CudaFused
        );
    }
}
