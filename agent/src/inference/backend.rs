use async_trait::async_trait;
use std::any::Any;
use std::time::Instant;
use uuid::Uuid;

use crate::errors::Result;
use crate::executor::ring_allreduce::{RingAllReduceMetrics, WorkerRing};
use crate::provider::{selected_execution_provider, ExecutionProviderKind};
use candle_core::Tensor as CandleTensor;

use super::engine::BackendOptimizationProfile;
use std::sync::Arc;

use super::forward_pass::{ForwardPass, SharedModelResidency};
use super::kv_cache::KVCacheSnapshot;
use super::tensor_ops::Tensor1D;

#[derive(Clone)]
pub enum BackendLogits {
    Host(Tensor1D),
    Device(CandleTensor),
}

#[async_trait]
pub trait ExecutionBackend: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn provider_kind(&self) -> ExecutionProviderKind;

    fn optimization_profile(&self) -> BackendOptimizationProfile;

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits>;

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits>;

    fn sample(&self, logits: &BackendLogits, temperature: f32, top_p: f32, seed: u64) -> u32;

    fn cache_seq_len(&self) -> usize;

    fn live_kv_cache_bytes(&self) -> usize;

    fn logical_kv_tokens(&self) -> usize;

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
    pub logits: BackendLogits,
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
        if let Some(outputs) =
            CandleExecutionBackend::decode_step_batch(requests, worker_ring).await?
        {
            return Ok(outputs);
        }
        Self::decode_step_batch_serial(requests, worker_ring).await
    }
}

pub struct CandleExecutionBackend {
    pub(crate) model_id: String,
    pub(crate) provider: ExecutionProviderKind,
    pub(crate) forward_pass: ForwardPass,
}

impl CandleExecutionBackend {
    pub fn new(
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            model_id: model.model_id().to_string(),
            provider: selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu),
            forward_pass: ForwardPass::from_residency(
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?,
        })
    }

    async fn decode_step_batch(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
    ) -> Result<Option<Vec<DecodeMicrobatchOutput>>> {
        if requests.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let mut backends = Vec::with_capacity(requests.len());
        let mut session_ids = Vec::with_capacity(requests.len());
        let mut job_ids = Vec::with_capacity(requests.len());
        let mut tokens = Vec::with_capacity(requests.len());
        let mut expected_model_id = None::<String>;
        let mut expected_provider = None::<ExecutionProviderKind>;

        for request in requests.iter_mut() {
            let Some(backend) = request
                .backend
                .as_any_mut()
                .downcast_mut::<CandleExecutionBackend>()
            else {
                return Ok(None);
            };

            if backend.forward_pass.position == 0 {
                return Ok(None);
            }

            if let Some(model_id) = &expected_model_id {
                if model_id != &backend.model_id {
                    return Ok(None);
                }
            } else {
                expected_model_id = Some(backend.model_id.clone());
            }

            if let Some(provider) = expected_provider {
                if provider != backend.provider {
                    return Ok(None);
                }
            } else {
                expected_provider = Some(backend.provider);
            }

            backends.push(backend);
            session_ids.push(request.session_id);
            job_ids.push(request.job_id);
            tokens.push(request.token);
        }

        let step_start = Instant::now();
        let logits =
            ForwardPass::decode_microbatch(&mut backends, &tokens, &job_ids, worker_ring).await?;
        let execution_time_ms = step_start.elapsed().as_millis() as u64;

        Ok(Some(
            session_ids
                .into_iter()
                .zip(logits)
                .map(|(session_id, logits)| DecodeMicrobatchOutput {
                    session_id,
                    logits,
                    execution_time_ms,
                })
                .collect(),
        ))
    }
}

#[async_trait]
impl ExecutionBackend for CandleExecutionBackend {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

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
    ) -> Result<BackendLogits> {
        self.forward_pass.prefill(tokens, worker_ring, job_id).await
    }

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits> {
        self.forward_pass
            .decode_step(token, worker_ring, job_id)
            .await
    }

    fn sample(&self, logits: &BackendLogits, temperature: f32, top_p: f32, seed: u64) -> u32 {
        self.forward_pass.sample(logits, temperature, top_p, seed)
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

    fn export_kv_cache(&self) -> Result<Option<KVCacheSnapshot>> {
        self.forward_pass.export_kv_cache_snapshot()
    }

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        self.forward_pass.import_kv_cache_snapshot(snapshot)
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
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

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
        ) -> Result<BackendLogits> {
            Ok(BackendLogits::Host(Tensor1D::zeros(1)))
        }

        async fn decode_step(
            &mut self,
            _token: u32,
            _worker_ring: &mut WorkerRing<'_>,
            _job_id: Uuid,
        ) -> Result<BackendLogits> {
            Ok(BackendLogits::Host(Tensor1D::zeros(1)))
        }

        fn sample(
            &self,
            _logits: &BackendLogits,
            _temperature: f32,
            _top_p: f32,
            _seed: u64,
        ) -> u32 {
            0
        }

        fn cache_seq_len(&self) -> usize {
            0
        }

        fn live_kv_cache_bytes(&self) -> usize {
            0
        }

        fn logical_kv_tokens(&self) -> usize {
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
