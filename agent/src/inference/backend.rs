use async_trait::async_trait;
use std::any::Any;
use std::time::Instant;
use tracing::warn;
use uuid::Uuid;

use crate::errors::Result;
use crate::executor::ring_allreduce::{RingAllReduceMetrics, WorkerRing};
use crate::provider::{selected_execution_provider, ExecutionProviderKind};

use super::engine::{BackendOptimizationProfile, LocalExecutorContract};
use super::fast_path::{
    DecodeWorkspaceLease, FastPathBackendContext, FastPathExecutionPlan, FastPathPlanner,
    PrefillWorkspaceLease,
};
use std::sync::Arc;

use super::forward_pass::{ForwardPass, SharedModelResidency};
use super::kv_cache::KVCacheSnapshot;
use super::runtime::{runtime_error, sample_tokens_device_with_seeds, DeviceTensor};
use super::tensor_ops::Tensor1D;

#[derive(Clone)]
pub enum BackendLogits {
    Host(Tensor1D),
    Device(DeviceTensor),
}

#[async_trait]
pub trait ExecutionBackend: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn provider_kind(&self) -> ExecutionProviderKind;

    fn optimization_profile(&self) -> BackendOptimizationProfile;

    fn executor_contract(&self) -> &LocalExecutorContract;

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        workspace: Option<&mut PrefillWorkspaceLease>,
    ) -> Result<BackendLogits>;

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits>;

    fn sample(
        &self,
        logits: &BackendLogits,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<u32>;

    fn cache_seq_len(&self) -> usize;

    fn live_kv_cache_bytes(&self) -> usize;

    fn logical_kv_tokens(&self) -> usize;

    fn sequence_position(&self) -> usize;

    fn last_allreduce_metrics(&self) -> RingAllReduceMetrics;

    fn export_kv_cache(&self, max_cached_tokens: Option<usize>) -> Result<Option<KVCacheSnapshot>>;

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()>;

    fn clear(&mut self);

    fn fast_path_context(&self) -> FastPathBackendContext;

    fn supports_decode_microbatch(&self) -> bool {
        self.executor_contract().supports_decode_microbatch()
    }

    fn is_fast_path_backend(&self) -> bool {
        self.executor_contract().is_fast_path()
    }
}

pub struct DecodeMicrobatchRequest<'a> {
    pub session_id: Uuid,
    pub job_id: Uuid,
    pub token: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub sampling_seed: u64,
    pub backend: &'a mut dyn ExecutionBackend,
}

pub struct DecodeMicrobatchOutput {
    pub session_id: Uuid,
    pub logits: Option<BackendLogits>,
    pub sampled_token: Option<u32>,
    pub execution_time_ms: u64,
}

pub struct BackendMicrobatchExecutor;

impl BackendMicrobatchExecutor {
    pub async fn decode_step_batch(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        plan: Option<&FastPathExecutionPlan>,
        worker_ring: &mut WorkerRing<'_>,
        workspace: Option<&mut DecodeWorkspaceLease>,
    ) -> Result<Vec<DecodeMicrobatchOutput>> {
        if let Some(plan) = plan {
            let contexts = requests
                .iter()
                .map(|request| request.backend.fast_path_context())
                .collect::<Vec<_>>();
            FastPathPlanner::validate_decode_contexts(plan, &contexts)?;
        }

        if !Self::can_use_accelerated_decode_batch(requests) {
            return Self::decode_step_batch_serial(requests, worker_ring).await;
        }

        let profile = requests
            .first()
            .map(|request| request.backend.optimization_profile())
            .unwrap_or(BackendOptimizationProfile::CpuSerial);
        let supports_decode_microbatch = requests
            .first()
            .map(|request| request.backend.supports_decode_microbatch())
            .unwrap_or(false);
        match profile {
            BackendOptimizationProfile::CpuSerial => {
                if supports_decode_microbatch {
                    Self::decode_step_batch_fast_path(requests, worker_ring, workspace).await
                } else {
                    Self::decode_step_batch_serial(requests, worker_ring).await
                }
            }
            BackendOptimizationProfile::MetalVectorized => {
                Self::decode_step_batch_fast_path(requests, worker_ring, workspace).await
            }
            BackendOptimizationProfile::CudaFused => {
                Self::decode_step_batch_fast_path(requests, worker_ring, workspace).await
            }
        }
    }

    fn can_use_accelerated_decode_batch(requests: &[DecodeMicrobatchRequest<'_>]) -> bool {
        let Some(first_request) = requests.first() else {
            return true;
        };

        if !first_request.backend.supports_decode_microbatch() {
            return false;
        }

        let profile = first_request.backend.optimization_profile();
        let contract = first_request.backend.executor_contract().clone();
        requests.iter().all(|request| {
            request.backend.supports_decode_microbatch()
                && request.backend.optimization_profile() == profile
                && request.backend.executor_contract() == &contract
        })
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
                logits: Some(logits),
                sampled_token: None,
                execution_time_ms: step_start.elapsed().as_millis() as u64,
            });
        }
        Ok(outputs)
    }

    async fn decode_step_batch_fast_path(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
        workspace: Option<&mut DecodeWorkspaceLease>,
    ) -> Result<Vec<DecodeMicrobatchOutput>> {
        if let Some(outputs) =
            ProviderExecutionBackend::decode_step_batch_fast_path(requests, worker_ring, workspace)
                .await?
        {
            return Ok(outputs);
        }
        Err(crate::errors::AgentError::Execution(
            "Fast-path decode batch could not be materialized for the requested provider contract"
                .to_string(),
        ))
    }
}

/// Shared provider runtime core used by the concrete CPU / Metal / CUDA backends.
pub(crate) struct ProviderRuntimeCore {
    pub(crate) model_id: String,
    pub(crate) provider: ExecutionProviderKind,
    pub(crate) executor_contract: LocalExecutorContract,
    pub(crate) forward_pass: ForwardPass,
}

impl ProviderRuntimeCore {
    pub fn new(
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        let provider = selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu);
        Self::new_for_provider(
            provider,
            model,
            worker_position,
            shard_start,
            shard_end,
            total_workers,
            allreduce_timeout,
        )
    }

    pub fn new_for_provider(
        provider: ExecutionProviderKind,
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            model_id: model.model_id().to_string(),
            provider,
            executor_contract: LocalExecutorContract::for_provider(provider),
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

    fn split_logits_rows(logits_2d: &DeviceTensor) -> Result<Vec<BackendLogits>> {
        let dims = logits_2d.dims();
        let mut logits = Vec::with_capacity(dims.first().copied().unwrap_or(0));
        for row_idx in 0..dims.first().copied().unwrap_or(0) {
            let row = logits_2d.narrow(0, row_idx, 1).map_err(runtime_error)?;
            logits.push(BackendLogits::Device(row));
        }
        Ok(logits)
    }

    fn sample_fast_path_logits_batch(
        logits_2d: &DeviceTensor,
        requests: &[DecodeMicrobatchRequest<'_>],
    ) -> Result<Vec<u32>> {
        let dims = logits_2d.dims();
        if dims.len() != 2 || dims[0] != requests.len() {
            return Err(crate::errors::AgentError::Execution(format!(
                "Fast-path decode logits shape {:?} does not match {} requests",
                dims,
                requests.len()
            )));
        }

        let mut sampled_tokens = vec![0u32; requests.len()];
        let mut start = 0usize;
        while start < requests.len() {
            let temperature = requests[start].temperature;
            let top_p = requests[start].top_p;
            let mut end = start + 1;
            while end < requests.len()
                && requests[end].temperature.to_bits() == temperature.to_bits()
                && requests[end].top_p.to_bits() == top_p.to_bits()
            {
                end += 1;
            }

            let row_count = end - start;
            let row_logits = logits_2d
                .narrow(0, start, row_count)
                .map_err(runtime_error)?;
            let seeds = requests[start..end]
                .iter()
                .map(|request| request.sampling_seed)
                .collect::<Vec<_>>();
            let group_tokens =
                sample_tokens_device_with_seeds(&row_logits, temperature, top_p, &seeds)?;
            sampled_tokens[start..end].copy_from_slice(&group_tokens);
            start = end;
        }

        Ok(sampled_tokens)
    }
}

pub struct CpuExecutionBackend {
    core: ProviderRuntimeCore,
}

impl CpuExecutionBackend {
    pub fn new(
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            core: ProviderRuntimeCore::new_for_provider(
                ExecutionProviderKind::Cpu,
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?,
        })
    }
}

pub struct MetalExecutionBackend {
    core: ProviderRuntimeCore,
}

impl MetalExecutionBackend {
    pub fn new(
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            core: ProviderRuntimeCore::new_for_provider(
                ExecutionProviderKind::Metal,
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?,
        })
    }
}

pub struct CudaExecutionBackend {
    core: ProviderRuntimeCore,
}

impl CudaExecutionBackend {
    pub fn new(
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        Ok(Self {
            core: ProviderRuntimeCore::new_for_provider(
                ExecutionProviderKind::Cuda,
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?,
        })
    }
}

pub enum ProviderExecutionBackend {
    Cpu(CpuExecutionBackend),
    Metal(MetalExecutionBackend),
    Cuda(CudaExecutionBackend),
}

impl ProviderExecutionBackend {
    pub fn new(
        model: Arc<SharedModelResidency>,
        worker_position: u32,
        shard_start: usize,
        shard_end: usize,
        total_workers: u32,
        allreduce_timeout: std::time::Duration,
    ) -> Result<Self> {
        match selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu) {
            ExecutionProviderKind::Cpu => Ok(Self::Cpu(CpuExecutionBackend::new(
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?)),
            ExecutionProviderKind::Metal => Ok(Self::Metal(MetalExecutionBackend::new(
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?)),
            ExecutionProviderKind::Cuda => Ok(Self::Cuda(CudaExecutionBackend::new(
                model,
                worker_position,
                shard_start,
                shard_end,
                total_workers,
                allreduce_timeout,
            )?)),
        }
    }

    fn core(&self) -> &ProviderRuntimeCore {
        match self {
            Self::Cpu(backend) => &backend.core,
            Self::Metal(backend) => &backend.core,
            Self::Cuda(backend) => &backend.core,
        }
    }

    fn core_mut(&mut self) -> &mut ProviderRuntimeCore {
        match self {
            Self::Cpu(backend) => &mut backend.core,
            Self::Metal(backend) => &mut backend.core,
            Self::Cuda(backend) => &mut backend.core,
        }
    }

    async fn decode_step_batch_fast_path(
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
        workspace: Option<&mut DecodeWorkspaceLease>,
    ) -> Result<Option<Vec<DecodeMicrobatchOutput>>> {
        let Some(first_request) = requests.first() else {
            return Ok(Some(Vec::new()));
        };
        let provider = first_request.backend.provider_kind();
        Self::decode_step_batch_fast_path_with_provider(provider, requests, worker_ring, workspace)
            .await
    }

    async fn decode_step_batch_fast_path_with_provider(
        expected_provider: ExecutionProviderKind,
        requests: &mut [DecodeMicrobatchRequest<'_>],
        worker_ring: &mut WorkerRing<'_>,
        workspace: Option<&mut DecodeWorkspaceLease>,
    ) -> Result<Option<Vec<DecodeMicrobatchOutput>>> {
        let mut backends = Vec::with_capacity(requests.len());
        let mut job_ids = Vec::with_capacity(requests.len());
        let mut tokens = Vec::with_capacity(requests.len());
        let mut expected_model_id = None::<String>;
        let mut expected_contract = None::<LocalExecutorContract>;

        for request in requests.iter_mut() {
            let Some(provider_backend) = request
                .backend
                .as_any_mut()
                .downcast_mut::<ProviderExecutionBackend>()
            else {
                return Ok(None);
            };
            if provider_backend.provider_kind() != expected_provider {
                return Ok(None);
            }
            let backend = provider_backend.core_mut();

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

            if !backend.supports_decode_microbatch() {
                return Ok(None);
            }

            if let Some(contract) = &expected_contract {
                if contract != &backend.executor_contract {
                    return Ok(None);
                }
            } else {
                expected_contract = Some(backend.executor_contract.clone());
            }

            backends.push(backend);
            job_ids.push(request.job_id);
            tokens.push(request.token);
        }

        let step_start = Instant::now();
        let logits_2d = ForwardPass::fast_path_decode_microbatch(
            &mut backends,
            &tokens,
            &job_ids,
            worker_ring,
            workspace,
        )
        .await?;
        let execution_time_ms = step_start.elapsed().as_millis() as u64;

        if let Ok(sampled_tokens) =
            ProviderRuntimeCore::sample_fast_path_logits_batch(&logits_2d, requests)
        {
            return Ok(Some(
                requests
                    .iter()
                    .map(|request| request.session_id)
                    .zip(sampled_tokens)
                    .map(|(session_id, sampled_token)| DecodeMicrobatchOutput {
                        session_id,
                        logits: None,
                        sampled_token: Some(sampled_token),
                        execution_time_ms,
                    })
                    .collect(),
            ));
        }

        warn!(
            provider = %expected_provider.as_str(),
            "batched provider sampling failed, falling back to per-session logits sampling"
        );

        Ok(Some(
            ProviderRuntimeCore::split_logits_rows(&logits_2d)?
                .into_iter()
                .zip(requests.iter().map(|request| request.session_id))
                .map(|(logits, session_id)| DecodeMicrobatchOutput {
                    session_id,
                    logits: Some(logits),
                    sampled_token: None,
                    execution_time_ms,
                })
                .collect(),
        ))
    }
}

#[async_trait]
impl ExecutionBackend for ProviderExecutionBackend {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn provider_kind(&self) -> ExecutionProviderKind {
        self.core().provider_kind()
    }

    fn optimization_profile(&self) -> BackendOptimizationProfile {
        self.core().optimization_profile()
    }

    fn executor_contract(&self) -> &LocalExecutorContract {
        self.core().executor_contract()
    }

    async fn prefill(
        &mut self,
        tokens: &[u32],
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
        workspace: Option<&mut PrefillWorkspaceLease>,
    ) -> Result<BackendLogits> {
        self.core_mut()
            .prefill(tokens, worker_ring, job_id, workspace)
            .await
    }

    async fn decode_step(
        &mut self,
        token: u32,
        worker_ring: &mut WorkerRing<'_>,
        job_id: Uuid,
    ) -> Result<BackendLogits> {
        self.core_mut()
            .decode_step(token, worker_ring, job_id)
            .await
    }

    fn sample(
        &self,
        logits: &BackendLogits,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<u32> {
        self.core().sample(logits, temperature, top_p, seed)
    }

    fn cache_seq_len(&self) -> usize {
        self.core().cache_seq_len()
    }

    fn live_kv_cache_bytes(&self) -> usize {
        self.core().live_kv_cache_bytes()
    }

    fn logical_kv_tokens(&self) -> usize {
        self.core().logical_kv_tokens()
    }

    fn sequence_position(&self) -> usize {
        self.core().sequence_position()
    }

    fn last_allreduce_metrics(&self) -> RingAllReduceMetrics {
        self.core().last_allreduce_metrics()
    }

    fn export_kv_cache(&self, max_cached_tokens: Option<usize>) -> Result<Option<KVCacheSnapshot>> {
        self.core().export_kv_cache(max_cached_tokens)
    }

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        self.core_mut().import_kv_cache(snapshot)
    }

    fn clear(&mut self) {
        self.core_mut().clear();
    }

    fn fast_path_context(&self) -> FastPathBackendContext {
        self.core().fast_path_context()
    }
}

#[async_trait]
impl ExecutionBackend for ProviderRuntimeCore {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn provider_kind(&self) -> ExecutionProviderKind {
        self.provider
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
        workspace: Option<&mut PrefillWorkspaceLease>,
    ) -> Result<BackendLogits> {
        self.forward_pass
            .prefill(tokens, worker_ring, job_id, workspace)
            .await
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

    fn sample(
        &self,
        logits: &BackendLogits,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<u32> {
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

    fn export_kv_cache(&self, max_cached_tokens: Option<usize>) -> Result<Option<KVCacheSnapshot>> {
        self.forward_pass
            .export_kv_cache_snapshot(max_cached_tokens)
    }

    fn import_kv_cache(&mut self, snapshot: &KVCacheSnapshot) -> Result<()> {
        self.forward_pass.import_kv_cache_snapshot(snapshot)
    }

    fn clear(&mut self) {
        self.forward_pass.clear_cache();
    }

    fn fast_path_context(&self) -> FastPathBackendContext {
        FastPathBackendContext {
            provider: self.provider,
            optimization_profile: self.optimization_profile(),
            model_id: Some(self.model_id.clone()),
            logical_kv_tokens: self.logical_kv_tokens(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ProfileBackend {
        provider: ExecutionProviderKind,
        contract: LocalExecutorContract,
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
            self.contract.optimization_profile
        }

        fn executor_contract(&self) -> &LocalExecutorContract {
            &self.contract
        }

        async fn prefill(
            &mut self,
            _tokens: &[u32],
            _worker_ring: &mut WorkerRing<'_>,
            _job_id: Uuid,
            _workspace: Option<&mut PrefillWorkspaceLease>,
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
        ) -> Result<u32> {
            Ok(0)
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

        fn export_kv_cache(
            &self,
            _max_cached_tokens: Option<usize>,
        ) -> Result<Option<KVCacheSnapshot>> {
            Ok(None)
        }

        fn import_kv_cache(&mut self, _snapshot: &KVCacheSnapshot) -> Result<()> {
            Ok(())
        }

        fn clear(&mut self) {}

        fn fast_path_context(&self) -> FastPathBackendContext {
            FastPathBackendContext {
                provider: self.provider,
                optimization_profile: self.optimization_profile(),
                model_id: Some(format!("test-{}", self.provider.as_str())),
                logical_kv_tokens: 0,
            }
        }
    }

    #[test]
    fn backend_optimization_profile_tracks_provider_kind() {
        assert_eq!(
            ProfileBackend {
                provider: ExecutionProviderKind::Cpu,
                contract: LocalExecutorContract::for_provider(ExecutionProviderKind::Cpu),
            }
            .optimization_profile(),
            BackendOptimizationProfile::CpuSerial
        );
        assert_eq!(
            ProfileBackend {
                provider: ExecutionProviderKind::Metal,
                contract: LocalExecutorContract::for_provider(ExecutionProviderKind::Metal),
            }
            .optimization_profile(),
            BackendOptimizationProfile::MetalVectorized
        );
        assert_eq!(
            ProfileBackend {
                provider: ExecutionProviderKind::Cuda,
                contract: LocalExecutorContract::for_provider(ExecutionProviderKind::Cuda),
            }
            .optimization_profile(),
            BackendOptimizationProfile::CudaFused
        );
    }

    #[test]
    fn backend_executor_contract_assigns_decode_microbatch_to_all_production_backends() {
        let cpu = LocalExecutorContract::for_provider(ExecutionProviderKind::Cpu);
        let metal = LocalExecutorContract::for_provider(ExecutionProviderKind::Metal);
        let cuda = LocalExecutorContract::for_provider(ExecutionProviderKind::Cuda);

        assert!(cpu.is_fast_path());
        assert!(cpu.supports_decode_microbatch());
        assert!(metal.is_fast_path());
        assert!(metal.supports_decode_microbatch());
        assert!(cuda.is_fast_path());
        assert!(cuda.supports_decode_microbatch());
    }

    #[test]
    fn accelerated_decode_batch_requires_homogeneous_fast_path_contracts() {
        let mut metal_a = ProfileBackend {
            provider: ExecutionProviderKind::Metal,
            contract: LocalExecutorContract::for_provider(ExecutionProviderKind::Metal),
        };
        let mut metal_b = ProfileBackend {
            provider: ExecutionProviderKind::Metal,
            contract: LocalExecutorContract::for_provider(ExecutionProviderKind::Metal),
        };
        let mut cpu = ProfileBackend {
            provider: ExecutionProviderKind::Cpu,
            contract: LocalExecutorContract::for_provider(ExecutionProviderKind::Cpu),
        };

        let fast_requests = vec![
            DecodeMicrobatchRequest {
                session_id: Uuid::nil(),
                job_id: Uuid::nil(),
                token: 1,
                temperature: 0.7,
                top_p: 0.9,
                sampling_seed: 1,
                backend: &mut metal_a,
            },
            DecodeMicrobatchRequest {
                session_id: Uuid::new_v4(),
                job_id: Uuid::new_v4(),
                token: 2,
                temperature: 0.7,
                top_p: 0.9,
                sampling_seed: 2,
                backend: &mut metal_b,
            },
        ];
        assert!(BackendMicrobatchExecutor::can_use_accelerated_decode_batch(
            &fast_requests
        ));

        let mixed_requests = vec![
            DecodeMicrobatchRequest {
                session_id: Uuid::nil(),
                job_id: Uuid::nil(),
                token: 1,
                temperature: 0.7,
                top_p: 0.9,
                sampling_seed: 1,
                backend: &mut metal_a,
            },
            DecodeMicrobatchRequest {
                session_id: Uuid::new_v4(),
                job_id: Uuid::new_v4(),
                token: 2,
                temperature: 0.7,
                top_p: 0.9,
                sampling_seed: 2,
                backend: &mut cpu,
            },
        ];
        assert!(!BackendMicrobatchExecutor::can_use_accelerated_decode_batch(&mixed_requests));
    }
}
