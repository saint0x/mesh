use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::{
    StageSendChunkMode, StageSendScratch, StagedCollectiveBuffer,
};
use crate::provider::{selected_execution_provider, ExecutionProviderKind};
use crate::wire_f32::{accumulate_into_f32_slice, copy_into_f32_slice, decode_into_f32_scratch};
#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::cuda_backend::{cudarc::driver::PinnedHostSlice, CudaStorage};
#[cfg(all(target_os = "linux", feature = "cuda"))]
use candle_core::Storage;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use candle_core::{backend::BackendStorage, MetalStorage, Storage};
use candle_core::{DType, Device, Tensor as CandleTensor, D};
use candle_nn::ops as candle_ops;
use std::collections::HashMap;
use std::ops::Range;
use std::slice;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::stats::{
    record_runtime_collective_host_restore, record_runtime_collective_host_stage,
    record_runtime_device_sampling,
};
use super::tensor_ops::{Tensor1D, Tensor2D};

pub(crate) type DeviceTensor = CandleTensor;
pub(crate) type DeviceDType = DType;
pub(crate) type RuntimeDevice = Device;
pub(crate) type RuntimeError = candle_core::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RopeFrequencyKey {
    head_dim: usize,
    base_bits: u32,
}

fn rope_inverse_frequency(head_dim: usize, half_dim: usize, base: f32) -> Arc<[f32]> {
    static CACHE: OnceLock<Mutex<HashMap<RopeFrequencyKey, Arc<[f32]>>>> = OnceLock::new();
    let key = RopeFrequencyKey {
        head_dim,
        base_bits: base.to_bits(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .expect("rope inverse frequency cache mutex poisoned");
    guard
        .entry(key)
        .or_insert_with(|| {
            (0..half_dim)
                .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
                .collect::<Vec<_>>()
                .into()
        })
        .clone()
}

fn rope_positions_tensor(
    positions: &[u32],
    rows: usize,
    device: &RuntimeDevice,
) -> Result<DeviceTensor> {
    let contiguous = positions
        .first()
        .copied()
        .map(|start| {
            positions
                .iter()
                .enumerate()
                .all(|(offset, position)| *position == start.saturating_add(offset as u32))
        })
        .unwrap_or(true);

    if contiguous {
        let start = positions.first().copied().unwrap_or_default() as f32;
        let end = start + rows as f32;
        return DeviceTensor::arange(start, end, device)
            .and_then(|tensor| tensor.reshape((rows, 1)))
            .map_err(runtime_error);
    }

    let pos = positions.iter().map(|p| *p as f32).collect::<Vec<_>>();
    DeviceTensor::from_slice(&pos, (rows, 1), device).map_err(runtime_error)
}

pub(crate) fn runtime_error(err: RuntimeError) -> AgentError {
    AgentError::Execution(format!("Tensor backend error: {}", err))
}

pub(crate) fn probe_provider(provider: ExecutionProviderKind) -> (bool, Option<String>) {
    match provider {
        ExecutionProviderKind::Cpu => (true, None),
        ExecutionProviderKind::Metal => probe_metal_provider(),
        ExecutionProviderKind::Cuda => probe_cuda_provider(),
        ExecutionProviderKind::Rocm => probe_rocm_provider(),
    }
}

fn probe_metal_provider() -> (bool, Option<String>) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        match Device::new_metal(0) {
            Ok(_) => (true, None),
            Err(err) => (false, Some(format!("metal runtime probe failed: {}", err))),
        }
    }

    #[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
    {
        (
            false,
            Some("metal provider requires Apple Silicon for production support".to_string()),
        )
    }

    #[cfg(not(target_os = "macos"))]
    {
        (
            false,
            Some("metal provider is only available on macOS".to_string()),
        )
    }
}

fn probe_cuda_provider() -> (bool, Option<String>) {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        match Device::new_cuda(0) {
            Ok(_) => (true, None),
            Err(err) => (false, Some(format!("cuda runtime probe failed: {}", err))),
        }
    }

    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    {
        (
            false,
            Some("cuda provider was not compiled into this Mesh agent; rebuild with `--features cuda` on a CUDA host".to_string()),
        )
    }

    #[cfg(not(target_os = "linux"))]
    {
        (
            false,
            Some("cuda provider is only available on Linux builds".to_string()),
        )
    }
}

fn probe_rocm_provider() -> (bool, Option<String>) {
    #[cfg(target_os = "linux")]
    {
        if !host_has_rocm_runtime() {
            return (
                false,
                Some("rocm runtime probe failed: rocminfo, rocm-smi, and HIP runtime markers were not found".to_string()),
            );
        }
        (
            false,
            Some(
                "rocm runtime detected, but this Mesh build does not link a native HIP/ROCm tensor backend; refusing CPU fallback"
                    .to_string(),
            ),
        )
    }

    #[cfg(not(target_os = "linux"))]
    {
        (
            false,
            Some("rocm provider is only available on Linux builds".to_string()),
        )
    }
}

#[cfg(target_os = "linux")]
fn host_has_rocm_runtime() -> bool {
    std::process::Command::new("rocminfo")
        .arg("--help")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
        || std::process::Command::new("rocm-smi")
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
        || std::path::Path::new("/opt/rocm").exists()
        || std::path::Path::new("/dev/kfd").exists()
}

pub(crate) fn execution_device() -> Result<&'static RuntimeDevice> {
    static DEVICE: OnceLock<std::result::Result<RuntimeDevice, String>> = OnceLock::new();
    match DEVICE.get_or_init(|| init_execution_device().map_err(|e| e.to_string())) {
        Ok(device) => Ok(device),
        Err(err) => Err(AgentError::Execution(format!(
            "Execution backend unavailable: {}",
            err
        ))),
    }
}

fn init_execution_device() -> std::result::Result<RuntimeDevice, RuntimeError> {
    let provider = selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu);
    match provider {
        ExecutionProviderKind::Cpu => Ok(RuntimeDevice::Cpu),
        ExecutionProviderKind::Cuda => {
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            {
                RuntimeDevice::new_cuda(0)
            }
            #[cfg(all(target_os = "linux", not(feature = "cuda")))]
            {
                Err(RuntimeError::Msg(
                    "cuda provider was not compiled into this Mesh agent; rebuild with `--features cuda` on a CUDA host"
                        .to_string(),
                ))
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err(RuntimeError::Msg(
                    "cuda provider is unavailable on this platform".to_string(),
                ))
            }
        }
        ExecutionProviderKind::Rocm => Err(RuntimeError::Msg(
            "rocm provider requires a native HIP/ROCm tensor backend; this Mesh build refuses CPU fallback"
                .to_string(),
        )),
        ExecutionProviderKind::Metal => {
            #[cfg(target_os = "macos")]
            {
                RuntimeDevice::new_metal(0)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(RuntimeError::Msg(
                    "metal provider is unavailable on this platform".to_string(),
                ))
            }
        }
    }
}

pub(crate) fn device_tensor_from_2d(tensor: &Tensor2D) -> Result<DeviceTensor> {
    DeviceTensor::from_vec(
        tensor.data.clone(),
        (tensor.rows, tensor.cols),
        execution_device()?,
    )
    .map_err(runtime_error)
}

pub(crate) fn device_tensor_from_1d(tensor: &Tensor1D) -> Result<DeviceTensor> {
    DeviceTensor::from_vec(tensor.data.clone(), tensor.len(), execution_device()?)
        .map_err(runtime_error)
}

pub(crate) fn host_tensor_2d_from_device(tensor: &DeviceTensor) -> Result<Tensor2D> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Expected 2D device tensor, got shape {:?}",
            dims
        )));
    }
    let data = tensor
        .flatten_all()
        .map_err(runtime_error)?
        .to_dtype(DeviceDType::F32)
        .map_err(runtime_error)?
        .to_vec1::<f32>()
        .map_err(runtime_error)?;
    Tensor2D::new(data, dims[0], dims[1])
}

fn apply_top_p_device(sorted_probs: &DeviceTensor, top_p: f32) -> Result<DeviceTensor> {
    let dims = sorted_probs.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Device top-p sampling expects rank-2 probabilities, got {:?}",
            dims
        )));
    }

    let cumulative = sorted_probs.cumsum(D::Minus1).map_err(runtime_error)?;
    let shifted = cumulative
        .broadcast_sub(&sorted_probs)
        .map_err(runtime_error)?;
    let threshold =
        DeviceTensor::full(top_p, dims, sorted_probs.device()).map_err(runtime_error)?;
    let keep_sorted = shifted.lt(&threshold).map_err(runtime_error)?;
    keep_sorted
        .where_cond(
            &sorted_probs,
            &sorted_probs.zeros_like().map_err(runtime_error)?,
        )
        .map_err(runtime_error)
}

fn deterministic_sample_threshold(seed: u64) -> f32 {
    const MIX_A: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_B: u64 = 0xBF58_476D_1CE4_E5B9;
    const MIX_C: u64 = 0x94D0_49BB_1331_11EB;

    let mut x = seed.wrapping_add(MIX_A);
    x = (x ^ (x >> 30)).wrapping_mul(MIX_B);
    x = (x ^ (x >> 27)).wrapping_mul(MIX_C);
    x ^= x >> 31;

    let upper = (x >> 40) as u32;
    let threshold = (upper as f64) / ((1u64 << 24) as f64);
    threshold.clamp(0.0, 1.0 - f64::EPSILON) as f32
}

fn deterministic_sample_thresholds_for_seeds(rng_seeds: &[u64]) -> Vec<f32> {
    rng_seeds
        .iter()
        .map(|seed| deterministic_sample_threshold(*seed))
        .collect()
}

fn sample_indices_from_cdf(
    cdf: &DeviceTensor,
    thresholds: &[f32],
    sorted_indices: Option<&DeviceTensor>,
) -> Result<Vec<u32>> {
    let dims = cdf.dims();
    let device = cdf.device().clone();
    let threshold = DeviceTensor::from_vec(thresholds.to_vec(), (dims[0], 1), &device)
        .map_err(runtime_error)?
        .broadcast_as((dims[0], dims[1]))
        .map_err(runtime_error)?;
    let crossing = cdf
        .ge(&threshold)
        .map_err(runtime_error)?
        .to_dtype(DeviceDType::U32)
        .map_err(runtime_error)?;
    let sampled = crossing.argmax(1).map_err(runtime_error)?;
    match sorted_indices {
        Some(indices) => indices
            .gather(&sampled.unsqueeze(1).map_err(runtime_error)?, 1)
            .and_then(|ids| ids.squeeze(1))
            .and_then(|ids| ids.to_vec1::<u32>())
            .map_err(runtime_error),
        None => sampled.to_vec1::<u32>().map_err(runtime_error),
    }
}

pub(crate) fn sample_tokens_device_with_seeds(
    logits: &DeviceTensor,
    temperature: f32,
    top_p: f32,
    rng_seeds: &[u64],
) -> Result<Vec<u32>> {
    let started = Instant::now();
    let dims = logits.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Device sampling expects rank-2 logits, got {:?}",
            dims
        )));
    }
    if dims[0] == 0 || dims[1] == 0 {
        return Err(AgentError::Execution(
            "Device sampling received an empty logits tensor".to_string(),
        ));
    }
    if dims[0] != rng_seeds.len() {
        return Err(AgentError::Execution(format!(
            "Device sampling received {} rows but {} seeds",
            dims[0],
            rng_seeds.len()
        )));
    }

    if temperature <= 0.0 || top_p <= 0.0 {
        let result = logits
            .argmax(1)
            .and_then(|idx| idx.to_vec1::<u32>())
            .map_err(runtime_error);
        if result.is_ok() {
            record_runtime_device_sampling(dims[0] as u64, started.elapsed().as_millis() as u64);
        }
        return result;
    }

    let logits = logits.to_dtype(DeviceDType::F32).map_err(runtime_error)?;
    let scaled_logits = if temperature == 1.0 {
        logits
    } else {
        logits
            .affine((1.0 / temperature) as f64, 0.0)
            .map_err(runtime_error)?
    };
    let probs = candle_ops::softmax(&scaled_logits, 1).map_err(runtime_error)?;
    let thresholds = deterministic_sample_thresholds_for_seeds(rng_seeds);
    let result = if top_p >= 1.0 {
        let cdf = probs.cumsum(D::Minus1).map_err(runtime_error)?;
        sample_indices_from_cdf(&cdf, &thresholds, None)
    } else {
        let (sorted_probs, sorted_indices) = probs.sort_last_dim(false).map_err(runtime_error)?;
        let filtered_sorted_probs = apply_top_p_device(&sorted_probs, top_p)?;
        let denom = filtered_sorted_probs
            .sum_keepdim(1)
            .map_err(runtime_error)?;
        let renormalized = filtered_sorted_probs
            .broadcast_mul(&denom.recip().map_err(runtime_error)?)
            .map_err(runtime_error)?;
        let cdf = renormalized.cumsum(D::Minus1).map_err(runtime_error)?;
        sample_indices_from_cdf(&cdf, &thresholds, Some(&sorted_indices))
    };
    if result.is_ok() {
        record_runtime_device_sampling(dims[0] as u64, started.elapsed().as_millis() as u64);
    }
    result
}

pub(crate) fn sample_tokens_device(
    logits: &DeviceTensor,
    temperature: f32,
    top_p: f32,
    rng_seed: u64,
) -> Result<Vec<u32>> {
    let dims = logits.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Device sampling expects rank-2 logits, got {:?}",
            dims
        )));
    }
    let seeds = (0..dims[0])
        .map(|idx| rng_seed.wrapping_add(idx as u64))
        .collect::<Vec<_>>();
    sample_tokens_device_with_seeds(logits, temperature, top_p, &seeds)
}

pub(crate) fn sample_token_device(
    logits: &DeviceTensor,
    temperature: f32,
    top_p: f32,
    rng_seed: u64,
) -> Result<u32> {
    sample_tokens_device(logits, temperature, top_p, rng_seed)?
        .into_iter()
        .next()
        .ok_or_else(|| AgentError::Execution("Device sampling returned no token ids".to_string()))
}

pub(crate) struct DeviceCollectiveBuffer {
    flat: DeviceTensor,
    rows: usize,
    cols: usize,
    receive_decode_scratch: Vec<f32>,
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    cuda_receive_pinned: Option<PinnedHostSlice<f32>>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    shared_metal: Option<MetalStorage>,
}

impl DeviceCollectiveBuffer {
    pub(crate) fn from_device_tensor(tensor: &DeviceTensor) -> Result<Self> {
        let dims = tensor.dims();
        if dims.len() != 2 {
            return Err(AgentError::Execution(format!(
                "Expected 2D device tensor for staged collective execution, got shape {:?}",
                dims
            )));
        }

        let flat = tensor
            .flatten_all()
            .map_err(runtime_error)?
            .to_dtype(DeviceDType::F32)
            .map_err(runtime_error)?
            .contiguous()
            .map_err(runtime_error)?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let shared_metal = shared_metal_collective_storage(&flat);

        Ok(Self {
            flat,
            rows: dims[0],
            cols: dims[1],
            receive_decode_scratch: Vec::new(),
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            cuda_receive_pinned: None,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            shared_metal,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[cfg(test)]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    #[cfg(test)]
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    fn stage_send_chunk_impl(
        &mut self,
        range: Range<usize>,
        scratch: &mut StageSendScratch,
    ) -> Result<StageSendChunkMode> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(storage) = &self.shared_metal {
            let scratch = scratch.ensure_vec(range.len());
            let len = self.len();
            let staged = unsafe {
                &slice::from_raw_parts(storage.buffer().contents() as *const f32, len)[range]
            };
            scratch.extend_from_slice(staged);
            return Ok(StageSendChunkMode::SharedVisibleScratch);
        }

        if let Some(stage_mode) =
            stage_send_chunk_from_dense_cuda_tensor(&self.flat, range.clone(), scratch)?
        {
            record_runtime_collective_host_stage(
                (range.len().saturating_mul(std::mem::size_of::<f32>())) as u64,
            );
            return Ok(stage_mode);
        }

        let staged = self
            .flat
            .narrow(0, range.start, range.len())
            .map_err(runtime_error)?
            .to_vec1::<f32>()
            .map_err(runtime_error)?;
        record_runtime_collective_host_stage(
            (range.len().saturating_mul(std::mem::size_of::<f32>())) as u64,
        );
        let scratch = scratch.ensure_vec(range.len());
        scratch.extend(staged);
        Ok(StageSendChunkMode::HostMaterializedScratch)
    }

    fn accumulate_range_from_wire_bytes_impl(
        &mut self,
        range: Range<usize>,
        payload_bytes: &[u8],
    ) -> Result<()> {
        if payload_bytes.len() != range.len().saturating_mul(std::mem::size_of::<f32>()) {
            return Err(AgentError::Execution(format!(
                "Wire payload byte length {} did not match expected byte length {}",
                payload_bytes.len(),
                range.len().saturating_mul(std::mem::size_of::<f32>())
            )));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(storage) = &self.shared_metal {
            let len = self.len();
            let dst = unsafe {
                &mut slice::from_raw_parts_mut(storage.buffer().contents() as *mut f32, len)[range]
            };
            accumulate_into_f32_slice(dst, payload_bytes);
            return Ok(());
        }
        let update = self.device_tensor_from_wire_bytes(range.len(), payload_bytes)?;
        let current = self
            .flat
            .narrow(0, range.start, range.len())
            .map_err(runtime_error)?;
        let accumulated = current.broadcast_add(&update).map_err(runtime_error)?;
        self.flat
            .slice_set(&accumulated, 0, range.start)
            .map_err(runtime_error)
    }

    fn copy_range_from_wire_bytes_impl(
        &mut self,
        range: Range<usize>,
        payload_bytes: &[u8],
    ) -> Result<()> {
        if payload_bytes.len() != range.len().saturating_mul(std::mem::size_of::<f32>()) {
            return Err(AgentError::Execution(format!(
                "Wire payload byte length {} did not match expected byte length {}",
                payload_bytes.len(),
                range.len().saturating_mul(std::mem::size_of::<f32>())
            )));
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(storage) = &self.shared_metal {
            let len = self.len();
            let dst = unsafe {
                &mut slice::from_raw_parts_mut(storage.buffer().contents() as *mut f32, len)[range]
            };
            copy_into_f32_slice(dst, payload_bytes);
            return Ok(());
        }
        let update = self.device_tensor_from_wire_bytes(range.len(), payload_bytes)?;
        self.flat
            .slice_set(&update, 0, range.start)
            .map_err(runtime_error)
    }

    fn device_tensor_from_wire_bytes(
        &mut self,
        expected_len: usize,
        payload_bytes: &[u8],
    ) -> Result<DeviceTensor> {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        if let Some(tensor) =
            self.cuda_upload_tensor_from_wire_bytes(expected_len, payload_bytes)?
        {
            record_runtime_collective_host_restore(payload_bytes.len() as u64);
            return Ok(tensor);
        }

        let payload = decode_into_f32_scratch(
            expected_len,
            payload_bytes,
            &mut self.receive_decode_scratch,
        )?;
        let update = DeviceTensor::from_slice(payload, payload.len(), self.flat.device())
            .map_err(runtime_error)?;
        record_runtime_collective_host_restore(payload_bytes.len() as u64);
        Ok(update)
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn cuda_upload_tensor_from_wire_bytes(
        &mut self,
        expected_len: usize,
        payload_bytes: &[u8],
    ) -> Result<Option<DeviceTensor>> {
        let (storage, _) = self.flat.storage_and_layout();
        let Storage::Cuda(cuda_storage) = &*storage else {
            return Ok(None);
        };
        let pinned =
            self.ensure_cuda_receive_pinned(&cuda_storage.device().cuda_stream(), expected_len)?;
        copy_wire_f32_bytes_into_slice(
            &mut pinned.as_mut_slice().map_err(runtime_error)?[..expected_len],
            payload_bytes,
        );
        let upload = cuda_storage
            .device()
            .clone_htod(pinned)
            .map_err(runtime_error)?;
        let tensor = DeviceTensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(
                upload,
                cuda_storage.device().clone(),
            )),
            payload.len(),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(tensor))
    }

    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn ensure_cuda_receive_pinned(
        &mut self,
        stream: &std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
        len: usize,
    ) -> Result<&mut PinnedHostSlice<f32>> {
        let needs_realloc = match &self.cuda_receive_pinned {
            Some(buffer) => buffer.len() != len,
            None => true,
        };
        if needs_realloc {
            let pinned = unsafe { stream.context().alloc_pinned::<f32>(len) }.map_err(|err| {
                AgentError::Execution(format!("CUDA pinned receive host allocation failed: {err}"))
            })?;
            self.cuda_receive_pinned = Some(pinned);
        }
        self.cuda_receive_pinned.as_mut().ok_or_else(|| {
            AgentError::Execution("CUDA pinned receive host buffer missing".to_string())
        })
    }

    pub(crate) fn into_device_tensor_like(self, template: &DeviceTensor) -> Result<DeviceTensor> {
        let reshaped = self
            .flat
            .reshape((self.rows, self.cols))
            .map_err(runtime_error)?;
        if reshaped.device().same_device(template.device()) {
            return Ok(reshaped);
        }
        reshaped.to_device(template.device()).map_err(runtime_error)
    }
}

impl StagedCollectiveBuffer for DeviceCollectiveBuffer {
    fn len(&self) -> usize {
        DeviceCollectiveBuffer::len(self)
    }

    fn stage_send_chunk(
        &mut self,
        range: Range<usize>,
        scratch: &mut StageSendScratch,
    ) -> Result<StageSendChunkMode> {
        self.stage_send_chunk_impl(range, scratch)
    }

    fn accumulate_recv_chunk(&mut self, range: Range<usize>, payload: &[u8]) -> Result<()> {
        self.accumulate_range_from_wire_bytes_impl(range, payload)
    }

    fn copy_recv_chunk(&mut self, range: Range<usize>, payload: &[u8]) -> Result<()> {
        self.copy_range_from_wire_bytes_impl(range, payload)
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
fn stage_send_chunk_from_dense_cuda_tensor(
    tensor: &DeviceTensor,
    range: Range<usize>,
    scratch: &mut StageSendScratch,
) -> Result<Option<StageSendChunkMode>> {
    let elem_count = tensor.elem_count();
    let (storage, layout) = tensor.storage_and_layout();
    if let Storage::Cuda(cuda_storage) = &*storage {
        if let Some((start, end)) = layout.contiguous_offsets() {
            let total_len = end.saturating_sub(start);
            if start == 0 && total_len == elem_count {
                let src = cuda_storage.as_cuda_slice::<f32>().map_err(runtime_error)?;
                let src = src.slice(range.start..range.end);
                let pinned = scratch
                    .ensure_cuda_pinned(&cuda_storage.device().cuda_stream(), range.len())?;
                cuda_storage
                    .device()
                    .memcpy_dtoh(&src, pinned)
                    .map_err(runtime_error)?;
                return Ok(Some(StageSendChunkMode::HostMaterializedScratch));
            }
        }
    }
    Ok(None)
}

#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn stage_send_chunk_from_dense_cuda_tensor(
    _tensor: &DeviceTensor,
    _range: Range<usize>,
    _scratch: &mut StageSendScratch,
) -> Result<Option<StageSendChunkMode>> {
    Ok(None)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn shared_metal_collective_storage(tensor: &DeviceTensor) -> Option<MetalStorage> {
    let elem_count = tensor.elem_count();
    let (storage, layout) = tensor.storage_and_layout();
    if let Storage::Metal(metal_storage) = &*storage {
        if let Some((start, end)) = layout.contiguous_offsets() {
            let total_len = end.saturating_sub(start);
            if start == 0 && total_len == elem_count && metal_storage.dtype() == DType::F32 {
                return Some(metal_storage.clone());
            }
        }
    }
    None
}

pub(crate) fn rms_norm_device(
    tensor: &DeviceTensor,
    gamma: &DeviceTensor,
    eps: f32,
) -> Result<DeviceTensor> {
    candle_ops::rms_norm(tensor, gamma, eps).map_err(runtime_error)
}

pub(crate) fn silu_device(tensor: &DeviceTensor) -> Result<DeviceTensor> {
    candle_ops::silu(tensor).map_err(runtime_error)
}

pub(crate) fn softmax_device(tensor: &DeviceTensor, dim: usize) -> Result<DeviceTensor> {
    candle_ops::softmax(tensor, dim).map_err(runtime_error)
}

pub(crate) fn apply_rope_device(
    tensor: &DeviceTensor,
    rows: usize,
    cols: usize,
    positions: &[u32],
    head_dim: usize,
    base: f32,
) -> Result<DeviceTensor> {
    if rows != positions.len() {
        return Err(AgentError::Execution(format!(
            "RoPE position count {} doesn't match sequence length {}",
            positions.len(),
            rows
        )));
    }
    if head_dim % 2 != 0 {
        return Err(AgentError::Execution(format!(
            "RoPE requires even head_dim, got {}",
            head_dim
        )));
    }
    if cols % head_dim != 0 {
        return Err(AgentError::Execution(format!(
            "RoPE head_dim {} does not divide tensor width {}",
            head_dim, cols
        )));
    }

    let num_heads = cols / head_dim;
    let half_dim = head_dim / 2;
    let x = tensor
        .reshape((rows, num_heads, head_dim))
        .map_err(runtime_error)?;
    let x1 = x.narrow(2, 0, half_dim).map_err(runtime_error)?;
    let x2 = x.narrow(2, half_dim, half_dim).map_err(runtime_error)?;

    let device = execution_device()?;
    let inv_freq = rope_inverse_frequency(head_dim, half_dim, base);
    let pos = rope_positions_tensor(positions, rows, device)?;
    let inv = DeviceTensor::from_slice(&inv_freq, (1, half_dim), device).map_err(runtime_error)?;
    let freqs = pos.broadcast_matmul(&inv).map_err(runtime_error)?;
    let cos = freqs
        .cos()
        .map_err(runtime_error)?
        .unsqueeze(1)
        .map_err(runtime_error)?
        .expand((rows, num_heads, half_dim))
        .map_err(runtime_error)?;
    let sin = freqs
        .sin()
        .map_err(runtime_error)?
        .unsqueeze(1)
        .map_err(runtime_error)?
        .expand((rows, num_heads, half_dim))
        .map_err(runtime_error)?;

    let rot1 = x1
        .broadcast_mul(&cos)
        .map_err(runtime_error)?
        .broadcast_sub(&x2.broadcast_mul(&sin).map_err(runtime_error)?)
        .map_err(runtime_error)?;
    let rot2 = x1
        .broadcast_mul(&sin)
        .map_err(runtime_error)?
        .broadcast_add(&x2.broadcast_mul(&cos).map_err(runtime_error)?)
        .map_err(runtime_error)?;
    DeviceTensor::cat(&[&rot1, &rot2], 2)
        .map_err(runtime_error)?
        .reshape((rows, cols))
        .map_err(runtime_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_collective_buffer_updates_device_resident_tensor() {
        let template =
            device_tensor_from_2d(&Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap()).unwrap();
        let mut buffer = DeviceCollectiveBuffer::from_device_tensor(&template).unwrap();
        let mut send_scratch = StageSendScratch::default();

        let _ = buffer
            .stage_send_chunk_impl(0..2, &mut send_scratch)
            .unwrap();
        let first_row = send_scratch.as_slice(2).unwrap().to_vec();
        assert_eq!(first_row, vec![1.0, 2.0]);

        let accumulate = wire_bytes(&[10.0f32, 20.0f32]);
        buffer
            .accumulate_range_from_wire_bytes_impl(0..2, &accumulate)
            .unwrap();

        let replace = wire_bytes(&[7.0f32, 8.0f32]);
        buffer
            .copy_range_from_wire_bytes_impl(2..4, &replace)
            .unwrap();

        let restored =
            host_tensor_2d_from_device(&buffer.into_device_tensor_like(&template).unwrap())
                .unwrap();
        assert_eq!(restored.data, vec![11.0, 22.0, 7.0, 8.0]);
    }

    fn wire_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        bytes
    }
}
