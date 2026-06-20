use crate::errors::{AgentError, Result};
use crate::executor::ring_allreduce::CollectiveMatrix;
use crate::provider::{selected_execution_provider, ExecutionProviderKind};
#[cfg(target_os = "linux")]
use candle_core::Storage;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use candle_core::{CustomOp1, Layout, MetalStorage, Result as CandleResult, Shape};
use candle_core::{DType, Device, Tensor as CandleTensor, D};
use candle_nn::ops as candle_ops;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

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
    #[cfg(target_os = "linux")]
    {
        match Device::new_cuda(0) {
            Ok(_) => (true, None),
            Err(err) => (false, Some(format!("cuda runtime probe failed: {}", err))),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        (
            false,
            Some("cuda provider is only available on Linux builds".to_string()),
        )
    }
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
            #[cfg(target_os = "linux")]
            {
                RuntimeDevice::new_cuda(0)
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err(RuntimeError::Msg(
                    "cuda provider is unavailable on this platform".to_string(),
                ))
            }
        }
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

pub(crate) fn add_tensor2d(lhs: &Tensor2D, rhs: &Tensor2D) -> Result<Tensor2D> {
    let lhs = device_tensor_from_2d(lhs)?;
    let rhs = device_tensor_from_2d(rhs)?;
    host_tensor_2d_from_device(&lhs.broadcast_add(&rhs).map_err(runtime_error)?)
}

pub(crate) fn mul_tensor2d(lhs: &Tensor2D, rhs: &Tensor2D) -> Result<Tensor2D> {
    let lhs = device_tensor_from_2d(lhs)?;
    let rhs = device_tensor_from_2d(rhs)?;
    host_tensor_2d_from_device(&lhs.broadcast_mul(&rhs).map_err(runtime_error)?)
}

pub(crate) fn scale_tensor2d(tensor: &Tensor2D, scalar: f32) -> Result<Tensor2D> {
    let tensor = device_tensor_from_2d(tensor)?;
    host_tensor_2d_from_device(&tensor.affine(scalar as f64, 0.0).map_err(runtime_error)?)
}

pub(crate) fn transpose_tensor2d(tensor: &Tensor2D) -> Result<Tensor2D> {
    let tensor = device_tensor_from_2d(tensor)?;
    host_tensor_2d_from_device(&tensor.transpose(0, 1).map_err(runtime_error)?)
}

pub(crate) fn narrow_tensor2d_columns(
    tensor: &Tensor2D,
    col_start: usize,
    width: usize,
) -> Result<Tensor2D> {
    let tensor = device_tensor_from_2d(tensor)?;
    host_tensor_2d_from_device(&tensor.narrow(1, col_start, width).map_err(runtime_error)?)
}

pub(crate) fn matmul_tensor2d(lhs: &Tensor2D, rhs: &Tensor2D) -> Result<Tensor2D> {
    let lhs = device_tensor_from_2d(lhs)?;
    let rhs = device_tensor_from_2d(rhs)?;
    host_tensor_2d_from_device(&lhs.matmul(&rhs).map_err(runtime_error)?)
}

pub(crate) fn gelu_tensor2d(tensor: &Tensor2D) -> Result<Tensor2D> {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt() as f64;
    let x = device_tensor_from_2d(tensor)?;
    let x3 = x
        .sqr()
        .map_err(runtime_error)?
        .broadcast_mul(&x)
        .map_err(runtime_error)?;
    let inner = x
        .affine(0.044715, 0.0)
        .map_err(runtime_error)?
        .broadcast_mul(&x3)
        .map_err(runtime_error)?;
    let inner = x.broadcast_add(&inner).map_err(runtime_error)?;
    let inner = inner
        .affine(sqrt_2_over_pi, 0.0)
        .map_err(runtime_error)?
        .tanh()
        .map_err(runtime_error)?;
    let one_plus = inner.affine(1.0, 1.0).map_err(runtime_error)?;
    let scaled = x.affine(0.5, 0.0).map_err(runtime_error)?;
    host_tensor_2d_from_device(&scaled.broadcast_mul(&one_plus).map_err(runtime_error)?)
}

pub(crate) fn silu_tensor2d(tensor: &Tensor2D) -> Result<Tensor2D> {
    let tensor = device_tensor_from_2d(tensor)?;
    host_tensor_2d_from_device(&candle_ops::silu(&tensor).map_err(runtime_error)?)
}

pub(crate) fn rms_norm_tensor2d(tensor: &Tensor2D, gamma: &Tensor1D, eps: f32) -> Result<Tensor2D> {
    let x = device_tensor_from_2d(tensor)?;
    let weight = device_tensor_from_1d(gamma)?;
    host_tensor_2d_from_device(&candle_ops::rms_norm(&x, &weight, eps).map_err(runtime_error)?)
}

pub(crate) fn layer_norm_tensor2d(
    tensor: &Tensor2D,
    gamma: &Tensor1D,
    beta: &Tensor1D,
    eps: f32,
) -> Result<Tensor2D> {
    let x = device_tensor_from_2d(tensor)?;
    let gamma = device_tensor_from_1d(gamma)?;
    let beta = device_tensor_from_1d(beta)?;
    host_tensor_2d_from_device(
        &candle_ops::layer_norm(&x, &gamma, &beta, eps).map_err(runtime_error)?,
    )
}

pub(crate) fn softmax_tensor2d(tensor: &Tensor2D) -> Result<Tensor2D> {
    let tensor = device_tensor_from_2d(tensor)?;
    host_tensor_2d_from_device(&candle_ops::softmax(&tensor, 1).map_err(runtime_error)?)
}

pub(crate) fn apply_rope_tensor2d(
    tensor: &Tensor2D,
    positions: &[u32],
    head_dim: usize,
    base: f32,
) -> Result<Tensor2D> {
    let device_tensor = device_tensor_from_2d(tensor)?;
    host_tensor_2d_from_device(&apply_rope_device(
        &device_tensor,
        tensor.rows,
        tensor.cols,
        positions,
        head_dim,
        base,
    )?)
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

pub(crate) fn sample_tokens_device_with_seeds(
    logits: &DeviceTensor,
    temperature: f32,
    top_p: f32,
    rng_seeds: &[u64],
) -> Result<Vec<u32>> {
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
        return logits
            .argmax(1)
            .and_then(|idx| idx.to_vec1::<u32>())
            .map_err(runtime_error);
    }

    let logits = logits.to_dtype(DeviceDType::F32).map_err(runtime_error)?;
    let device = logits.device().clone();
    let scaled_logits = if temperature == 1.0 {
        logits
    } else {
        logits
            .affine((1.0 / temperature) as f64, 0.0)
            .map_err(runtime_error)?
    };
    let probs = candle_ops::softmax(&scaled_logits, 1).map_err(runtime_error)?;
    let (sorted_probs, sorted_indices) = probs.sort_last_dim(false).map_err(runtime_error)?;
    let filtered_sorted_probs = if top_p >= 1.0 {
        sorted_probs
    } else {
        apply_top_p_device(&sorted_probs, top_p)?
    };
    let denom = filtered_sorted_probs
        .sum_keepdim(1)
        .map_err(runtime_error)?;
    let renormalized = filtered_sorted_probs
        .broadcast_mul(&denom.recip().map_err(runtime_error)?)
        .map_err(runtime_error)?;
    let cdf = renormalized.cumsum(D::Minus1).map_err(runtime_error)?;
    let thresholds = deterministic_sample_thresholds_for_seeds(rng_seeds);
    let threshold = DeviceTensor::from_vec(thresholds, (dims[0], 1), &device)
        .map_err(runtime_error)?
        .broadcast_as((dims[0], dims[1]))
        .map_err(runtime_error)?;
    let crossing = cdf
        .ge(&threshold)
        .map_err(runtime_error)?
        .to_dtype(DeviceDType::U32)
        .map_err(runtime_error)?;
    let sampled_sorted = crossing.argmax(1).map_err(runtime_error)?;
    sorted_indices
        .gather(&sampled_sorted.unsqueeze(1).map_err(runtime_error)?, 1)
        .and_then(|ids| ids.squeeze(1))
        .and_then(|ids| ids.to_vec1::<u32>())
        .map_err(runtime_error)
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

pub(crate) fn embed_tokens_device(embedding_table: &Tensor2D, tokens: &[u32]) -> Result<Tensor2D> {
    let table = device_tensor_from_2d(embedding_table)?;
    let ids = DeviceTensor::from_slice(tokens, tokens.len(), execution_device()?)
        .map_err(runtime_error)?;
    host_tensor_2d_from_device(&table.embedding(&ids).map_err(runtime_error)?)
}

pub(crate) fn collective_matrix_from_device_tensor(
    tensor: &DeviceTensor,
) -> Result<CollectiveMatrix> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Expected 2D device tensor for collective buffer conversion, got shape {:?}",
            dims
        )));
    }

    let flattened = tensor
        .flatten_all()
        .map_err(runtime_error)?
        .to_dtype(DeviceDType::F32)
        .map_err(runtime_error)?
        .contiguous()
        .map_err(runtime_error)?;

    if let Some(matrix) = collective_matrix_from_dense_cuda_tensor(&flattened, dims[0], dims[1])? {
        return Ok(matrix);
    }

    let data = flattened.to_vec1::<f32>().map_err(runtime_error)?;
    Ok(CollectiveMatrix::new(data, dims[0], dims[1]))
}

#[cfg(target_os = "linux")]
fn collective_matrix_from_dense_cuda_tensor(
    tensor: &DeviceTensor,
    rows: usize,
    cols: usize,
) -> Result<Option<CollectiveMatrix>> {
    let elem_count = tensor.elem_count();
    let (storage, layout) = tensor.storage_and_layout();
    if let Storage::Cuda(cuda_storage) = &*storage {
        if let Some((start, end)) = layout.contiguous_offsets() {
            let total_len = end.saturating_sub(start);
            if start == 0 && total_len == elem_count {
                let src = cuda_storage.as_cuda_slice::<f32>().map_err(runtime_error)?;
                let device = cuda_storage.device();
                let mut matrix = CollectiveMatrix::from_pooled_host_buffer(rows, cols);
                device
                    .memcpy_dtoh(src, matrix.host_slice_mut())
                    .map_err(runtime_error)?;
                return Ok(Some(matrix));
            }
        }
    }
    drop(storage);
    Ok(None)
}

#[cfg(not(target_os = "linux"))]
fn collective_matrix_from_dense_cuda_tensor(
    _tensor: &DeviceTensor,
    _rows: usize,
    _cols: usize,
) -> Result<Option<CollectiveMatrix>> {
    Ok(None)
}

#[cfg(test)]
pub(crate) fn device_tensor_from_collective_matrix(
    tensor: &CollectiveMatrix,
) -> Result<DeviceTensor> {
    DeviceTensor::from_vec(
        tensor.to_host_vec(),
        (tensor.rows, tensor.cols),
        execution_device()?,
    )
    .map_err(runtime_error)
}

pub(crate) fn device_tensor_from_collective_owned_like(
    tensor: CollectiveMatrix,
    template: &DeviceTensor,
) -> Result<DeviceTensor> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(storage) = tensor.metal_shared_storage() {
        return restore_shared_metal_collective(storage, rows, cols, template)
            .map_err(runtime_error);
    }

    DeviceTensor::from_vec(tensor.into_host_vec(), (rows, cols), template.device())
        .map_err(runtime_error)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Clone)]
struct RestoreSharedMetalCollective {
    storage: MetalStorage,
    shape: Shape,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl CustomOp1 for RestoreSharedMetalCollective {
    fn name(&self) -> &'static str {
        "restore-shared-metal-collective"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> CandleResult<(candle_core::CpuStorage, Shape)> {
        Err(RuntimeError::Msg(
            "restore-shared-metal-collective requires a metal tensor".into(),
        ))
    }

    fn metal_fwd(
        &self,
        _storage: &MetalStorage,
        _layout: &Layout,
    ) -> CandleResult<(MetalStorage, Shape)> {
        Ok((self.storage.clone(), self.shape.clone()))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn restore_shared_metal_collective(
    storage: MetalStorage,
    rows: usize,
    cols: usize,
    template: &DeviceTensor,
) -> CandleResult<DeviceTensor> {
    template.apply_op1_no_bwd(&RestoreSharedMetalCollective {
        storage,
        shape: Shape::from((rows, cols)),
    })
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
