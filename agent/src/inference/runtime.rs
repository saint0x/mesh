use crate::errors::{AgentError, Result};
#[cfg(test)]
use crate::executor::ring_allreduce::CollectiveMatrix;
use crate::executor::ring_allreduce::StagedCollectiveBuffer;
use crate::provider::{selected_execution_provider, ExecutionProviderKind};
#[cfg(target_os = "linux")]
use candle_core::Storage;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use candle_core::{CustomOp1, Layout, MetalStorage, Result as CandleResult, Shape};
use candle_core::{DType, Device, Tensor as CandleTensor, D};
use candle_nn::ops as candle_ops;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use super::stats::record_runtime_device_sampling;
#[cfg(test)]
use super::stats::{record_runtime_collective_host_restore, record_runtime_collective_host_stage};
use super::tensor_ops::{Tensor1D, Tensor2D};

pub(crate) type DeviceTensor = CandleTensor;
pub(crate) type DeviceDType = DType;
pub(crate) type RuntimeDevice = Device;
pub(crate) type RuntimeError = candle_core::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorResidency {
    Host,
    Device(ExecutionProviderKind),
}

#[derive(Debug, Clone)]
pub enum RuntimeTensor1D {
    Host(Tensor1D),
    Device(DeviceTensor),
}

impl RuntimeTensor1D {
    pub fn from_host(tensor: Tensor1D) -> Self {
        Self::Host(tensor)
    }

    pub fn from_device(tensor: DeviceTensor) -> Result<Self> {
        let dims = tensor.dims();
        if dims.len() != 1 {
            return Err(AgentError::Execution(format!(
                "Expected 1D device tensor, got shape {:?}",
                dims
            )));
        }
        Ok(Self::Device(tensor))
    }

    pub fn residency(&self) -> TensorResidency {
        match self {
            Self::Host(_) => TensorResidency::Host,
            Self::Device(_) => TensorResidency::Device(
                selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu),
            ),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.len(),
            Self::Device(tensor) => tensor.dims().first().copied().unwrap_or(0),
        }
    }

    pub fn to_host(&self) -> Result<Tensor1D> {
        match self {
            Self::Host(tensor) => Ok(tensor.clone()),
            Self::Device(tensor) => {
                let data = tensor
                    .flatten_all()
                    .map_err(runtime_error)?
                    .to_dtype(DeviceDType::F32)
                    .map_err(runtime_error)?
                    .to_vec1::<f32>()
                    .map_err(runtime_error)?;
                Ok(Tensor1D::new(data))
            }
        }
    }

    pub fn to_device(&self) -> Result<DeviceTensor> {
        match self {
            Self::Host(tensor) => device_tensor_from_1d(tensor),
            Self::Device(tensor) => Ok(tensor.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuntimeTensor2D {
    Host(Tensor2D),
    Device(DeviceTensor),
}

impl RuntimeTensor2D {
    pub fn from_host(tensor: Tensor2D) -> Self {
        Self::Host(tensor)
    }

    pub fn from_device(tensor: DeviceTensor) -> Result<Self> {
        let dims = tensor.dims();
        if dims.len() != 2 {
            return Err(AgentError::Execution(format!(
                "Expected 2D device tensor, got shape {:?}",
                dims
            )));
        }
        Ok(Self::Device(tensor))
    }

    pub fn residency(&self) -> TensorResidency {
        match self {
            Self::Host(_) => TensorResidency::Host,
            Self::Device(_) => TensorResidency::Device(
                selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu),
            ),
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.rows,
            Self::Device(tensor) => tensor.dims().first().copied().unwrap_or(0),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Host(tensor) => tensor.cols,
            Self::Device(tensor) => tensor.dims().get(1).copied().unwrap_or(0),
        }
    }

    pub fn to_host(&self) -> Result<Tensor2D> {
        match self {
            Self::Host(tensor) => Ok(tensor.clone()),
            Self::Device(tensor) => host_tensor_2d_from_device(tensor),
        }
    }

    pub fn to_device(&self) -> Result<DeviceTensor> {
        match self {
            Self::Host(tensor) => device_tensor_from_2d(tensor),
            Self::Device(tensor) => Ok(tensor.clone()),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return Err(AgentError::Execution(format!(
                "Shape mismatch for add: {}x{} vs {}x{}",
                self.rows(),
                self.cols(),
                other.rows(),
                other.cols()
            )));
        }
        let lhs = self.to_device()?;
        let rhs = other.to_device()?;
        Ok(Self::Device(
            lhs.broadcast_add(&rhs).map_err(runtime_error)?,
        ))
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            return Err(AgentError::Execution(format!(
                "Shape mismatch for mul: {}x{} vs {}x{}",
                self.rows(),
                self.cols(),
                other.rows(),
                other.cols()
            )));
        }
        let lhs = self.to_device()?;
        let rhs = other.to_device()?;
        Ok(Self::Device(
            lhs.broadcast_mul(&rhs).map_err(runtime_error)?,
        ))
    }

    pub fn scale(&self, scalar: f32) -> Result<Self> {
        Ok(Self::Device(
            self.to_device()?
                .affine(scalar as f64, 0.0)
                .map_err(runtime_error)?,
        ))
    }

    pub fn transpose(&self) -> Result<Self> {
        Ok(Self::Device(
            self.to_device()?.transpose(0, 1).map_err(runtime_error)?,
        ))
    }

    pub fn column_slice(&self, col_start: usize, width: usize) -> Result<Self> {
        Ok(Self::Device(
            self.to_device()?
                .narrow(1, col_start, width)
                .map_err(runtime_error)?,
        ))
    }

    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.cols() != other.rows() {
            return Err(AgentError::Execution(format!(
                "Matmul shape mismatch: {}x{} @ {}x{}",
                self.rows(),
                self.cols(),
                other.rows(),
                other.cols()
            )));
        }
        let lhs = self.to_device()?;
        let rhs = other.to_device()?;
        Ok(Self::Device(lhs.matmul(&rhs).map_err(runtime_error)?))
    }

    pub fn gelu(&self) -> Result<Self> {
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt() as f64;
        let x = self.to_device()?;
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
        Ok(Self::Device(
            scaled.broadcast_mul(&one_plus).map_err(runtime_error)?,
        ))
    }

    pub fn silu(&self) -> Result<Self> {
        Ok(Self::Device(
            candle_ops::silu(&self.to_device()?).map_err(runtime_error)?,
        ))
    }

    pub fn softmax(&self) -> Result<Self> {
        Ok(Self::Device(
            candle_ops::softmax(&self.to_device()?, 1).map_err(runtime_error)?,
        ))
    }

    pub fn apply_rope(&self, positions: &[u32], head_dim: usize, base: f32) -> Result<Self> {
        let tensor = self.to_device()?;
        Ok(Self::Device(apply_rope_device(
            &tensor,
            self.rows(),
            self.cols(),
            positions,
            head_dim,
            base,
        )?))
    }
}

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

#[cfg(test)]
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
        record_runtime_collective_host_stage(
            (matrix.len().saturating_mul(std::mem::size_of::<f32>())) as u64,
        );
        return Ok(matrix);
    }

    let data = flattened.to_vec1::<f32>().map_err(runtime_error)?;
    if selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu)
        != ExecutionProviderKind::Cpu
    {
        record_runtime_collective_host_stage(
            (data.len().saturating_mul(std::mem::size_of::<f32>())) as u64,
        );
    }
    Ok(CollectiveMatrix::new(data, dims[0], dims[1]))
}

pub(crate) struct DeviceCollectiveBuffer {
    flat: DeviceTensor,
    rows: usize,
    cols: usize,
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

        Ok(Self {
            flat,
            rows: dims[0],
            cols: dims[1],
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    fn stage_send_chunk_impl(&mut self, range: Range<usize>) -> Result<Vec<f32>> {
        self.flat
            .narrow(0, range.start, range.len())
            .map_err(runtime_error)?
            .to_vec1::<f32>()
            .map_err(runtime_error)
    }

    fn accumulate_range_from_wire_bytes_impl(
        &mut self,
        range: Range<usize>,
        payload_bytes: &[u8],
    ) -> Result<()> {
        let payload = wire_bytes_to_f32_vec(range.len(), payload_bytes)?;
        let update = DeviceTensor::from_vec(payload, range.len(), self.flat.device())
            .map_err(runtime_error)?;
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
        let payload = wire_bytes_to_f32_vec(range.len(), payload_bytes)?;
        let update = DeviceTensor::from_vec(payload, range.len(), self.flat.device())
            .map_err(runtime_error)?;
        self.flat
            .slice_set(&update, 0, range.start)
            .map_err(runtime_error)
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

    fn stage_send_chunk(&mut self, range: Range<usize>) -> Result<Vec<f32>> {
        self.stage_send_chunk_impl(range)
    }

    fn accumulate_recv_chunk(&mut self, range: Range<usize>, payload_bytes: &[u8]) -> Result<()> {
        self.accumulate_range_from_wire_bytes_impl(range, payload_bytes)
    }

    fn copy_recv_chunk(&mut self, range: Range<usize>, payload_bytes: &[u8]) -> Result<()> {
        self.copy_range_from_wire_bytes_impl(range, payload_bytes)
    }
}

fn wire_bytes_to_f32_vec(expected_len: usize, payload_bytes: &[u8]) -> Result<Vec<f32>> {
    let expected_bytes = expected_len.saturating_mul(std::mem::size_of::<f32>());
    if payload_bytes.len() != expected_bytes {
        return Err(AgentError::Execution(format!(
            "Wire payload byte length {} did not match expected byte length {}",
            payload_bytes.len(),
            expected_bytes
        )));
    }

    Ok(payload_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect())
}

#[cfg(all(test, target_os = "linux"))]
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

#[cfg(all(test, not(target_os = "linux")))]
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

#[cfg(test)]
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

    if selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu)
        != ExecutionProviderKind::Cpu
    {
        record_runtime_collective_host_restore(
            (rows
                .saturating_mul(cols)
                .saturating_mul(std::mem::size_of::<f32>())) as u64,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_tensor_residency_is_explicit() {
        let host = RuntimeTensor2D::from_host(Tensor2D::filled(2, 2, 1.5));
        assert_eq!(host.residency(), TensorResidency::Host);

        let device_tensor = device_tensor_from_2d(&Tensor2D::filled(1, 2, 2.0)).unwrap();
        let device = RuntimeTensor2D::from_device(device_tensor).unwrap();
        assert!(matches!(device.residency(), TensorResidency::Device(_)));
    }

    #[test]
    fn runtime_tensor_ops_stay_device_resident_until_materialized() {
        let lhs =
            RuntimeTensor2D::from_host(Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap());
        let rhs =
            RuntimeTensor2D::from_host(Tensor2D::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap());

        let added = lhs.add(&rhs).unwrap();
        assert!(matches!(added, RuntimeTensor2D::Device(_)));

        let host = added.to_host().unwrap();
        assert_eq!(host.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn staged_collective_buffer_updates_device_resident_tensor() {
        let template =
            device_tensor_from_2d(&Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap()).unwrap();
        let mut buffer = DeviceCollectiveBuffer::from_device_tensor(&template).unwrap();

        let first_row = buffer.stage_send_chunk_impl(0..2).unwrap();
        assert_eq!(first_row, vec![1.0, 2.0]);

        let accumulate = [10.0f32.to_le_bytes(), 20.0f32.to_le_bytes()].concat();
        buffer
            .accumulate_range_from_wire_bytes_impl(0..2, &accumulate)
            .unwrap();

        let replace = [7.0f32.to_le_bytes(), 8.0f32.to_le_bytes()].concat();
        buffer
            .copy_range_from_wire_bytes_impl(2..4, &replace)
            .unwrap();

        let restored =
            host_tensor_2d_from_device(&buffer.into_device_tensor_like(&template).unwrap())
                .unwrap();
        assert_eq!(restored.data, vec![11.0, 22.0, 7.0, 8.0]);
    }
}
