//! Tensor operations for distributed inference
//!
//! This module provides tensor operations needed for transformer forward passes:
//! - Matrix multiplication (partial matmul for tensor parallelism)
//! - Activation functions (GELU, ReLU, SiLU)
//! - Layer normalization
//! - Softmax for attention and sampling
//! - Token embedding lookup

use crate::errors::{AgentError, Result};
use crate::provider::{selected_execution_provider, ExecutionProviderKind};
use candle_core::{DType, Device, Tensor as CandleTensor};
use candle_nn::ops as candle_ops;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::OnceLock;

/// 2D Tensor for transformer computations
///
/// Shape convention: [sequence_length, hidden_dim] or [batch, features]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor2D {
    /// Row-major data storage
    pub data: Vec<f32>,
    /// Number of rows (sequence length or batch size)
    pub rows: usize,
    /// Number of columns (hidden dimension or features)
    pub cols: usize,
}

impl Tensor2D {
    /// Create a new tensor with given dimensions
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(AgentError::Execution(format!(
                "Data length {} doesn't match shape {}x{}={}",
                data.len(),
                rows,
                cols,
                rows * cols
            )));
        }
        Ok(Self { data, rows, cols })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Create a tensor filled with a constant value
    pub fn filled(rows: usize, cols: usize, value: f32) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    /// Get element at (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col)
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    /// Get a row as a slice
    pub fn row(&self, row: usize) -> &[f32] {
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor2D) -> Result<Tensor2D> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(AgentError::Execution(format!(
                "Shape mismatch for add: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let lhs = to_candle_2d(self)?;
        let rhs = to_candle_2d(other)?;
        from_candle_2d(&lhs.broadcast_add(&rhs).map_err(candle_error)?)
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor2D) -> Result<Tensor2D> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(AgentError::Execution(format!(
                "Shape mismatch for mul: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let lhs = to_candle_2d(self)?;
        let rhs = to_candle_2d(other)?;
        from_candle_2d(&lhs.broadcast_mul(&rhs).map_err(candle_error)?)
    }

    /// Scale by a scalar
    pub fn scale(&self, scalar: f32) -> Tensor2D {
        let tensor = to_candle_2d(self)
            .and_then(|x| from_candle_2d(&x.affine(scalar as f64, 0.0).map_err(candle_error)?))
            .expect("GPU tensor scaling failed");
        tensor
    }

    /// Transpose the tensor
    pub fn transpose(&self) -> Tensor2D {
        let tensor = to_candle_2d(self)
            .and_then(|x| from_candle_2d(&x.transpose(0, 1).map_err(candle_error)?))
            .expect("GPU tensor transpose failed");
        tensor
    }

    /// Convert to 1D tensor (flatten)
    pub fn flatten(&self) -> Tensor1D {
        Tensor1D {
            data: self.data.clone(),
        }
    }

    /// Extract column range for tensor parallelism
    ///
    /// Used to get the columns this worker is responsible for
    pub fn column_slice(&self, col_start: usize, col_end: usize) -> Result<Tensor2D> {
        if col_end > self.cols || col_start >= col_end {
            return Err(AgentError::Execution(format!(
                "Invalid column slice {}..{} for tensor with {} cols",
                col_start, col_end, self.cols
            )));
        }

        let slice_cols = col_end - col_start;
        let tensor = to_candle_2d(self)?;
        from_candle_2d(&tensor.narrow(1, col_start, slice_cols).map_err(candle_error)?)
    }
}

/// 1D Tensor (vector)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor1D {
    pub data: Vec<f32>,
}

impl Tensor1D {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshape to 2D tensor
    pub fn reshape(&self, rows: usize, cols: usize) -> Result<Tensor2D> {
        Tensor2D::new(self.data.clone(), rows, cols)
    }
}

// ============== Matrix Operations ==============

/// Matrix multiplication: A[m, k] @ B[k, n] -> C[m, n]
///
/// For tensor parallelism, this computes partial results when B contains
/// only a subset of columns.
pub fn matmul(a: &Tensor2D, b: &Tensor2D) -> Result<Tensor2D> {
    if a.cols != b.rows {
        return Err(AgentError::Execution(format!(
            "Matmul shape mismatch: {}x{} @ {}x{}",
            a.rows, a.cols, b.rows, b.cols
        )));
    }

    let lhs = to_candle_2d(a)?;
    let rhs = to_candle_2d(b)?;
    from_candle_2d(&lhs.matmul(&rhs).map_err(candle_error)?)
}

/// Matrix-vector multiplication: A[m, n] @ v[n] -> result[m]
pub fn matvec(a: &Tensor2D, v: &Tensor1D) -> Result<Tensor1D> {
    if a.cols != v.len() {
        return Err(AgentError::Execution(format!(
            "Matvec shape mismatch: {}x{} @ {}",
            a.rows,
            a.cols,
            v.len()
        )));
    }

    let lhs = to_candle_2d(a)?;
    let rhs = CandleTensor::from_vec(v.data.clone(), (v.len(), 1), execution_device()?)
        .map_err(candle_error)?;
    let result = lhs.matmul(&rhs).map_err(candle_error)?;
    Ok(Tensor1D {
        data: result
            .flatten_all()
            .map_err(candle_error)?
            .to_vec1::<f32>()
            .map_err(candle_error)?,
    })
}

// ============== Activation Functions ==============

/// GELU activation function (Gaussian Error Linear Unit)
///
/// Used in GPT-2, BERT, and many modern transformers.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(tensor: &Tensor2D) -> Tensor2D {
    let sqrt_2_over_pi = (2.0 / PI).sqrt() as f64;
    let result = (|| -> Result<Tensor2D> {
        let x = to_candle_2d(tensor)?;
        let x3 = x.sqr().map_err(candle_error)?.broadcast_mul(&x).map_err(candle_error)?;
        let inner = x
            .affine(0.044715, 0.0)
            .map_err(candle_error)?
            .broadcast_mul(&x3)
            .map_err(candle_error)?;
        let inner = x.broadcast_add(&inner).map_err(candle_error)?;
        let inner = inner.affine(sqrt_2_over_pi, 0.0).map_err(candle_error)?.tanh().map_err(candle_error)?;
        let one_plus = inner.affine(1.0, 1.0).map_err(candle_error)?;
        let scaled = x.affine(0.5, 0.0).map_err(candle_error)?;
        from_candle_2d(&scaled.broadcast_mul(&one_plus).map_err(candle_error)?)
    })();
    result.expect("GPU GELU failed")
}

/// ReLU activation function
pub fn relu(tensor: &Tensor2D) -> Tensor2D {
    let result = to_candle_2d(tensor)
        .and_then(|x| from_candle_2d(&x.relu().map_err(candle_error)?))
        .expect("GPU ReLU failed");
    result
}

/// SiLU (Sigmoid Linear Unit) / Swish activation
///
/// Used in LLaMA and other modern models.
/// silu(x) = x * sigmoid(x)
pub fn silu(tensor: &Tensor2D) -> Tensor2D {
    let result = to_candle_2d(tensor)
        .and_then(|x| from_candle_2d(&candle_ops::silu(&x).map_err(candle_error)?))
        .expect("GPU SiLU failed");
    result
}

// ============== Normalization ==============

/// RMS Layer Normalization (used in LLaMA)
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
pub fn rms_norm(tensor: &Tensor2D, gamma: &Tensor1D, eps: f32) -> Result<Tensor2D> {
    if tensor.cols != gamma.len() {
        return Err(AgentError::Execution(format!(
            "RMS norm dimension mismatch: tensor cols {} vs gamma {}",
            tensor.cols,
            gamma.len()
        )));
    }

    let x = to_candle_2d(tensor)?;
    let weight = to_candle_1d(gamma)?;
    from_candle_2d(&candle_ops::rms_norm(&x, &weight, eps).map_err(candle_error)?)
}

/// Standard Layer Normalization
///
/// LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
pub fn layer_norm(
    tensor: &Tensor2D,
    gamma: &Tensor1D,
    beta: &Tensor1D,
    eps: f32,
) -> Result<Tensor2D> {
    if tensor.cols != gamma.len() || tensor.cols != beta.len() {
        return Err(AgentError::Execution(format!(
            "Layer norm dimension mismatch: tensor cols {} vs gamma {} vs beta {}",
            tensor.cols,
            gamma.len(),
            beta.len()
        )));
    }

    let x = to_candle_2d(tensor)?;
    let gamma = to_candle_1d(gamma)?;
    let beta = to_candle_1d(beta)?;
    from_candle_2d(&candle_ops::layer_norm(&x, &gamma, &beta, eps).map_err(candle_error)?)
}

// ============== Softmax ==============

/// Softmax over the last dimension (columns)
///
/// softmax(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax(tensor: &Tensor2D) -> Tensor2D {
    let result = to_candle_2d(tensor)
        .and_then(|x| from_candle_2d(&candle_ops::softmax(&x, 1).map_err(candle_error)?))
        .expect("GPU softmax failed");
    result
}

/// Softmax for 1D tensor (single row of logits)
pub fn softmax_1d(tensor: &Tensor1D) -> Tensor1D {
    let max_val = tensor
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = tensor.data.iter().map(|x| (x - max_val).exp()).sum();

    let data: Vec<f32> = tensor
        .data
        .iter()
        .map(|x| (x - max_val).exp() / exp_sum)
        .collect();

    Tensor1D { data }
}

// ============== Token Sampling ==============

/// Sample a token from logits with temperature and top-p (nucleus) sampling
///
/// # Arguments
/// * `logits` - Raw logits from the model (vocab_size,)
/// * `temperature` - Temperature for scaling (1.0 = no change, lower = more deterministic)
/// * `top_p` - Nucleus sampling threshold (0.9 = consider tokens comprising 90% probability mass)
/// * `rng_seed` - Random seed for reproducibility
pub fn sample_token(logits: &Tensor1D, temperature: f32, top_p: f32, rng_seed: u64) -> u32 {
    // Apply temperature
    let scaled_logits: Vec<f32> = if temperature != 1.0 && temperature > 0.0 {
        logits.data.iter().map(|x| x / temperature).collect()
    } else {
        logits.data.clone()
    };

    // Convert to probabilities via softmax
    let max_val = scaled_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled_logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

    // Sort by probability (descending) for top-p
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Apply top-p (nucleus) sampling
    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed_probs.len();
    for (i, (_, prob)) in indexed_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Renormalize the selected tokens
    let selected: Vec<(usize, f32)> = indexed_probs[..cutoff_idx].to_vec();
    let selected_sum: f32 = selected.iter().map(|(_, p)| p).sum();
    let renormalized: Vec<(usize, f32)> = selected
        .iter()
        .map(|(idx, p)| (*idx, p / selected_sum))
        .collect();

    // Simple LCG random number generator for reproducibility
    let mut rng_state = rng_seed;
    let random_val = {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32)
    };

    // Sample from the distribution
    let mut cumsum = 0.0;
    for (idx, prob) in renormalized {
        cumsum += prob;
        if random_val < cumsum {
            return idx as u32;
        }
    }

    // Fallback to highest probability token
    indexed_probs[0].0 as u32
}

/// Greedy sampling (argmax)
pub fn sample_greedy(logits: &Tensor1D) -> u32 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;

    for (i, &val) in logits.data.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    max_idx as u32
}

// ============== Embedding Operations ==============

/// Look up token embeddings from embedding table
///
/// # Arguments
/// * `embedding_table` - [vocab_size, hidden_dim] embedding matrix
/// * `tokens` - Token IDs to look up
pub fn embed_tokens(embedding_table: &Tensor2D, tokens: &[u32]) -> Result<Tensor2D> {
    for &token in tokens {
        let token_idx = token as usize;
        if token_idx >= embedding_table.rows {
            return Err(AgentError::Execution(format!(
                "Token {} out of vocabulary bounds {}",
                token, embedding_table.rows
            )));
        }
    }

    let table = to_candle_2d(embedding_table)?;
    let ids = CandleTensor::from_vec(tokens.to_vec(), tokens.len(), execution_device()?).map_err(candle_error)?;
    from_candle_2d(&table.embedding(&ids).map_err(candle_error)?)
}

// ============== Rotary Position Embedding (RoPE) ==============

/// Apply Rotary Position Embedding (used in LLaMA)
///
/// RoPE rotates pairs of dimensions based on position, enabling
/// relative position encoding without explicit position embeddings.
pub fn apply_rope(
    tensor: &Tensor2D,
    positions: &[u32],
    head_dim: usize,
    base: f32,
) -> Result<Tensor2D> {
    if tensor.rows != positions.len() {
        return Err(AgentError::Execution(format!(
            "RoPE position count {} doesn't match sequence length {}",
            positions.len(),
            tensor.rows
        )));
    }

    if head_dim % 2 != 0 {
        return Err(AgentError::Execution(format!(
            "RoPE requires even head_dim, got {}",
            head_dim
        )));
    }
    if tensor.cols % head_dim != 0 {
        return Err(AgentError::Execution(format!(
            "RoPE head_dim {} does not divide tensor width {}",
            head_dim, tensor.cols
        )));
    }

    let seq_len = tensor.rows;
    let num_heads = tensor.cols / head_dim;
    let half_dim = head_dim / 2;
    let x = to_candle_2d(tensor)?
        .reshape((seq_len, num_heads, head_dim))
        .map_err(candle_error)?;
    let x1 = x.narrow(2, 0, half_dim).map_err(candle_error)?;
    let x2 = x.narrow(2, half_dim, half_dim).map_err(candle_error)?;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();
    let pos = CandleTensor::from_vec(
        positions.iter().map(|p| *p as f32).collect::<Vec<_>>(),
        (seq_len, 1),
        execution_device()?,
    )
    .map_err(candle_error)?;
    let inv = CandleTensor::from_vec(inv_freq, (1, half_dim), execution_device()?).map_err(candle_error)?;
    let freqs = pos.broadcast_matmul(&inv).map_err(candle_error)?;
    let cos = freqs
        .cos()
        .map_err(candle_error)?
        .unsqueeze(1)
        .map_err(candle_error)?
        .expand((seq_len, num_heads, half_dim))
        .map_err(candle_error)?;
    let sin = freqs
        .sin()
        .map_err(candle_error)?
        .unsqueeze(1)
        .map_err(candle_error)?
        .expand((seq_len, num_heads, half_dim))
        .map_err(candle_error)?;

    let rot1 = x1
        .broadcast_mul(&cos)
        .map_err(candle_error)?
        .broadcast_sub(&x2.broadcast_mul(&sin).map_err(candle_error)?)
        .map_err(candle_error)?;
    let rot2 = x1
        .broadcast_mul(&sin)
        .map_err(candle_error)?
        .broadcast_add(&x2.broadcast_mul(&cos).map_err(candle_error)?)
        .map_err(candle_error)?;
    let rotated = CandleTensor::cat(&[&rot1, &rot2], 2).map_err(candle_error)?;
    from_candle_2d(&rotated.reshape((seq_len, tensor.cols)).map_err(candle_error)?)
}

// ============== Conversion to/from ring_allreduce::Tensor ==============

impl Tensor2D {
    /// Convert to flat Tensor for ring all-reduce
    pub fn to_allreduce_tensor(&self) -> crate::executor::ring_allreduce::Tensor {
        crate::executor::ring_allreduce::Tensor::new(self.data.clone(), vec![self.rows, self.cols])
    }

    /// Create from ring all-reduce Tensor
    pub fn from_allreduce_tensor(tensor: &crate::executor::ring_allreduce::Tensor) -> Result<Self> {
        if tensor.shape.len() != 2 {
            return Err(AgentError::Execution(format!(
                "Expected 2D tensor, got {}D",
                tensor.shape.len()
            )));
        }
        Ok(Self {
            data: tensor.data.clone(),
            rows: tensor.shape[0],
            cols: tensor.shape[1],
        })
    }
}

fn candle_error(err: candle_core::Error) -> AgentError {
    AgentError::Execution(format!("Tensor backend error: {}", err))
}

pub(crate) fn execution_device() -> Result<&'static Device> {
    static DEVICE: OnceLock<std::result::Result<Device, String>> = OnceLock::new();
    match DEVICE.get_or_init(|| init_execution_device().map_err(|e| e.to_string())) {
        Ok(device) => Ok(device),
        Err(err) => Err(AgentError::Execution(format!(
            "Execution backend unavailable: {}",
            err
        ))),
    }
}

fn init_execution_device() -> std::result::Result<Device, candle_core::Error> {
    let provider = selected_execution_provider().unwrap_or(ExecutionProviderKind::Cpu);
    match provider {
        ExecutionProviderKind::Cpu => Ok(Device::Cpu),
        ExecutionProviderKind::Cuda => {
            #[cfg(target_os = "linux")]
            {
                Device::new_cuda(0)
            }
            #[cfg(not(target_os = "linux"))]
            {
                Err(candle_core::Error::Msg(
                    "cuda provider is unavailable on this platform".to_string(),
                ))
            }
        }
        ExecutionProviderKind::Metal => {
            #[cfg(target_os = "macos")]
            {
                Device::new_metal(0)
            }
            #[cfg(not(target_os = "macos"))]
            {
                Err(candle_core::Error::Msg(
                    "metal provider is unavailable on this platform".to_string(),
                ))
            }
        }
    }
}

pub(crate) fn to_candle_2d(tensor: &Tensor2D) -> Result<CandleTensor> {
    CandleTensor::from_vec(tensor.data.clone(), (tensor.rows, tensor.cols), execution_device()?)
        .map_err(candle_error)
}

pub(crate) fn to_candle_1d(tensor: &Tensor1D) -> Result<CandleTensor> {
    CandleTensor::from_vec(tensor.data.clone(), tensor.len(), execution_device()?).map_err(candle_error)
}

pub(crate) fn from_candle_2d(tensor: &CandleTensor) -> Result<Tensor2D> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(AgentError::Execution(format!(
            "Expected 2D GPU tensor, got shape {:?}",
            dims
        )));
    }
    let data = tensor
        .flatten_all()
        .map_err(candle_error)?
        .to_dtype(DType::F32)
        .map_err(candle_error)?
        .to_vec1::<f32>()
        .map_err(candle_error)?;
    Tensor2D::new(data, dims[0], dims[1])
}

pub(crate) fn rms_norm_candle(tensor: &CandleTensor, gamma: &CandleTensor, eps: f32) -> Result<CandleTensor> {
    candle_ops::rms_norm(tensor, gamma, eps).map_err(candle_error)
}

pub(crate) fn silu_candle(tensor: &CandleTensor) -> Result<CandleTensor> {
    candle_ops::silu(tensor).map_err(candle_error)
}

pub(crate) fn apply_rope_candle(
    tensor: &CandleTensor,
    rows: usize,
    cols: usize,
    positions: &[u32],
    head_dim: usize,
    base: f32,
) -> Result<CandleTensor> {
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
        .map_err(candle_error)?;
    let x1 = x.narrow(2, 0, half_dim).map_err(candle_error)?;
    let x2 = x.narrow(2, half_dim, half_dim).map_err(candle_error)?;

    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();
    let pos = CandleTensor::from_vec(
        positions.iter().map(|p| *p as f32).collect::<Vec<_>>(),
        (rows, 1),
        execution_device()?,
    )
    .map_err(candle_error)?;
    let inv = CandleTensor::from_vec(inv_freq, (1, half_dim), execution_device()?).map_err(candle_error)?;
    let freqs = pos.broadcast_matmul(&inv).map_err(candle_error)?;
    let cos = freqs
        .cos()
        .map_err(candle_error)?
        .unsqueeze(1)
        .map_err(candle_error)?
        .expand((rows, num_heads, half_dim))
        .map_err(candle_error)?;
    let sin = freqs
        .sin()
        .map_err(candle_error)?
        .unsqueeze(1)
        .map_err(candle_error)?
        .expand((rows, num_heads, half_dim))
        .map_err(candle_error)?;

    let rot1 = x1
        .broadcast_mul(&cos)
        .map_err(candle_error)?
        .broadcast_sub(&x2.broadcast_mul(&sin).map_err(candle_error)?)
        .map_err(candle_error)?;
    let rot2 = x1
        .broadcast_mul(&sin)
        .map_err(candle_error)?
        .broadcast_add(&x2.broadcast_mul(&cos).map_err(candle_error)?)
        .map_err(candle_error)?;
    CandleTensor::cat(&[&rot1, &rot2], 2)
        .map_err(candle_error)?
        .reshape((rows, cols))
        .map_err(candle_error)
}

pub(crate) fn causal_self_attention_candle(
    q_t: &CandleTensor,
    k_t: &CandleTensor,
    v_t: &CandleTensor,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<CandleTensor> {
    let q_t = q_t.contiguous().map_err(candle_error)?;
    let k_t = k_t
        .transpose(0, 1)
        .map_err(candle_error)?
        .contiguous()
        .map_err(candle_error)?;
    let v_t = v_t.contiguous().map_err(candle_error)?;
    let scores = q_t
        .matmul(&k_t)
        .map_err(candle_error)?;
    let scores = scores.affine(scale as f64, 0.0).map_err(candle_error)?;

    let mut mask = vec![0.0f32; rows * rows];
    for i in 0..rows {
        for j in (i + 1)..rows {
            mask[i * rows + j] = f32::NEG_INFINITY;
        }
    }
    let mask = CandleTensor::from_vec(mask, (rows, rows), execution_device()?).map_err(candle_error)?;
    let masked = scores.broadcast_add(&mask).map_err(candle_error)?;
    let probs = candle_ops::softmax(&masked, 1).map_err(candle_error)?;
    let probs = probs.contiguous().map_err(candle_error)?;
    let output = probs.matmul(&v_t).map_err(candle_error)?;
    let dims = output.dims();
    if dims != [rows, cols] {
        return Err(AgentError::Execution(format!(
            "GPU attention output shape mismatch: expected [{}, {}], got {:?}",
            rows, cols, dims
        )));
    }
    Ok(output)
}

pub fn causal_self_attention_gpu(q: &Tensor2D, k: &Tensor2D, v: &Tensor2D, scale: f32) -> Result<Tensor2D> {
    if q.rows != k.rows || q.rows != v.rows || q.cols != k.cols || q.cols != v.cols {
        return Err(AgentError::Execution(format!(
            "Attention shape mismatch: q {}x{}, k {}x{}, v {}x{}",
            q.rows, q.cols, k.rows, k.cols, v.rows, v.cols
        )));
    }

    let q_t = to_candle_2d(q)?;
    let k_t = to_candle_2d(k)?;
    let v_t = to_candle_2d(v)?;
    from_candle_2d(&causal_self_attention_candle(
        &q_t, &k_t, &v_t, q.rows, q.cols, scale,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_tensor2d_creation() {
        let t = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 1), 4.0);
    }

    #[test]
    #[serial]
    fn test_matmul() {
        // [2,3] @ [3,2] = [2,2]
        let a = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let b = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        let c = matmul(&a, &b).unwrap();

        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);
        // First row: [1,2,3] @ [1,3,5; 2,4,6] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        assert_eq!(c.get(0, 0), 22.0);
        assert_eq!(c.get(0, 1), 28.0);
    }

    #[test]
    #[serial]
    fn test_gelu() {
        let t = Tensor2D::new(vec![0.0, 1.0, -1.0, 2.0], 2, 2).unwrap();
        let result = gelu(&t);

        // GELU(0) ≈ 0
        assert!((result.get(0, 0) - 0.0).abs() < 0.01);
        // GELU(1) ≈ 0.841
        assert!((result.get(0, 1) - 0.841).abs() < 0.01);
    }

    #[test]
    #[serial]
    fn test_softmax() {
        let t = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 1, 4).unwrap();
        let result = softmax(&t);

        // Sum should be 1.0
        let sum: f32 = result.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);

        // Values should be monotonically increasing
        assert!(result.get(0, 3) > result.get(0, 2));
        assert!(result.get(0, 2) > result.get(0, 1));
    }

    #[test]
    #[serial]
    fn test_layer_norm() {
        let t = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let gamma = Tensor1D::new(vec![1.0, 1.0]);
        let beta = Tensor1D::new(vec![0.0, 0.0]);

        let result = layer_norm(&t, &gamma, &beta, 1e-5).unwrap();

        // After normalization, each row should have mean ≈ 0 and std ≈ 1
        let row0_mean = (result.get(0, 0) + result.get(0, 1)) / 2.0;
        assert!(row0_mean.abs() < 0.01);
    }

    #[test]
    #[serial]
    fn test_sample_greedy() {
        let logits = Tensor1D::new(vec![1.0, 5.0, 2.0, 3.0]);
        let token = sample_greedy(&logits);
        assert_eq!(token, 1); // Index of max value (5.0)
    }

    #[test]
    #[serial]
    fn test_column_slice() {
        let t = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let slice = t.column_slice(1, 3).unwrap();

        assert_eq!(slice.rows, 2);
        assert_eq!(slice.cols, 2);
        assert_eq!(slice.get(0, 0), 2.0);
        assert_eq!(slice.get(0, 1), 3.0);
        assert_eq!(slice.get(1, 0), 5.0);
        assert_eq!(slice.get(1, 1), 6.0);
    }

    #[test]
    #[serial]
    fn test_embed_tokens() {
        // 4 tokens x 3 dim embedding
        let embedding = Tensor2D::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            4,
            3,
        )
        .unwrap();

        let tokens = vec![0, 2, 1];
        let result = embed_tokens(&embedding, &tokens).unwrap();

        assert_eq!(result.rows, 3); // 3 tokens
        assert_eq!(result.cols, 3); // 3 dim
        assert_eq!(result.get(0, 0), 0.1); // Token 0
        assert_eq!(result.get(1, 0), 0.7); // Token 2
        assert_eq!(result.get(2, 0), 0.4); // Token 1
    }

    #[test]
    #[serial]
    fn test_transpose() {
        let t = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let transposed = t.transpose();

        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 2);
        assert_eq!(transposed.get(0, 0), 1.0);
        assert_eq!(transposed.get(0, 1), 4.0);
        assert_eq!(transposed.get(2, 1), 6.0);
    }
}
