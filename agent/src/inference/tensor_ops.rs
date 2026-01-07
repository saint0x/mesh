//! Tensor operations for distributed inference
//!
//! This module provides tensor operations needed for transformer forward passes:
//! - Matrix multiplication (partial matmul for tensor parallelism)
//! - Activation functions (GELU, ReLU, SiLU)
//! - Layer normalization
//! - Softmax for attention and sampling
//! - Token embedding lookup

use crate::errors::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

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

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();

        Ok(Tensor2D {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Element-wise multiplication (Hadamard product)
    pub fn mul(&self, other: &Tensor2D) -> Result<Tensor2D> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(AgentError::Execution(format!(
                "Shape mismatch for mul: {}x{} vs {}x{}",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();

        Ok(Tensor2D {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Scale by a scalar
    pub fn scale(&self, scalar: f32) -> Tensor2D {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Tensor2D {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Transpose the tensor
    pub fn transpose(&self) -> Tensor2D {
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Tensor2D {
            data,
            rows: self.cols,
            cols: self.rows,
        }
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
        let mut data = Vec::with_capacity(self.rows * slice_cols);

        for row in 0..self.rows {
            let row_start = row * self.cols + col_start;
            let row_end = row * self.cols + col_end;
            data.extend_from_slice(&self.data[row_start..row_end]);
        }

        Ok(Tensor2D {
            data,
            rows: self.rows,
            cols: slice_cols,
        })
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

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    let mut result = vec![0.0; m * n];

    // Standard matmul with cache-friendly access pattern
    for i in 0..m {
        for p in 0..k {
            let a_ip = a.data[i * k + p];
            for j in 0..n {
                result[i * n + j] += a_ip * b.data[p * n + j];
            }
        }
    }

    Ok(Tensor2D {
        data: result,
        rows: m,
        cols: n,
    })
}

/// Matrix-vector multiplication: A[m, n] @ v[n] -> result[m]
pub fn matvec(a: &Tensor2D, v: &Tensor1D) -> Result<Tensor1D> {
    if a.cols != v.len() {
        return Err(AgentError::Execution(format!(
            "Matvec shape mismatch: {}x{} @ {}",
            a.rows, a.cols, v.len()
        )));
    }

    let mut result = vec![0.0; a.rows];
    for i in 0..a.rows {
        let mut sum = 0.0;
        for j in 0..a.cols {
            sum += a.data[i * a.cols + j] * v.data[j];
        }
        result[i] = sum;
    }

    Ok(Tensor1D { data: result })
}

// ============== Activation Functions ==============

/// GELU activation function (Gaussian Error Linear Unit)
///
/// Used in GPT-2, BERT, and many modern transformers.
/// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(tensor: &Tensor2D) -> Tensor2D {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();

    let data: Vec<f32> = tensor
        .data
        .iter()
        .map(|&x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();

    Tensor2D {
        data,
        rows: tensor.rows,
        cols: tensor.cols,
    }
}

/// ReLU activation function
pub fn relu(tensor: &Tensor2D) -> Tensor2D {
    let data: Vec<f32> = tensor.data.iter().map(|&x| x.max(0.0)).collect();
    Tensor2D {
        data,
        rows: tensor.rows,
        cols: tensor.cols,
    }
}

/// SiLU (Sigmoid Linear Unit) / Swish activation
///
/// Used in LLaMA and other modern models.
/// silu(x) = x * sigmoid(x)
pub fn silu(tensor: &Tensor2D) -> Tensor2D {
    let data: Vec<f32> = tensor
        .data
        .iter()
        .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
        .collect();

    Tensor2D {
        data,
        rows: tensor.rows,
        cols: tensor.cols,
    }
}

// ============== Normalization ==============

/// RMS Layer Normalization (used in LLaMA)
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
pub fn rms_norm(tensor: &Tensor2D, gamma: &Tensor1D, eps: f32) -> Result<Tensor2D> {
    if tensor.cols != gamma.len() {
        return Err(AgentError::Execution(format!(
            "RMS norm dimension mismatch: tensor cols {} vs gamma {}",
            tensor.cols, gamma.len()
        )));
    }

    let mut result = Vec::with_capacity(tensor.data.len());

    for row in 0..tensor.rows {
        // Compute RMS for this row
        let row_start = row * tensor.cols;
        let row_end = row_start + tensor.cols;
        let row_data = &tensor.data[row_start..row_end];

        let mean_sq: f32 = row_data.iter().map(|x| x * x).sum::<f32>() / tensor.cols as f32;
        let rms = (mean_sq + eps).sqrt();

        // Normalize and scale by gamma
        for (i, &x) in row_data.iter().enumerate() {
            result.push((x / rms) * gamma.data[i]);
        }
    }

    Ok(Tensor2D {
        data: result,
        rows: tensor.rows,
        cols: tensor.cols,
    })
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

    let mut result = Vec::with_capacity(tensor.data.len());

    for row in 0..tensor.rows {
        let row_start = row * tensor.cols;
        let row_end = row_start + tensor.cols;
        let row_data = &tensor.data[row_start..row_end];

        // Compute mean
        let mean: f32 = row_data.iter().sum::<f32>() / tensor.cols as f32;

        // Compute variance
        let variance: f32 =
            row_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / tensor.cols as f32;

        let std = (variance + eps).sqrt();

        // Normalize, scale by gamma, shift by beta
        for (i, &x) in row_data.iter().enumerate() {
            result.push(((x - mean) / std) * gamma.data[i] + beta.data[i]);
        }
    }

    Ok(Tensor2D {
        data: result,
        rows: tensor.rows,
        cols: tensor.cols,
    })
}

// ============== Softmax ==============

/// Softmax over the last dimension (columns)
///
/// softmax(x_i) = exp(x_i) / sum(exp(x_j))
pub fn softmax(tensor: &Tensor2D) -> Tensor2D {
    let mut result = Vec::with_capacity(tensor.data.len());

    for row in 0..tensor.rows {
        let row_start = row * tensor.cols;
        let row_end = row_start + tensor.cols;
        let row_data = &tensor.data[row_start..row_end];

        // Numerical stability: subtract max
        let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row_data.iter().map(|x| (x - max_val).exp()).sum();

        for &x in row_data {
            result.push((x - max_val).exp() / exp_sum);
        }
    }

    Tensor2D {
        data: result,
        rows: tensor.rows,
        cols: tensor.cols,
    }
}

/// Softmax for 1D tensor (single row of logits)
pub fn softmax_1d(tensor: &Tensor1D) -> Tensor1D {
    let max_val = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
pub fn sample_token(
    logits: &Tensor1D,
    temperature: f32,
    top_p: f32,
    rng_seed: u64,
) -> u32 {
    // Apply temperature
    let scaled_logits: Vec<f32> = if temperature != 1.0 && temperature > 0.0 {
        logits.data.iter().map(|x| x / temperature).collect()
    } else {
        logits.data.clone()
    };

    // Convert to probabilities via softmax
    let max_val = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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
    let hidden_dim = embedding_table.cols;
    let seq_len = tokens.len();

    let mut data = Vec::with_capacity(seq_len * hidden_dim);

    for &token in tokens {
        let token_idx = token as usize;
        if token_idx >= embedding_table.rows {
            return Err(AgentError::Execution(format!(
                "Token {} out of vocabulary bounds {}",
                token, embedding_table.rows
            )));
        }

        let row_start = token_idx * hidden_dim;
        let row_end = row_start + hidden_dim;
        data.extend_from_slice(&embedding_table.data[row_start..row_end]);
    }

    Ok(Tensor2D {
        data,
        rows: seq_len,
        cols: hidden_dim,
    })
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

    let mut result = tensor.data.clone();
    let half_dim = head_dim / 2;

    // Precompute inverse frequencies
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
        .collect();

    for (seq_idx, &pos) in positions.iter().enumerate() {
        let pos_f = pos as f32;

        // Apply RoPE to each head dimension pair
        for d in 0..half_dim {
            let freq = pos_f * inv_freq[d];
            let cos_val = freq.cos();
            let sin_val = freq.sin();

            // For each column that should be rotated
            for col_base in (0..tensor.cols).step_by(head_dim) {
                let idx1 = seq_idx * tensor.cols + col_base + d;
                let idx2 = seq_idx * tensor.cols + col_base + d + half_dim;

                if idx2 < result.len() {
                    let x1 = tensor.data[idx1];
                    let x2 = tensor.data[idx2];

                    result[idx1] = x1 * cos_val - x2 * sin_val;
                    result[idx2] = x1 * sin_val + x2 * cos_val;
                }
            }
        }
    }

    Ok(Tensor2D {
        data: result,
        rows: tensor.rows,
        cols: tensor.cols,
    })
}

// ============== Conversion to/from ring_allreduce::Tensor ==============

impl Tensor2D {
    /// Convert to flat Tensor for ring all-reduce
    pub fn to_allreduce_tensor(&self) -> crate::executor::ring_allreduce::Tensor {
        crate::executor::ring_allreduce::Tensor::new(
            self.data.clone(),
            vec![self.rows, self.cols],
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor2d_creation() {
        let t = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(t.rows, 2);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 1), 4.0);
    }

    #[test]
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
    fn test_gelu() {
        let t = Tensor2D::new(vec![0.0, 1.0, -1.0, 2.0], 2, 2).unwrap();
        let result = gelu(&t);

        // GELU(0) ≈ 0
        assert!((result.get(0, 0) - 0.0).abs() < 0.01);
        // GELU(1) ≈ 0.841
        assert!((result.get(0, 1) - 0.841).abs() < 0.01);
    }

    #[test]
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
    fn test_sample_greedy() {
        let logits = Tensor1D::new(vec![1.0, 5.0, 2.0, 3.0]);
        let token = sample_greedy(&logits);
        assert_eq!(token, 1); // Index of max value (5.0)
    }

    #[test]
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
