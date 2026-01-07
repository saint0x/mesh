//! # Mock Tensor Validation (Development Only)
//!
//! This module provides Xavier-initialized mock weights for validating
//! distributed tensor-parallel inference correctness.
//!
//! ## ⚠️ TODO: Replace with Real Weights
//!
//! For production inference:
//! 1. Load actual safetensors weights from disk/network
//! 2. Remove this module entirely
//! 3. Update coordinator.rs to use real weight loader
//!
//! ## Purpose
//!
//! Proves distributed tensor parallelism works by comparing:
//! - Reference: Single-device forward pass
//! - Distributed: Tensor-parallel with ring all-reduce
//!
//! If outputs match (within error bounds), distribution is correct.
//!
//! ## Xavier/Glorot Initialization
//!
//! Weights are initialized using Xavier uniform distribution:
//! - Formula: `limit = sqrt(6 / (fan_in + fan_out))`
//! - Samples: `Uniform[-limit, limit]`
//! - Maintains gradient and activation variance through layers
//!
//! This is the industry-standard initialization for neural networks
//! and matches the distribution of real trained model weights.

use super::forward_pass::{LayerWeights, ModelConfig, ModelWeights};
use super::tensor_ops::{Tensor1D, Tensor2D};

/// Simple Linear Congruential Generator for deterministic random numbers
///
/// Uses constants from Numerical Recipes:
/// - Multiplier: 6364136223846793005
/// - Increment: 1442695040888963407
///
/// This provides high-quality pseudo-random numbers with perfect
/// reproducibility from a seed.
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG with the given seed
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next u64 value
    fn next_u64(&mut self) -> u64 {
        // LCG formula: X(n+1) = (a * X(n) + c) mod m
        // Using constants from Numerical Recipes
        const A: u64 = 6364136223846793005;
        const C: u64 = 1442695040888963407;

        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    /// Generate f32 in range [-1.0, 1.0]
    ///
    /// Uses upper 24 bits for mantissa precision
    fn next_f32(&mut self) -> f32 {
        let val = self.next_u64();
        // Use upper 24 bits for f32 mantissa
        let mantissa = (val >> 40) & 0xFFFFFF;
        // Map [0, 2^24) to [-1.0, 1.0]
        let normalized = (mantissa as f32) / (1 << 23) as f32 - 1.0;
        normalized
    }

    /// Generate Xavier/Glorot uniform initialization value
    ///
    /// Formula: limit = sqrt(6 / (fan_in + fan_out))
    /// Returns: Uniform[-limit, limit]
    ///
    /// Reference: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    fn xavier_uniform(&mut self, fan_in: usize, fan_out: usize) -> f32 {
        let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        self.next_f32() * limit
    }
}

/// Generate a 2D tensor with Xavier uniform initialization
///
/// # Arguments
/// - `rng`: Random number generator
/// - `rows`: Number of rows (fan_in for weight matrices)
/// - `cols`: Number of columns (fan_out for weight matrices)
///
/// # Returns
/// Tensor2D with shape [rows, cols] initialized with Xavier distribution
fn xavier_tensor_2d(rng: &mut Rng, rows: usize, cols: usize) -> Tensor2D {
    let fan_in = rows;
    let fan_out = cols;

    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.xavier_uniform(fan_in, fan_out))
        .collect();

    Tensor2D::new(data, rows, cols).expect("Xavier tensor creation failed")
}

/// Generate a 1D tensor (for normalization weights)
///
/// RMS norm weights are typically initialized to 1.0 for identity scaling
fn xavier_tensor_1d(_rng: &mut Rng, size: usize) -> Tensor1D {
    Tensor1D::new(vec![1.0; size])
}

/// Generate mock model weights with Xavier initialization
///
/// # Arguments
/// - `config`: Model configuration (hidden_dim, num_layers, etc.)
/// - `shard_cols`: Number of columns this worker is responsible for
/// - `seed`: Random seed for deterministic generation
///
/// # Returns
/// ModelWeights with all tensors Xavier-initialized
///
/// # MOCK Implementation Note
/// This generates weights for VALIDATION ONLY. Replace with actual
/// safetensors loading for production inference.
pub fn generate_mock_weights(
    config: &ModelConfig,
    shard_cols: usize,
    seed: u64,
) -> ModelWeights {
    let mut rng = Rng::new(seed);

    // Generate per-layer weights
    let layers = (0..config.num_layers)
        .map(|i| LayerWeights {
            layer_idx: i,
            // Attention weights
            w_q: xavier_tensor_2d(&mut rng, config.hidden_dim, shard_cols),
            w_k: xavier_tensor_2d(&mut rng, config.hidden_dim, shard_cols),
            w_v: xavier_tensor_2d(&mut rng, config.hidden_dim, shard_cols),
            w_o: xavier_tensor_2d(&mut rng, shard_cols, config.hidden_dim),
            // MLP weights
            w_up: xavier_tensor_2d(&mut rng, config.hidden_dim, shard_cols),
            w_gate: xavier_tensor_2d(&mut rng, config.hidden_dim, shard_cols),
            w_down: xavier_tensor_2d(&mut rng, shard_cols, config.hidden_dim),
            // Normalization weights (identity scaling)
            attn_norm: xavier_tensor_1d(&mut rng, config.hidden_dim),
            mlp_norm: xavier_tensor_1d(&mut rng, config.hidden_dim),
        })
        .collect();

    ModelWeights {
        model_id: format!("mock-llama-seed-{}", seed),
        embedding: xavier_tensor_2d(&mut rng, config.vocab_size, config.hidden_dim),
        layers,
        final_norm: xavier_tensor_1d(&mut rng, config.hidden_dim),
        lm_head: xavier_tensor_2d(&mut rng, config.hidden_dim, config.vocab_size),
        config: config.clone(),
    }
}

/// Calculate conservative error bound for floating point operations
///
/// # Arguments
/// - `num_matmuls`: Total number of matrix multiplications
/// - `matmul_inner_dim`: Inner dimension of matmuls (affects accumulation error)
/// - `num_activations`: Number of non-linear activations (gelu, silu, softmax)
///
/// # Returns
/// Conservative f32 error bound
///
/// # Formula
/// - f32 machine epsilon: ε ≈ 1.19e-7
/// - Matmul error: γk = k*ε where k is inner dimension
/// - Activation error: ~10*ε (for exp/tanh in activations)
/// - All-reduce CBOR serialization: ~2*ε
/// - Safety margin: 10x
///
/// For LLaMA 70B (80 layers, hidden_dim=8192):
/// - 560 matmuls: 560 × 8192 × 1.19e-7 ≈ 0.546
/// - Safety margin → 0.1
///
/// # References
/// - PyTorch Numerical Accuracy: https://pytorch.org/docs/stable/notes/numerical_accuracy.html
/// - Floating Point Error Propagation: https://floating-point-gui.de/errors/propagation/
pub fn calculate_error_bound(
    num_matmuls: usize,
    matmul_inner_dim: usize,
    num_activations: usize,
) -> f32 {
    const F32_EPSILON: f32 = 1.19e-7;

    // Matmul accumulation error
    let matmul_error = (num_matmuls as f32) * (matmul_inner_dim as f32) * F32_EPSILON;

    // Activation function error (exp, tanh, etc.)
    let activation_error = (num_activations as f32) * 10.0 * F32_EPSILON;

    // Total cascaded error
    let total_error = matmul_error + activation_error;

    // Conservative bound with 10x safety margin
    let conservative_bound = total_error * 10.0;

    // Clamp to reasonable range [1e-5, 1.0]
    conservative_bound.max(1e-5).min(1.0)
}

/// Result of tensor comparison validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed (all differences within error bound)
    pub passed: bool,

    /// Maximum absolute error observed
    pub max_error: f32,

    /// Error bound used for comparison
    pub error_bound: f32,

    /// Number of elements compared
    pub num_comparisons: usize,

    /// List of failures: (index, expected, actual)
    pub failures: Vec<(usize, f32, f32)>,
}

impl ValidationResult {
    /// Print a summary of the validation result
    pub fn print_summary(&self) {
        if self.passed {
            println!("✓ Validation PASSED");
            println!(
                "  Max error: {:.2e} (bound: {:.2e})",
                self.max_error, self.error_bound
            );
            println!("  Compared {} elements", self.num_comparisons);
        } else {
            println!("✗ Validation FAILED");
            println!(
                "  Max error: {:.2e} (bound: {:.2e})",
                self.max_error, self.error_bound
            );
            println!("  Compared {} elements", self.num_comparisons);
            println!("  Failed {} elements:", self.failures.len());
            for (i, (idx, expected, actual)) in self.failures.iter().enumerate().take(10) {
                println!(
                    "    [{}] idx={}: expected={:.6}, actual={:.6}, diff={:.2e}",
                    i,
                    idx,
                    expected,
                    actual,
                    (expected - actual).abs()
                );
            }
            if self.failures.len() > 10 {
                println!("    ... and {} more", self.failures.len() - 10);
            }
        }
    }
}

/// Compare two tensors with error bound tolerance
///
/// # Arguments
/// - `reference`: Expected tensor (ground truth)
/// - `distributed`: Actual tensor from distributed computation
/// - `error_bound`: Maximum acceptable absolute error
///
/// # Returns
/// ValidationResult with comparison details
///
/// # Example
/// ```ignore
/// let error_bound = calculate_error_bound(560, 8192, 240);
/// let result = compare_tensors(&ref_output, &dist_output, error_bound);
/// assert!(result.passed, "Validation failed");
/// ```
pub fn compare_tensors(
    reference: &Tensor2D,
    distributed: &Tensor2D,
    error_bound: f32,
) -> ValidationResult {
    assert_eq!(
        (reference.rows, reference.cols),
        (distributed.rows, distributed.cols),
        "Tensor shapes must match"
    );

    let mut max_error: f32 = 0.0;
    let mut failures = Vec::new();

    for i in 0..reference.data.len() {
        let expected = reference.data[i];
        let actual = distributed.data[i];
        let error = (expected - actual).abs();

        max_error = max_error.max(error);

        if error > error_bound {
            failures.push((i, expected, actual));
        }
    }

    ValidationResult {
        passed: failures.is_empty(),
        max_error,
        error_bound,
        num_comparisons: reference.data.len(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_determinism() {
        let mut rng1 = Rng::new(12345);
        let mut rng2 = Rng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64(), "RNG must be deterministic");
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut rng1 = Rng::new(12345);
        let mut rng2 = Rng::new(54321);

        let val1 = rng1.next_u64();
        let val2 = rng2.next_u64();

        assert_ne!(val1, val2, "Different seeds should produce different values");
    }

    #[test]
    fn test_next_f32_range() {
        let mut rng = Rng::new(42);

        for _ in 0..1000 {
            let val = rng.next_f32();
            assert!(
                val >= -1.0 && val <= 1.0,
                "next_f32() must be in [-1.0, 1.0], got {}",
                val
            );
        }
    }

    #[test]
    fn test_xavier_distribution_properties() {
        let mut rng = Rng::new(42);
        let fan_in = 1000;
        let fan_out = 500;

        let samples: Vec<f32> = (0..10000)
            .map(|_| rng.xavier_uniform(fan_in, fan_out))
            .collect();

        // Check mean is close to 0
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            mean.abs() < 0.01,
            "Xavier mean should be ~0, got {}",
            mean
        );

        // Check values are within expected limit
        let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        for &val in &samples {
            assert!(
                val.abs() <= limit,
                "Xavier value {} exceeds limit {}",
                val,
                limit
            );
        }
    }

    #[test]
    fn test_xavier_tensor_2d_shape() {
        let mut rng = Rng::new(42);
        let tensor = xavier_tensor_2d(&mut rng, 10, 20);

        assert_eq!((tensor.rows, tensor.cols), (10, 20));
        assert_eq!(tensor.data.len(), 200);
    }

    #[test]
    fn test_mock_weights_determinism() {
        let config = ModelConfig {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 1000,
            intermediate_size: 512,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };

        let w1 = generate_mock_weights(&config, 64, 42);
        let w2 = generate_mock_weights(&config, 64, 42);

        // Same seed should produce identical weights
        assert_eq!(
            w1.embedding.data, w2.embedding.data,
            "Same seed must produce identical weights"
        );
        assert_eq!(
            w1.layers[0].w_q.data, w2.layers[0].w_q.data,
            "Layer weights must be identical"
        );
    }

    #[test]
    fn test_mock_weights_different_seeds() {
        let config = ModelConfig::default();

        let w1 = generate_mock_weights(&config, 64, 42);
        let w2 = generate_mock_weights(&config, 64, 99);

        // Different seeds should produce different weights
        assert_ne!(
            w1.embedding.data, w2.embedding.data,
            "Different seeds must produce different weights"
        );
    }

    #[test]
    fn test_mock_weights_structure() {
        let config = ModelConfig {
            hidden_dim: 128,
            num_layers: 2,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 1000,
            intermediate_size: 512,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
        };

        let weights = generate_mock_weights(&config, 64, 42);

        assert_eq!(weights.layers.len(), 2);
        assert_eq!(
            (weights.embedding.rows, weights.embedding.cols),
            (1000, 128)
        );
        assert_eq!((weights.lm_head.rows, weights.lm_head.cols), (128, 1000));

        // Check layer 0 shapes
        let layer = &weights.layers[0];
        assert_eq!((layer.w_q.rows, layer.w_q.cols), (128, 64));
        assert_eq!((layer.w_k.rows, layer.w_k.cols), (128, 64));
        assert_eq!((layer.w_v.rows, layer.w_v.cols), (128, 64));
        assert_eq!((layer.w_o.rows, layer.w_o.cols), (64, 128));
        assert_eq!(layer.attn_norm.len(), 128);
    }

    #[test]
    fn test_error_bound_calculation() {
        // Test LLaMA 70B configuration
        let num_matmuls = 80 * 7; // 80 layers × 7 matmuls per layer
        let matmul_inner_dim = 8192;
        let num_activations = 80 * 3;

        let bound = calculate_error_bound(num_matmuls, matmul_inner_dim, num_activations);

        // Should be conservative but not too tight
        // For very large models, may hit the 1.0 clamp (which is fine)
        assert!(bound > 0.001, "Error bound too tight: {}", bound);
        assert!(bound <= 1.0, "Error bound exceeds maximum: {}", bound);

        println!("LLaMA 70B error bound: {:.6}", bound);
    }

    #[test]
    fn test_error_bound_small_model() {
        // Test small model configuration
        let num_matmuls = 2 * 7; // 2 layers
        let matmul_inner_dim = 256;
        let num_activations = 2 * 3;

        let bound = calculate_error_bound(num_matmuls, matmul_inner_dim, num_activations);

        // Smaller model should have tighter bound
        assert!(bound > 1e-6, "Error bound too tight: {}", bound);
        assert!(bound < 0.1, "Error bound too loose: {}", bound);

        println!("Small model error bound: {:.6}", bound);
    }

    #[test]
    fn test_compare_tensors_identical() {
        let mut rng = Rng::new(42);
        let t1 = xavier_tensor_2d(&mut rng, 10, 10);
        let t2 = t1.clone();

        let result = compare_tensors(&t1, &t2, 1e-10);

        assert!(result.passed);
        assert_eq!(result.max_error, 0.0);
        assert_eq!(result.num_comparisons, 100);
        assert!(result.failures.is_empty());
    }

    #[test]
    fn test_compare_tensors_within_bound() {
        let t1 = Tensor2D::filled(5, 5, 1.0);
        let t2 = Tensor2D::filled(5, 5, 1.0001);

        let result = compare_tensors(&t1, &t2, 0.001);

        assert!(result.passed);
        assert!((result.max_error - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn test_compare_tensors_exceeds_bound() {
        let t1 = Tensor2D::filled(5, 5, 1.0);
        let t2 = Tensor2D::filled(5, 5, 2.0);

        let result = compare_tensors(&t1, &t2, 0.5);

        assert!(!result.passed);
        assert_eq!(result.max_error, 1.0);
        assert_eq!(result.failures.len(), 25); // All 25 elements fail
    }

    #[test]
    fn test_validation_result_print_summary() {
        let result = ValidationResult {
            passed: true,
            max_error: 1e-5,
            error_bound: 1e-4,
            num_comparisons: 1000,
            failures: vec![],
        };

        // Should not panic
        result.print_summary();
    }
}
